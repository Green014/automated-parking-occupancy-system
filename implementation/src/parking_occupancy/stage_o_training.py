from __future__ import annotations

import configparser
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .stage_n_lmot import LmotAnnotation, parse_lmot_gt, sha256_file
from .stage_n_lmot_v2 import (
    ConcatenatedPartReader,
    _safe_member_path,
    discover_split_tar_parts,
)
from .stage_o_low_light import (
    STAGE_O_PROTOCOL_ID,
    StageOProtocolError,
    canonical_records_sha256,
    load_stage_o_protocol,
)


D1_SHA256 = "0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64"
MOTOR_CLASS_IDS = frozenset({3, 4, 5, 6})


def deterministic_frame_numbers(
    *, first: int = 1, stride: int = 10, last: int = 1210
) -> tuple[int, ...]:
    if first <= 0 or stride <= 0 or last < first:
        raise ValueError("invalid deterministic sampling rule")
    return tuple(range(first, last + 1, stride))


def _copy_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = archive.extractfile(member)
    if source is None:
        raise StageOProtocolError(f"Could not read tar member {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def extract_train_annotations(
    *,
    annotations_tar: Path,
    sequence_ids: Sequence[str],
    output_root: Path,
) -> dict[str, Any]:
    accepted = {sequence: set() for sequence in sequence_ids}
    with tarfile.open(annotations_tar, "r:*") as archive:
        for member in archive:
            path = _safe_member_path(member.name)
            if not member.isfile():
                continue
            if (
                len(path.parts) < 4
                or path.parts[0:2] != ("LMOT_annotations", "train")
                or path.parts[2] not in accepted
            ):
                continue
            sequence = path.parts[2]
            if path.parts[3:] == ("seqinfo.ini",):
                relative = Path(sequence) / "seqinfo.ini"
                accepted[sequence].add("seqinfo.ini")
            elif path.parts[3:] == ("gt", "gt.txt"):
                relative = Path(sequence) / "gt" / "gt.txt"
                accepted[sequence].add("gt/gt.txt")
            else:
                continue
            _copy_member(archive, member, output_root / relative)
    incomplete = {
        key: sorted({"seqinfo.ini", "gt/gt.txt"} - values)
        for key, values in accepted.items()
        if values != {"seqinfo.ini", "gt/gt.txt"}
    }
    if incomplete:
        raise StageOProtocolError(
            f"LMOT train annotations incomplete: {incomplete}"
        )
    return {
        "sequences": list(sequence_ids),
        "files": 2 * len(sequence_ids),
        "source_bytes": annotations_tar.stat().st_size,
        "source_sha256": sha256_file(annotations_tar),
    }


def extract_sampled_rgb(
    *,
    parts: Sequence[Path],
    archive_root: str,
    image_directory: str,
    sequence_ids: Sequence[str],
    frame_numbers: Sequence[int],
    output_root: Path,
) -> dict[str, Any]:
    selected_sequences = set(sequence_ids)
    selected_frames = set(frame_numbers)
    seen: dict[str, set[int]] = {
        sequence: set() for sequence in sequence_ids
    }
    extracted_bytes = 0
    reader = io.BufferedReader(
        ConcatenatedPartReader(parts), buffer_size=1024 * 1024
    )
    with reader, tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = _safe_member_path(member.name)
            if (
                len(path.parts) != 5
                or path.parts[0] != archive_root
                or path.parts[1] != "train"
                or path.parts[2] not in selected_sequences
                or path.parts[3] != image_directory
            ):
                continue
            try:
                frame_number = int(PurePosixPath(path.parts[4]).stem)
            except ValueError as exc:
                raise StageOProtocolError(
                    f"Non-numeric LMOT member: {member.name}"
                ) from exc
            if frame_number not in selected_frames:
                continue
            sequence = path.parts[2]
            if frame_number in seen[sequence]:
                raise StageOProtocolError(
                    f"Duplicate {image_directory} {sequence} {frame_number}"
                )
            destination = (
                output_root / sequence / image_directory / path.parts[4]
            )
            _copy_member(archive, member, destination)
            seen[sequence].add(frame_number)
            extracted_bytes += member.size
    missing = {
        sequence: sorted(selected_frames - frames)
        for sequence, frames in seen.items()
        if frames != selected_frames
    }
    if missing:
        raise StageOProtocolError(
            f"Sampled {image_directory} extraction incomplete: {missing}"
        )
    return {
        "image_directory": image_directory,
        "sequences": list(sequence_ids),
        "frames_per_sequence": len(frame_numbers),
        "extracted_files": sum(len(value) for value in seen.values()),
        "extracted_bytes": extracted_bytes,
        "parts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
    }


def _sequence_dimensions(seqinfo: Path) -> tuple[int, int, int]:
    parser = configparser.ConfigParser()
    parser.read(seqinfo, encoding="utf-8-sig")
    section = parser["Sequence"]
    return (
        section.getint("imWidth"),
        section.getint("imHeight"),
        section.getint("seqLength"),
    )


def annotations_to_yolo(
    rows: Iterable[LmotAnnotation],
    *,
    width: int,
    height: int,
) -> list[str]:
    labels = []
    for row in rows:
        if row.ignore != 1 or row.class_id not in MOTOR_CLASS_IDS:
            continue
        x1 = min(max(float(row.x), 0.0), float(width))
        y1 = min(max(float(row.y), 0.0), float(height))
        x2 = min(max(float(row.x + row.width), 0.0), float(width))
        y2 = min(max(float(row.y + row.height), 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_w = (x2 - x1) / width
        box_h = (y2 - y1) / height
        labels.append(
            f"0 {center_x:.8f} {center_y:.8f} {box_w:.8f} {box_h:.8f}"
        )
    return labels


def _hardlink(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _dataset_yaml(
    root: Path, *, train: str, val: str
) -> dict[str, Any]:
    return {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "names": {0: "vehicle"},
    }


def build_combined_yolo_dataset(
    *,
    extracted_lmot_root: Path,
    output_root: Path,
    declared_root: Path | None = None,
    training_sequences: Sequence[str],
    development_sequences: Sequence[str],
    ndispark_prepared_root: Path,
    frame_numbers: Sequence[int],
) -> dict[str, Any]:
    declared_root = (
        output_root if declared_root is None else declared_root.resolve()
    )
    records: list[dict[str, Any]] = []
    pair_groups: dict[str, str] = {}
    split_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }
    link_modes: Counter[str] = Counter()
    for split, sequences in (
        ("train", training_sequences),
        ("val", development_sequences),
    ):
        for sequence in sequences:
            root = extracted_lmot_root / sequence
            width, height, seq_length = _sequence_dimensions(
                root / "seqinfo.ini"
            )
            if seq_length != 1210:
                raise StageOProtocolError(
                    f"Unexpected LMOT train length: {sequence}={seq_length}"
                )
            by_frame: dict[int, list[LmotAnnotation]] = {}
            for row in parse_lmot_gt(root / "gt" / "gt.txt"):
                if row.frame_number in frame_numbers:
                    by_frame.setdefault(row.frame_number, []).append(row)
            for frame_number in frame_numbers:
                labels = annotations_to_yolo(
                    by_frame.get(frame_number, []),
                    width=width,
                    height=height,
                )
                group = f"{sequence}:{frame_number:06d}"
                if group in pair_groups and pair_groups[group] != split:
                    raise StageOProtocolError(
                        f"LMOT pair crosses splits: {group}"
                    )
                pair_groups[group] = split
                for illumination, directory, extension in (
                    ("light", "img_light_rgb", ".jpg"),
                    ("dark", "img_dark_rgb", ".png"),
                ):
                    source = root / directory / f"{frame_number:06d}{extension}"
                    name = (
                        f"lmot_{sequence}_{frame_number:06d}_{illumination}"
                        f"{extension}"
                    )
                    destination = output_root / "images" / split / name
                    link_modes[_hardlink(source, destination)] += 1
                    label_path = (
                        output_root
                        / "labels"
                        / split
                        / f"{Path(name).stem}.txt"
                    )
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text(
                        "\n".join(labels) + ("\n" if labels else ""),
                        encoding="utf-8",
                    )
                    split_counts[split]["lmot_images"] += 1
                    split_counts[split]["lmot_boxes"] += len(labels)
                    records.append(
                        {
                            "source": "LMOT",
                            "split": split,
                            "pair_group": group,
                            "sequence": sequence,
                            "frame_number": frame_number,
                            "illumination": illumination,
                            "image": destination.relative_to(
                                output_root
                            ).as_posix(),
                            "image_sha256": sha256_file(destination),
                            "labels": len(labels),
                        }
                    )

    source_images = ndispark_prepared_root / "images" / "train"
    source_labels = ndispark_prepared_root / "labels" / "train"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(
            f"NDISPark prepared train split missing: {ndispark_prepared_root}"
        )
    ndispark_images = sorted(
        path
        for path in source_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(ndispark_images) != 112:
        raise StageOProtocolError(
            f"Expected 112 NDISPark train images, got {len(ndispark_images)}"
        )
    for source in ndispark_images:
        name = f"ndispark_{source.name}"
        destination = output_root / "images" / "train" / name
        link_modes[_hardlink(source, destination)] += 1
        source_label = source_labels / f"{source.stem}.txt"
        destination_label = (
            output_root / "labels" / "train" / f"{Path(name).stem}.txt"
        )
        link_modes[_hardlink(source_label, destination_label)] += 1
        label_count = sum(
            bool(line.strip())
            for line in source_label.read_text(encoding="utf-8").splitlines()
        )
        split_counts["train"]["ndispark_images"] += 1
        split_counts["train"]["ndispark_boxes"] += label_count
        records.append(
            {
                "source": "NDISPark",
                "split": "train",
                "pair_group": f"NDISPark:{source.name}",
                "sequence": None,
                "frame_number": None,
                "illumination": "daylight",
                "image": destination.relative_to(output_root).as_posix(),
                "image_sha256": sha256_file(destination),
                "labels": label_count,
            }
        )

    dataset = _dataset_yaml(
        declared_root, train="images/train", val="images/val"
    )
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    smoke_train = [
        str((declared_root / row["image"]).resolve())
        for row in records
        if row["split"] == "train"
    ][:96]
    smoke_val = [
        str((declared_root / row["image"]).resolve())
        for row in records
        if row["split"] == "val"
    ][:48]
    (output_root / "smoke_train.txt").write_text(
        "\n".join(smoke_train) + "\n", encoding="utf-8"
    )
    (output_root / "smoke_val.txt").write_text(
        "\n".join(smoke_val) + "\n", encoding="utf-8"
    )
    smoke = _dataset_yaml(
        declared_root, train="smoke_train.txt", val="smoke_val.txt"
    )
    (output_root / "dataset_smoke.yaml").write_text(
        yaml.safe_dump(smoke, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "records": records,
        "records_sha256": canonical_records_sha256(records),
        "split_counts": {
            key: dict(value) for key, value in split_counts.items()
        },
        "pair_group_count": len(pair_groups),
        "cross_split_pair_count": 0,
        "link_modes": dict(link_modes),
        "dataset_yaml_sha256": sha256_file(output_root / "dataset.yaml"),
        "smoke_dataset_yaml_sha256": sha256_file(
            output_root / "dataset_smoke.yaml"
        ),
    }


def prepare_stage_o_training_data(
    *,
    protocol_path: Path,
    annotations_tar: Path,
    light_parts_dir: Path,
    dark_parts_dir: Path,
    ndispark_prepared_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    protocol = load_stage_o_protocol(protocol_path)
    sequences = tuple(protocol["data"]["lmot_train_sequences"])
    training_sequences = tuple(protocol["data"]["o3_training_sequences"])
    development_sequences = tuple(
        protocol["data"]["internal_development_sequences"]
    )
    if set(training_sequences) & set(development_sequences):
        raise StageOProtocolError("Training/development sequences overlap")
    sampling = protocol["data"]["deterministic_subsampling"]
    frames = deterministic_frame_numbers(
        first=int(sampling["first_frame"]),
        stride=int(sampling["stride"]),
        last=int(sampling["last_frame_inclusive"]),
    )
    partial = output_root.with_name(output_root.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Refusing to overwrite {partial}")
    partial.mkdir(parents=True)
    try:
        lmot_root = partial / "lmot_train_sampled"
        annotation_record = extract_train_annotations(
            annotations_tar=annotations_tar.resolve(),
            sequence_ids=sequences,
            output_root=lmot_root,
        )
        light_parts = discover_split_tar_parts(
            light_parts_dir, "LMOT_light_rgb_trainval.tar"
        )
        dark_parts = discover_split_tar_parts(
            dark_parts_dir, "LMOT_dark_rgb_trainval.tar"
        )
        light_record = extract_sampled_rgb(
            parts=light_parts,
            archive_root="LMOT_light_rgb_trainval",
            image_directory="img_light_rgb",
            sequence_ids=sequences,
            frame_numbers=frames,
            output_root=lmot_root,
        )
        dark_record = extract_sampled_rgb(
            parts=dark_parts,
            archive_root="LMOT_dark_rgb_trainval",
            image_directory="img_dark_rgb",
            sequence_ids=sequences,
            frame_numbers=frames,
            output_root=lmot_root,
        )
        dataset_root = partial / "yolo_combined"
        dataset_record = build_combined_yolo_dataset(
            extracted_lmot_root=lmot_root,
            output_root=dataset_root,
            declared_root=output_root / "yolo_combined",
            training_sequences=training_sequences,
            development_sequences=development_sequences,
            ndispark_prepared_root=ndispark_prepared_root.resolve(),
            frame_numbers=frames,
        )
        manifest = {
            "schema_version": 1,
            "protocol_id": STAGE_O_PROTOCOL_ID,
            "status": "complete_verified_training_data",
            "source_files_modified": False,
            "paired_group_rule": "sequence_id_plus_frame_number",
            "cross_split_pair_count": 0,
            "sampling": {
                "first": frames[0],
                "stride": int(sampling["stride"]),
                "last": frames[-1],
                "frames_per_sequence": len(frames),
            },
            "splits": {
                "training_sequences": list(training_sequences),
                "development_sequences": list(development_sequences),
            },
            "annotations": annotation_record,
            "light_rgb": light_record,
            "dark_rgb": dark_record,
            "dataset": {
                key: value
                for key, value in dataset_record.items()
                if key != "records"
            },
            "records": dataset_record["records"],
        }
        (partial / "training_manifest.yaml").write_text(
            yaml.safe_dump(
                manifest, sort_keys=False, allow_unicode=True, width=120
            ),
            encoding="utf-8",
        )
        os.replace(partial, output_root)
    except Exception:
        # The partial directory is intentionally retained as failure evidence.
        raise
    return {
        "output_root": str(output_root),
        "training_manifest": str(output_root / "training_manifest.yaml"),
        "dataset_yaml": str(output_root / "yolo_combined" / "dataset.yaml"),
        "smoke_dataset_yaml": str(
            output_root / "yolo_combined" / "dataset_smoke.yaml"
        ),
        "training_manifest_sha256": sha256_file(
            output_root / "training_manifest.yaml"
        ),
    }


def _finite_results_csv(path: Path) -> bool:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    for row in rows:
        for value in row.values():
            if value is None or value.strip() == "":
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if not math.isfinite(number):
                return False
    return True


def run_stage_o_training(
    *,
    protocol_path: Path,
    data_yaml: Path,
    training_manifest: Path,
    initial_weights: Path,
    output_dir: Path,
    smoke: bool,
    device: str = "0",
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    protocol = load_stage_o_protocol(protocol_path)
    settings = protocol["methods"]["O3"]["training"]
    initial_weights = initial_weights.resolve()
    if sha256_file(initial_weights) != D1_SHA256:
        raise StageOProtocolError("O3 must initialize from the frozen D1 best")
    manifest_payload = yaml.safe_load(
        training_manifest.read_text(encoding="utf-8")
    )
    if (
        manifest_payload.get("status") != "complete_verified_training_data"
        or manifest_payload.get("cross_split_pair_count") != 0
    ):
        raise StageOProtocolError("O3 training manifest did not pass")
    if smoke and not data_yaml.name.startswith("dataset_smoke"):
        raise StageOProtocolError(
            "Smoke training requires an explicitly named dataset_smoke YAML"
        )
    if not smoke and (
        not data_yaml.name.startswith("dataset")
        or data_yaml.name.startswith("dataset_smoke")
    ):
        raise StageOProtocolError(
            "Formal training requires a non-smoke dataset YAML"
        )

    from ultralytics import YOLO
    import torch

    if torch.cuda.is_available() and device.lower() not in {"cpu", "mps"}:
        torch.cuda.reset_peak_memory_stats()
        measure_cuda = True
    else:
        measure_cuda = False
    model = YOLO(str(initial_weights))
    started = time.perf_counter()
    phase = "smoke" if smoke else "formal"
    try:
        result = model.train(
            data=str(data_yaml.resolve()),
            epochs=(
                int(settings["smoke_epochs"])
                if smoke
                else int(settings["formal_max_epochs"])
            ),
            patience=int(settings["early_stopping_patience"]),
            imgsz=int(settings["imgsz"]),
            batch=int(settings["physical_batch"]),
            nbs=int(settings["nominal_batch"]),
            seed=int(settings["seed"]),
            deterministic=bool(settings["deterministic"]),
            amp=bool(settings["amp"]),
            device=device,
            workers=int(settings["workers"]),
            optimizer=str(settings["optimizer"]),
            lr0=float(settings["lr0"]),
            lrf=float(settings["lrf"]),
            momentum=float(settings["momentum"]),
            weight_decay=float(settings["weight_decay"]),
            warmup_epochs=float(settings["warmup_epochs"]),
            close_mosaic=int(settings["close_mosaic"]),
            pretrained=True,
            resume=False,
            cache=False,
            val=True,
            plots=True,
            save=True,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=False,
            verbose=True,
        )
    except Exception as exc:
        if output_dir.exists():
            (output_dir / "training_failure.json").write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    elapsed = time.perf_counter() - started
    if measure_cuda:
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = None
        peak_reserved = None
    best = output_dir / "weights" / "best.pt"
    last = output_dir / "weights" / "last.pt"
    results_csv = output_dir / "results.csv"
    if not best.is_file() or not last.is_file() or not results_csv.is_file():
        raise StageOProtocolError("Ultralytics did not retain O3 artifacts")
    if not _finite_results_csv(results_csv):
        raise StageOProtocolError("O3 results.csv is empty or non-finite")
    shutil.copy2(protocol_path, output_dir / "config_snapshot.yaml")
    shutil.copy2(training_manifest, output_dir / "training_manifest.yaml")
    metrics = {
        key: float(value)
        for key, value in getattr(result, "results_dict", {}).items()
        if isinstance(value, (int, float, np.number))
    }
    summary = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "phase": phase,
        "training_performed": True,
        "smoke": smoke,
        "formal_run": not smoke,
        "hyperparameter_search_performed": False,
        "consistency_loss_used": False,
        "initialization": {
            "path": str(initial_weights),
            "bytes": initial_weights.stat().st_size,
            "sha256": D1_SHA256,
        },
        "checkpoints": {
            "best": {
                "path": str(best),
                "bytes": best.stat().st_size,
                "sha256": sha256_file(best),
            },
            "last": {
                "path": str(last),
                "bytes": last.stat().st_size,
                "sha256": sha256_file(last),
            },
        },
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "device": (
            torch.cuda.get_device_name(0)
            if measure_cuda
            else device
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
