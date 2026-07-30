from __future__ import annotations

import csv
import json
import re
import shutil
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .stage_n_lmot import sha256_file
from .stage_q_external import (
    OccupancyTruthRecord,
    StageQDataGateError,
    parse_groundtruth_file,
)
from .slots import SlotMap, slot_map_from_dict


STAGE_Q_V2_PROTOCOL_ID = (
    "STAGE-Q-V2-UPM-GTI-EXTERNAL-NIGHT-OCCUPANCY-20260729-01"
)
EXPECTED_ARCHIVE_ROOT = "test"
EXPECTED_SLOT_COUNT = 21
UPM_SLOT_IDS = tuple(f"slot_{index:02d}" for index in range(21))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
NIGHT_AUXILIARY_LUMINANCE_THRESHOLD = 70.0


class StageQV2DataError(StageQDataGateError):
    """Raised when the UPM-GTI archive or extracted data fail a frozen gate."""


@dataclass(frozen=True, slots=True)
class SequenceInventory:
    sequence_id: str
    image_count: int
    truth_count: int
    image_truth_one_to_one: bool
    images_without_truth: int
    truth_without_images: int
    vector_length: int
    binary_vectors_only: bool
    occupied_labels: int
    vacant_labels: int
    transition_frames: int
    slot_state_changes: int
    file_names_naturally_sortable: bool
    image_formats: tuple[str, ...]
    resolutions: tuple[str, ...]
    sample_names: tuple[str, ...]
    sample_luminance: tuple[float, ...]
    median_sample_luminance: float
    auxiliary_low_light_candidate: bool
    timestamp_available: bool
    reliable_fps_available: bool


@dataclass(frozen=True, slots=True)
class NightManifestRow:
    sequence_id: str
    frame_index: int
    file_name: str
    relative_path: str
    bytes: int
    sha256: str
    width: int
    height: int
    mean_luminance: float
    source_occupancy_vector: str
    occupied_count: int
    vacant_count: int


def _safe_member_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise StageQV2DataError(f"Unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StageQV2DataError(f"Unsafe ZIP member path: {name!r}")
    return path


def inspect_zip_archive(
    archive_path: Path,
    *,
    expected_root: str = EXPECTED_ARCHIVE_ROOT,
    check_crc: bool = True,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos:
                raise StageQV2DataError("Archive is empty")
            roots: set[str] = set()
            normalized_seen: set[str] = set()
            file_count = 0
            directory_count = 0
            compressed_bytes = 0
            uncompressed_bytes = 0
            for info in infos:
                member = _safe_member_name(info.filename.rstrip("/"))
                roots.add(member.parts[0])
                normalized = member.as_posix().casefold()
                if normalized in normalized_seen:
                    raise StageQV2DataError(
                        f"Case-insensitive duplicate ZIP path: {info.filename}"
                    )
                normalized_seen.add(normalized)
                unix_mode = info.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise StageQV2DataError(
                        f"Symbolic links are prohibited: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise StageQV2DataError(
                        f"Encrypted ZIP member is prohibited: {info.filename}"
                    )
                if info.is_dir():
                    directory_count += 1
                else:
                    file_count += 1
                    compressed_bytes += int(info.compress_size)
                    uncompressed_bytes += int(info.file_size)
            if roots != {expected_root}:
                raise StageQV2DataError(
                    f"Unexpected archive roots: {sorted(roots)}"
                )
            bad_crc_member = archive.testzip() if check_crc else None
            if bad_crc_member is not None:
                raise StageQV2DataError(
                    f"ZIP CRC failure: {bad_crc_member}"
                )
    except zipfile.BadZipFile as exc:
        raise StageQV2DataError(f"Invalid ZIP archive: {archive_path}") from exc
    return {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "zip_readable": True,
        "crc_verified": check_crc,
        "path_traversal_safe": True,
        "symlinks_present": False,
        "encrypted_members_present": False,
        "entry_count": len(infos),
        "file_count": file_count,
        "directory_count": directory_count,
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "archive_roots": sorted(roots),
    }


def extract_zip_safely(
    archive_path: Path,
    output_parent: Path,
    *,
    expected_root: str = EXPECTED_ARCHIVE_ROOT,
) -> Path:
    audit = inspect_zip_archive(
        archive_path,
        expected_root=expected_root,
        check_crc=True,
    )
    output_parent = output_parent.resolve()
    final_root = output_parent / expected_root
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite {final_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path.resolve()) as archive:
        for info in archive.infolist():
            member = _safe_member_name(info.filename.rstrip("/"))
            target = (output_parent / Path(*member.parts)).resolve()
            try:
                target.relative_to(output_parent)
            except ValueError as exc:
                raise StageQV2DataError(
                    f"ZIP member escapes extraction root: {info.filename}"
                ) from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
    if not final_root.is_dir():
        raise StageQV2DataError(
            f"Expected extracted root was not created: {final_root}"
        )
    if audit["file_count"] <= 0:
        raise StageQV2DataError("No files were extracted")
    return final_root


def natural_sort_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(token) if token.isdigit() else token.casefold()
        for token in re.split(r"(\d+)", value)
    )


def _sample_indices(count: int, sample_count: int = 5) -> tuple[int, ...]:
    if count <= 0:
        return ()
    return tuple(
        sorted(
            {
                int(round(index * (count - 1) / max(sample_count - 1, 1)))
                for index in range(min(sample_count, count))
            }
        )
    )


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise StageQV2DataError(f"Could not decode image: {path}")
    return image


def _luminance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _transition_counts(
    records: Sequence[OccupancyTruthRecord],
) -> tuple[int, int]:
    transition_frames = 0
    slot_state_changes = 0
    for previous, current in zip(records, records[1:]):
        changes = sum(
            left is not None and right is not None and left != right
            for left, right in zip(
                previous.project_states,
                current.project_states,
            )
        )
        if changes:
            transition_frames += 1
            slot_state_changes += changes
    return transition_frames, slot_state_changes


def inspect_sequence(sequence_root: Path) -> SequenceInventory:
    sequence_root = sequence_root.resolve()
    truth_path = sequence_root / "groundtruth.txt"
    images_root = sequence_root / "images"
    if not truth_path.is_file() or not images_root.is_dir():
        raise StageQV2DataError(
            f"Incomplete sequence structure: {sequence_root}"
        )
    image_paths = sorted(
        (
            path
            for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: natural_sort_key(path.name),
    )
    if not image_paths:
        raise StageQV2DataError(f"No images in {images_root}")
    raw_records = parse_groundtruth_file(truth_path)
    by_name = {record.image_name: record for record in raw_records}
    image_names = {path.name for path in image_paths}
    truth_names = set(by_name)
    images_without_truth = sorted(image_names - truth_names)
    truth_without_images = sorted(truth_names - image_names)
    matched_paths = [path for path in image_paths if path.name in truth_names]
    records = [by_name[path.name] for path in matched_paths]
    if not records:
        raise StageQV2DataError(
            f"No image/truth matches in sequence: {sequence_root}"
        )
    all_values = [
        value
        for record in records
        for value in record.source_values
    ]
    if any(value is None for value in all_values):
        binary_only = False
    else:
        binary_only = set(all_values) <= {0, 1}
    transition_frames, slot_state_changes = _transition_counts(records)
    formats = tuple(sorted({path.suffix.lower() for path in image_paths}))
    resolutions: set[str] = set()
    sample_names: list[str] = []
    sample_luminance: list[float] = []
    sample_set = set(_sample_indices(len(image_paths)))
    for index, path in enumerate(image_paths):
        image = _read_image(path)
        height, width = image.shape[:2]
        resolutions.add(f"{width}x{height}")
        if index in sample_set:
            sample_names.append(path.name)
            sample_luminance.append(_luminance(image))
    median_luminance = float(np.median(sample_luminance))
    return SequenceInventory(
        sequence_id=sequence_root.name,
        image_count=len(image_paths),
        truth_count=len(raw_records),
        image_truth_one_to_one=(
            not images_without_truth and not truth_without_images
        ),
        images_without_truth=len(images_without_truth),
        truth_without_images=len(truth_without_images),
        vector_length=EXPECTED_SLOT_COUNT,
        binary_vectors_only=binary_only,
        occupied_labels=sum(value == 0 for value in all_values),
        vacant_labels=sum(value == 1 for value in all_values),
        transition_frames=transition_frames,
        slot_state_changes=slot_state_changes,
        file_names_naturally_sortable=(
            len({path.name for path in image_paths}) == len(image_paths)
        ),
        image_formats=formats,
        resolutions=tuple(sorted(resolutions)),
        sample_names=tuple(sample_names),
        sample_luminance=tuple(sample_luminance),
        median_sample_luminance=median_luminance,
        auxiliary_low_light_candidate=(
            median_luminance <= NIGHT_AUXILIARY_LUMINANCE_THRESHOLD
        ),
        timestamp_available=False,
        reliable_fps_available=False,
    )


def inspect_test_split(test_root: Path) -> list[SequenceInventory]:
    test_root = test_root.resolve()
    sequence_roots = sorted(
        (
            path
            for path in test_root.iterdir()
            if path.is_dir() and path.name.casefold().startswith("gopro")
        ),
        key=lambda path: natural_sort_key(path.name),
    )
    if not sequence_roots:
        raise StageQV2DataError(f"No gopro sequences under {test_root}")
    return [inspect_sequence(path) for path in sequence_roots]


def write_sequence_inventory(
    path: Path,
    inventories: Sequence[SequenceInventory],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not inventories:
        raise StageQV2DataError("Cannot write an empty sequence inventory")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in inventories:
        row = {
            field: (
                "|".join(str(value) for value in value)
                if isinstance(value, tuple)
                else value
            )
            for field, value in asdict(item).items()
        }
        rows.append(row)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def render_sequence_contact_sheet(
    *,
    test_root: Path,
    inventories: Sequence[SequenceInventory],
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    tile_width, tile_height = 300, 180
    label_height = 45
    samples_per_row = 5
    canvas = np.full(
        (
            len(inventories) * (tile_height + label_height),
            samples_per_row * tile_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )
    for row_index, inventory in enumerate(inventories):
        y0 = row_index * (tile_height + label_height)
        for column, name in enumerate(inventory.sample_names):
            image = _read_image(
                test_root / inventory.sequence_id / "images" / name
            )
            thumbnail = cv2.resize(
                image,
                (tile_width, tile_height),
                interpolation=cv2.INTER_AREA,
            )
            x0 = column * tile_width
            canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = thumbnail
            cv2.putText(
                canvas,
                f"{name} Y={inventory.sample_luminance[column]:.1f}",
                (x0 + 5, y0 + tile_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        label = (
            f"{inventory.sequence_id}: n={inventory.image_count}, "
            f"median Y={inventory.median_sample_luminance:.1f}, "
            f"transition frames={inventory.transition_frames}, "
            f"slot changes={inventory.slot_state_changes}"
        )
        cv2.putText(
            canvas,
            label,
            (8, y0 + tile_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imencode(".png", canvas)[0]:
        raise StageQV2DataError("Could not encode sequence contact sheet")
    encoded = cv2.imencode(".png", canvas)[1]
    encoded.tofile(output_path)


def validate_multi_sequence_polygon_isolation(
    selected_sequences: Sequence[str],
    polygon_binding: Mapping[str, str],
) -> dict[str, Any]:
    selected = tuple(selected_sequences)
    if not selected:
        raise StageQV2DataError("No selected sequences")
    if set(polygon_binding) != set(selected):
        raise StageQV2DataError(
            "Every selected sequence must have an explicit polygon binding"
        )
    if any(not str(value).strip() for value in polygon_binding.values()):
        raise StageQV2DataError("Empty polygon binding")
    return {
        "selected_sequence_count": len(selected),
        "polygon_binding_count": len(polygon_binding),
        "isolated": True,
    }


def validate_upm_slot_map(payload: Mapping[str, Any]) -> SlotMap:
    slot_map = slot_map_from_dict(dict(payload))
    actual_ids = tuple(slot.slot_id for slot in slot_map.slots)
    if actual_ids != UPM_SLOT_IDS:
        raise StageQV2DataError(
            "UPM polygon IDs/order must be slot_00 through slot_20"
        )
    for slot in slot_map.slots:
        for x, y in slot.points:
            if not (
                0.0 <= x < slot_map.source_width
                and 0.0 <= y < slot_map.source_height
            ):
                raise StageQV2DataError(
                    f"Polygon coordinate out of range for {slot.slot_id}"
                )
    return slot_map


def write_occupancy_truth_from_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise StageQV2DataError("Empty Stage Q-v2 test manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    unknown_excluded = 0
    for row in manifest_rows:
        vector = str(row["source_occupancy_vector"])
        if len(vector) != EXPECTED_SLOT_COUNT or set(vector) - {"0", "1", "?"}:
            raise StageQV2DataError(
                f"Invalid manifest occupancy vector: {vector!r}"
            )
        for slot_id, source_value in zip(UPM_SLOT_IDS, vector):
            if source_value == "?":
                unknown_excluded += 1
                continue
            output_rows.append(
                {
                    "video_id": row["sequence_id"],
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": "",
                    "slot_id": slot_id,
                    "state": 1 if source_value == "0" else 0,
                }
            )
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "input_images": len(manifest_rows),
        "truth_rows": len(output_rows),
        "unknown_excluded": unknown_excluded,
        "slot_ids": list(UPM_SLOT_IDS),
        "timestamp_semantics": "unavailable_sequence_index_only",
    }


def render_polygon_validation(
    *,
    image_path: Path,
    slot_map: SlotMap,
    output_path: Path,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    image = _read_image(image_path)
    height, width = image.shape[:2]
    if (width, height) != (slot_map.source_width, slot_map.source_height):
        raise StageQV2DataError("Polygon source dimensions do not match image")
    canvas = image.copy()
    for slot in slot_map.slots:
        contour = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [contour], True, (0, 255, 255), 2)
        center = np.mean(np.asarray(slot.points), axis=0)
        cv2.putText(
            canvas,
            slot.slot_id.removeprefix("slot_"),
            (int(center[0]) - 8, int(center[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise StageQV2DataError("Could not encode polygon validation image")
    encoded.tofile(output_path)


def build_night_test_manifest(
    test_root: Path,
    inventories: Sequence[SequenceInventory],
    *,
    luminance_threshold: float = NIGHT_AUXILIARY_LUMINANCE_THRESHOLD,
) -> tuple[list[NightManifestRow], list[dict[str, Any]]]:
    if luminance_threshold != NIGHT_AUXILIARY_LUMINANCE_THRESHOLD:
        raise StageQV2DataError(
            "Night luminance threshold differs from the predeclared value"
        )
    test_root = test_root.resolve()
    rows: list[NightManifestRow] = []
    decisions: list[dict[str, Any]] = []
    for inventory in inventories:
        sequence_root = test_root / inventory.sequence_id
        records = parse_groundtruth_file(sequence_root / "groundtruth.txt")
        truth_by_name = {record.image_name: record for record in records}
        image_paths = sorted(
            (
                path
                for path in (sequence_root / "images").iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: natural_sort_key(path.name),
        )
        reasons: list[str] = []
        if not inventory.image_truth_one_to_one:
            reasons.append("raw_sequence_image_truth_not_one_to_one")
        candidate_rows: list[NightManifestRow] = []
        for frame_index, path in enumerate(image_paths):
            truth = truth_by_name.get(path.name)
            if truth is None:
                continue
            image = _read_image(path)
            mean_luminance = _luminance(image)
            if mean_luminance > luminance_threshold:
                continue
            height, width = image.shape[:2]
            values = tuple(int(value) for value in truth.source_values)
            candidate_rows.append(
                NightManifestRow(
                    sequence_id=inventory.sequence_id,
                    frame_index=frame_index,
                    file_name=path.name,
                    relative_path=path.relative_to(test_root).as_posix(),
                    bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                    width=width,
                    height=height,
                    mean_luminance=mean_luminance,
                    source_occupancy_vector="".join(
                        str(value) for value in values
                    ),
                    occupied_count=sum(value == 0 for value in values),
                    vacant_count=sum(value == 1 for value in values),
                )
            )
        if not candidate_rows:
            reasons.append("no_images_at_frozen_low_light_threshold")
        occupied = sum(row.occupied_count for row in candidate_rows)
        vacant = sum(row.vacant_count for row in candidate_rows)
        if occupied == 0 or vacant == 0:
            reasons.append("selected_frames_do_not_contain_both_states")
        qualified = not reasons
        if qualified:
            rows.extend(candidate_rows)
        selected_records = [
            truth_by_name[row.file_name] for row in candidate_rows
        ]
        transition_frames, slot_changes = _transition_counts(selected_records)
        decisions.append(
            {
                "sequence_id": inventory.sequence_id,
                "qualified": qualified,
                "reasons": reasons,
                "raw_image_count": inventory.image_count,
                "truth_count": inventory.truth_count,
                "selected_low_light_images": len(candidate_rows),
                "selected_occupied_labels": occupied,
                "selected_vacant_labels": vacant,
                "selected_transition_frames": transition_frames,
                "selected_slot_state_changes": slot_changes,
                "resolution": list(inventory.resolutions),
                "camera_stability_evidence": (
                    "same_resolution_plus_pre_model_contact_sheet_review"
                ),
            }
        )
    return rows, decisions


def write_night_test_manifest(
    path: Path,
    rows: Sequence[NightManifestRow],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if not rows:
        raise StageQV2DataError("Cannot write an empty night-test manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(payload[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(payload)


def require_human_polygon_confirmation(
    *,
    gate_status: str,
    polygon_confirmation: bool,
    run: Any,
) -> Any:
    if gate_status != "PASS":
        raise StageQV2DataError(
            f"Night test gate does not authorize inference: {gate_status}"
        )
    if polygon_confirmation is not True:
        raise StageQV2DataError(
            "Human polygon numbering confirmation is required before inference"
        )
    return run()


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
