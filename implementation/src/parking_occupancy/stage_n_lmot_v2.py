from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from parking_occupancy.stage_n_lmot import (
    MOTOR_VEHICLE_CLASSES,
    NON_MOTOR_CLASSES,
    StageNDataGateError,
    parse_lmot_gt,
    read_image,
    sha256_file,
    write_image,
)


STAGE_N_V2_PROTOCOL_ID = "STAGE-N-V2-LMOT-TRACKING-DIAGNOSTIC-20260729-01"
EXPECTED_VALIDATION_SEQUENCES = (
    "LMOT-05",
    "LMOT-13",
    "LMOT-14",
    "LMOT-25",
)
EXPECTED_FRAMES_PER_SEQUENCE = 1210


def _part_code(index: int) -> str:
    if index < 0 or index >= 26 * 26:
        raise ValueError("split-part index is outside aa-zz")
    return chr(ord("a") + index // 26) + chr(ord("a") + index % 26)


def discover_split_tar_parts(directory: Path, archive_stem: str) -> list[Path]:
    """Return an exact, contiguous ``.taraa`` ... split-tar sequence."""

    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    pattern = re.compile(rf"^{re.escape(archive_stem)}([a-z]{{2}})$")
    matched: list[tuple[str, Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_file() and match:
            matched.append((match.group(1), path))
    matched.sort(key=lambda row: row[0])
    if not matched:
        raise StageNDataGateError(
            f"No split parts match {archive_stem}aa in {directory}"
        )
    expected = [_part_code(index) for index in range(len(matched))]
    observed = [code for code, _path in matched]
    if observed != expected:
        raise StageNDataGateError(
            f"Split parts are not contiguous: expected {expected}, got {observed}"
        )
    return [path for _code, path in matched]


class ConcatenatedPartReader(io.RawIOBase):
    """Read ordered split files as one binary stream without joining them."""

    def __init__(self, parts: Sequence[Path]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("At least one split part is required")
        self.parts = tuple(Path(path).resolve() for path in parts)
        self._part_index = 0
        self._handle: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def _open_current(self) -> BinaryIO | None:
        if self._part_index >= len(self.parts):
            return None
        if self._handle is None:
            self._handle = self.parts[self._part_index].open("rb")
        return self._handle

    def readinto(self, buffer: bytearray | memoryview) -> int:
        target = memoryview(buffer)
        written = 0
        while written < len(target):
            handle = self._open_current()
            if handle is None:
                break
            count = handle.readinto(target[written:])
            if count:
                written += count
                continue
            handle.close()
            self._handle = None
            self._part_index += 1
        return written

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise StageNDataGateError(f"Unsafe tar member path: {name}")
    return path


def _copy_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    source = archive.extractfile(member)
    if source is None:
        raise StageNDataGateError(f"Could not read tar member {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("xb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)


def inspect_or_extract_rgb_split_tar(
    *,
    parts: Sequence[Path],
    archive_root: str,
    image_directory: str,
    expected_extension: str,
    extract_to: Path | None = None,
) -> dict[str, Any]:
    """Audit a full train/val transport tar and optionally extract val RGB only."""

    if image_directory not in {"img_dark_rgb", "img_light_rgb"}:
        raise ValueError(f"Unsupported RGB directory: {image_directory}")
    expected_extension = expected_extension.lower()
    frame_numbers: dict[str, set[int]] = {
        sequence: set() for sequence in EXPECTED_VALIDATION_SEQUENCES
    }
    split_counts: Counter[str] = Counter()
    split_bytes: Counter[str] = Counter()
    member_count = 0
    file_count = 0
    extracted_files = 0
    extracted_bytes = 0

    reader = io.BufferedReader(
        ConcatenatedPartReader(parts), buffer_size=1024 * 1024
    )
    with reader, tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            member_count += 1
            path = _safe_member_path(member.name)
            if member.issym() or member.islnk():
                raise StageNDataGateError(
                    f"Archive links are prohibited: {member.name}"
                )
            if not path.parts or path.parts[0] != archive_root:
                raise StageNDataGateError(
                    f"Unexpected archive root in {member.name}"
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise StageNDataGateError(
                    f"Unsupported tar member type: {member.name}"
                )
            file_count += 1
            if len(path.parts) < 5:
                raise StageNDataGateError(
                    f"Unexpected image member path: {member.name}"
                )
            split = path.parts[1]
            if split not in {"train", "val"}:
                raise StageNDataGateError(
                    f"Only train/val transport members are allowed: {member.name}"
                )
            sequence = path.parts[2]
            entry = path.parts[3]
            if entry != image_directory or len(path.parts) != 5:
                raise StageNDataGateError(
                    f"Unexpected content in RGB archive: {member.name}"
                )
            extension = PurePosixPath(path.parts[4]).suffix.lower()
            if extension != expected_extension:
                raise StageNDataGateError(
                    f"Unexpected {image_directory} extension {extension}: "
                    f"{member.name}"
                )
            try:
                frame_number = int(PurePosixPath(path.parts[4]).stem)
            except ValueError as exc:
                raise StageNDataGateError(
                    f"Non-numeric frame name: {member.name}"
                ) from exc
            split_counts[split] += 1
            split_bytes[split] += member.size
            if split == "train":
                continue
            if sequence not in frame_numbers:
                raise StageNDataGateError(
                    f"Unexpected validation sequence: {sequence}"
                )
            if frame_number in frame_numbers[sequence]:
                raise StageNDataGateError(
                    f"Duplicate {sequence} frame {frame_number}"
                )
            frame_numbers[sequence].add(frame_number)
            if extract_to is not None:
                destination = (
                    extract_to
                    / sequence
                    / image_directory
                    / path.parts[4]
                )
                _copy_tar_member(archive, member, destination)
                extracted_files += 1
                extracted_bytes += member.size

    expected_frames = set(range(1, EXPECTED_FRAMES_PER_SEQUENCE + 1))
    incomplete = {
        sequence: {
            "missing": sorted(expected_frames - frames),
            "unexpected": sorted(frames - expected_frames),
        }
        for sequence, frames in frame_numbers.items()
        if frames != expected_frames
    }
    if incomplete:
        raise StageNDataGateError(
            f"Incomplete validation RGB streams: {incomplete}"
        )
    return {
        "archive_root": archive_root,
        "image_directory": image_directory,
        "expected_extension": expected_extension,
        "parts": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in parts
        ],
        "logical_bytes": sum(path.stat().st_size for path in parts),
        "member_count": member_count,
        "file_count": file_count,
        "transport_train_files_not_extracted": split_counts["train"],
        "transport_train_bytes_not_extracted": split_bytes["train"],
        "validation_files": split_counts["val"],
        "validation_bytes": split_bytes["val"],
        "validation_sequences": {
            sequence: len(frames)
            for sequence, frames in sorted(frame_numbers.items())
        },
        "extracted_files": extracted_files,
        "extracted_bytes": extracted_bytes,
    }


def extract_validation_annotations(
    *,
    archive_path: Path,
    extract_to: Path,
) -> dict[str, Any]:
    """Extract only validation ``gt.txt`` and ``seqinfo.ini`` from LMOT."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    accepted: dict[str, set[str]] = {
        sequence: set() for sequence in EXPECTED_VALIDATION_SEQUENCES
    }
    member_count = 0
    train_files = 0
    extracted_files = 0
    extracted_bytes = 0
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_count += 1
            path = _safe_member_path(member.name)
            if member.issym() or member.islnk():
                raise StageNDataGateError(
                    f"Archive links are prohibited: {member.name}"
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise StageNDataGateError(
                    f"Unsupported annotation member type: {member.name}"
                )
            if len(path.parts) < 4 or path.parts[0] != "LMOT_annotations":
                raise StageNDataGateError(
                    f"Unexpected annotation path: {member.name}"
                )
            split = path.parts[1]
            if split == "train":
                train_files += 1
                continue
            if split != "val":
                raise StageNDataGateError(
                    f"Unexpected annotation split: {member.name}"
                )
            sequence = path.parts[2]
            if sequence not in accepted:
                raise StageNDataGateError(
                    f"Unexpected validation sequence: {sequence}"
                )
            relative: Path
            if path.parts[3:] == ("seqinfo.ini",):
                relative = Path(sequence) / "seqinfo.ini"
                accepted[sequence].add("seqinfo.ini")
            elif path.parts[3:] == ("gt", "gt.txt"):
                relative = Path(sequence) / "gt" / "gt.txt"
                accepted[sequence].add("gt/gt.txt")
            else:
                raise StageNDataGateError(
                    f"Unexpected validation annotation: {member.name}"
                )
            _copy_tar_member(archive, member, extract_to / relative)
            extracted_files += 1
            extracted_bytes += member.size
    missing = {
        sequence: sorted({"seqinfo.ini", "gt/gt.txt"} - entries)
        for sequence, entries in accepted.items()
        if entries != {"seqinfo.ini", "gt/gt.txt"}
    }
    if missing:
        raise StageNDataGateError(
            f"Validation annotations are incomplete: {missing}"
        )
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "member_count": member_count,
        "transport_train_files_not_extracted": train_files,
        "extracted_files": extracted_files,
        "extracted_bytes": extracted_bytes,
    }


def build_file_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "root": str(root.resolve()),
        "file_count": len(records),
        "total_bytes": sum(row["bytes"] for row in records),
        "records": records,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def verify_file_manifest(
    manifest_path: Path, *, expected_root: Path
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise StageNDataGateError("Extracted-file manifest has no records")
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    canonical_hash = hashlib.sha256(canonical).hexdigest()
    if canonical_hash != payload.get("manifest_sha256"):
        raise StageNDataGateError("Extracted-file manifest record hash mismatch")
    root = expected_root.resolve()
    total_bytes = 0
    for record in records:
        relative = PurePosixPath(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise StageNDataGateError(
                f"Unsafe extracted-file manifest path: {relative}"
            )
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(record["bytes"]):
            raise StageNDataGateError(f"Extracted-file size mismatch: {path}")
        if sha256_file(path) != record["sha256"]:
            raise StageNDataGateError(f"Extracted-file hash mismatch: {path}")
        total_bytes += size
    if len(records) != int(payload.get("file_count", -1)):
        raise StageNDataGateError("Extracted-file count mismatch")
    if total_bytes != int(payload.get("total_bytes", -1)):
        raise StageNDataGateError("Extracted-file total byte mismatch")
    return {
        "file_count": len(records),
        "total_bytes": total_bytes,
        "manifest_sha256": canonical_hash,
    }


@dataclass(frozen=True, slots=True)
class LmotClassMapV2:
    """LMOT class map backed by frozen distribution and visual evidence."""

    id_to_name: Mapping[int, str]
    verification_status: str
    evidence: str
    evidence_sha256: str
    evaluated_mark_values: frozenset[int]

    def __post_init__(self) -> None:
        if self.verification_status != "empirical_visual_verified":
            raise StageNDataGateError(
                "Stage N-v2 requires empirical visual verification"
            )
        expected = MOTOR_VEHICLE_CLASSES | NON_MOTOR_CLASSES
        names = {str(value).lower() for value in self.id_to_name.values()}
        if names != expected or set(self.id_to_name) != set(range(1, 7)):
            raise StageNDataGateError(
                "Stage N-v2 class map must cover IDs 1-6 exactly"
            )
        if not self.evidence_sha256:
            raise StageNDataGateError("Class-map evidence hash is required")
        if self.evaluated_mark_values != frozenset({1}):
            raise StageNDataGateError(
                "LMOT validation evidence supports evaluated mark value 1 only"
            )

    def class_name(self, class_id: int) -> str:
        try:
            return str(self.id_to_name[int(class_id)]).lower()
        except KeyError as exc:
            raise StageNDataGateError(
                f"Class ID {class_id} is absent from the verified map"
            ) from exc

    def is_motor_vehicle(self, class_id: int) -> bool:
        return self.class_name(class_id) in MOTOR_VEHICLE_CLASSES

    def is_non_motor(self, class_id: int) -> bool:
        return self.class_name(class_id) in NON_MOTOR_CLASSES


def load_lmot_class_map_v2(path: Path) -> LmotClassMapV2:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    evidence_path = (path.parent / payload["evidence_file"]).resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    evidence_hash = sha256_file(evidence_path)
    if evidence_hash != payload["evidence_sha256"]:
        raise StageNDataGateError("LMOT class-map evidence SHA-256 mismatch")
    return LmotClassMapV2(
        id_to_name={
            int(key): str(value)
            for key, value in payload["id_to_name"].items()
        },
        verification_status=payload["verification_status"],
        evidence=str(evidence_path),
        evidence_sha256=evidence_hash,
        evaluated_mark_values=frozenset(
            int(value) for value in payload["evaluated_mark_values"]
        ),
    )


def load_stage_n_v2_protocol(
    path: Path, *, verify_preserved_files: bool = False
) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise StageNDataGateError("Unsupported Stage N-v2 schema")
    if payload.get("protocol_id") != STAGE_N_V2_PROTOCOL_ID:
        raise StageNDataGateError("Unexpected Stage N-v2 protocol ID")
    if verify_preserved_files:
        for record in payload["preservation"].values():
            preserved = (path.parent / record["path"]).resolve()
            if not preserved.is_file():
                raise FileNotFoundError(preserved)
            if preserved.stat().st_size != int(record["bytes"]):
                raise StageNDataGateError(
                    f"Preserved file byte mismatch: {preserved}"
                )
            if sha256_file(preserved) != record["sha256"]:
                raise StageNDataGateError(
                    f"Preserved file SHA-256 mismatch: {preserved}"
                )
    return payload


def class_distribution(validation_root: Path) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    mark_counts: Counter[int] = Counter()
    sequence_rows: dict[str, int] = {}
    for sequence in EXPECTED_VALIDATION_SEQUENCES:
        rows = parse_lmot_gt(validation_root / sequence / "gt" / "gt.txt")
        sequence_rows[sequence] = len(rows)
        class_counts.update(row.class_id for row in rows)
        mark_counts.update(row.ignore for row in rows)
    return {
        "validation_sequences": list(EXPECTED_VALIDATION_SEQUENCES),
        "sequence_rows": sequence_rows,
        "total_rows": sum(sequence_rows.values()),
        "class_id_counts": dict(sorted(class_counts.items())),
        "mark_value_counts": dict(sorted(mark_counts.items())),
    }


def _context_crop(
    image: np.ndarray, xyxy: tuple[float, float, float, float]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    margin_x = box_w * 0.55
    margin_y = box_h * 0.55
    left = max(0, int(x1 - margin_x))
    top = max(0, int(y1 - margin_y))
    right = min(width, int(x2 + margin_x))
    bottom = min(height, int(y2 + margin_y))
    local_box = (
        max(0, int(x1) - left),
        max(0, int(y1) - top),
        min(right - left - 1, int(x2) - left),
        min(bottom - top - 1, int(y2) - top),
    )
    return image[top:bottom, left:right].copy(), local_box


def render_class_mapping_contact_sheet(
    *,
    validation_root: Path,
    id_to_name: Mapping[int, str],
    output_path: Path,
    samples_per_class: int = 3,
) -> list[dict[str, Any]]:
    """Render high-visibility, large-box examples for IDs 1-6."""

    candidates: dict[int, list[tuple[float, str, Any]]] = {
        class_id: [] for class_id in range(1, 7)
    }
    for sequence in EXPECTED_VALIDATION_SEQUENCES:
        for row in parse_lmot_gt(
            validation_root / sequence / "gt" / "gt.txt"
        ):
            score = row.width * row.height * max(row.visibility, 0.01)
            candidates[row.class_id].append((score, sequence, row))

    tile_width, tile_height = 320, 230
    sheet = np.full(
        (tile_height * 6, tile_width * samples_per_class, 3),
        245,
        dtype=np.uint8,
    )
    evidence: list[dict[str, Any]] = []
    for class_id in range(1, 7):
        chosen: list[tuple[float, str, Any]] = []
        used_sequences: set[str] = set()
        ordered = sorted(
            candidates[class_id], key=lambda candidate: candidate[0], reverse=True
        )
        for candidate in ordered:
            if candidate[1] in used_sequences and len(used_sequences) < 3:
                continue
            chosen.append(candidate)
            used_sequences.add(candidate[1])
            if len(chosen) == samples_per_class:
                break
        if len(chosen) < samples_per_class:
            for candidate in ordered:
                if candidate not in chosen:
                    chosen.append(candidate)
                if len(chosen) == samples_per_class:
                    break
        for column, (_score, sequence, row) in enumerate(chosen):
            image_path = (
                validation_root
                / sequence
                / "img_light_rgb"
                / f"{row.frame_number:06d}.jpg"
            )
            image = read_image(image_path)
            if image is None:
                raise StageNDataGateError(f"Could not decode {image_path}")
            crop, local_box = _context_crop(image, row.xyxy)
            if crop.size == 0:
                raise StageNDataGateError(
                    f"Empty crop for class {class_id} in {image_path}"
                )
            cv2.rectangle(
                crop,
                (local_box[0], local_box[1]),
                (local_box[2], local_box[3]),
                (0, 255, 0),
                max(2, round(min(crop.shape[:2]) / 120)),
            )
            resized = cv2.resize(
                crop, (tile_width, tile_height), interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(resized, (0, 0), (tile_width - 1, 31), (0, 0, 0), -1)
            label = f"ID {class_id}: {id_to_name[class_id]}"
            cv2.putText(
                resized,
                label,
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row_start = (class_id - 1) * tile_height
            col_start = column * tile_width
            sheet[
                row_start : row_start + tile_height,
                col_start : col_start + tile_width,
            ] = resized
            evidence.append(
                {
                    "class_id": class_id,
                    "proposed_name": id_to_name[class_id],
                    "sequence": sequence,
                    "frame": row.frame_number,
                    "track_id": row.track_id,
                    "bbox_xywh": [row.x, row.y, row.width, row.height],
                    "visibility": row.visibility,
                    "source_image": str(image_path),
                }
            )
    write_image(output_path, sheet)
    return evidence
