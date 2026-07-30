from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


STAGE_K_CANDIDATE_ID = "P-COMP-PKLOT-TEST-STAGEK-CANDIDATE-20260727-02"
STAGE_K_GATE_V2_RECORD_ID = (
    "STAGE-K-PKLOT-DATA-GATE-RECORD-20260728-02"
)
CAMERA_NAMES = {
    "parking2": "pucpr",
    "parking1a": "ufpr04",
    "parking1b": "ufpr05",
}


class StageKDataGateError(ValueError):
    """Raised when the Stage K candidate cannot satisfy the data gate."""


@dataclass(frozen=True)
class CandidateGroup:
    archive_camera: str
    weather: str
    date: str
    count: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.archive_camera, self.weather, self.date)

    @property
    def camera(self) -> str:
        try:
            return CAMERA_NAMES[self.archive_camera]
        except KeyError as error:
            raise StageKDataGateError(
                f"Unknown PKLot camera directory: {self.archive_camera}"
            ) from error


@dataclass(frozen=True)
class ArchivePair:
    stem: str
    archive_camera: str
    weather: str
    date: str
    image_member: str
    xml_member: str

    @property
    def camera(self) -> str:
        return CAMERA_NAMES[self.archive_camera]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_member_name(name: str) -> tuple[str, str, str, str, str] | None:
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) != 5 or parts[0] != "PKLot":
        return None
    archive_camera, weather, date = parts[1:4]
    suffix = path.suffix.lower()
    if archive_camera not in CAMERA_NAMES or suffix not in {".jpg", ".xml"}:
        return None
    if path.stem[:10] != date:
        return None
    return archive_camera, weather, date, path.stem, suffix


def scan_archive_pairs(
    archive_path: Path,
) -> tuple[list[ArchivePair], dict[str, Any]]:
    members: dict[
        tuple[str, str, str, str],
        dict[str, str],
    ] = defaultdict(dict)
    members_read = 0
    stream_end = "clean_eof"
    with tarfile.open(archive_path, "r|gz") as archive:
        try:
            for member in archive:
                members_read += 1
                parsed = _parse_member_name(member.name)
                if parsed is None:
                    continue
                archive_camera, weather, date, stem, suffix = parsed
                members[(archive_camera, weather, date, stem)][suffix] = (
                    member.name
                )
        except tarfile.ReadError as error:
            stream_end = f"partial_archive: {error}"

    pairs = [
        ArchivePair(
            stem=stem,
            archive_camera=archive_camera,
            weather=weather,
            date=date,
            image_member=extensions[".jpg"],
            xml_member=extensions[".xml"],
        )
        for (
            archive_camera,
            weather,
            date,
            stem,
        ), extensions in members.items()
        if {".jpg", ".xml"}.issubset(extensions)
    ]
    pairs.sort(
        key=lambda pair: (
            pair.archive_camera,
            pair.weather,
            pair.date,
            pair.stem,
        )
    )
    return pairs, {
        "members_read": members_read,
        "unique_stems": len(members),
        "complete_pairs": len(pairs),
        "stream_end": stream_end,
    }


def _evenly_spaced(items: list[ArchivePair], count: int) -> list[ArchivePair]:
    if count <= 0:
        raise StageKDataGateError("Candidate count must be positive")
    if len(items) < count:
        raise StageKDataGateError(
            f"Requested {count} samples from only {len(items)} complete pairs"
        )
    if count == 1:
        return [items[len(items) // 2]]
    indices = [
        round(index * (len(items) - 1) / (count - 1))
        for index in range(count)
    ]
    if len(set(indices)) != count:
        raise StageKDataGateError("Even-spacing rule produced duplicate indices")
    return [items[index] for index in indices]


def select_candidate_pairs(
    pairs: Iterable[ArchivePair],
    groups: Iterable[CandidateGroup],
) -> list[ArchivePair]:
    by_group: dict[tuple[str, str, str], list[ArchivePair]] = defaultdict(list)
    for pair in pairs:
        by_group[(pair.archive_camera, pair.weather, pair.date)].append(pair)
    selected: list[ArchivePair] = []
    seen_groups: set[tuple[str, str, str]] = set()
    for group in groups:
        if group.key in seen_groups:
            raise StageKDataGateError(f"Duplicate candidate group: {group.key}")
        seen_groups.add(group.key)
        candidates = sorted(by_group[group.key], key=lambda pair: pair.stem)
        selected.extend(_evenly_spaced(candidates, group.count))
    return selected


def _extract_selected(
    *,
    archive_path: Path,
    selected: Iterable[ArchivePair],
) -> dict[str, bytes]:
    target_names = {
        name
        for pair in selected
        for name in (pair.image_member, pair.xml_member)
    }
    payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r|gz") as archive:
        try:
            for member in archive:
                if member.name not in target_names:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise StageKDataGateError(
                        f"Could not read archive member: {member.name}"
                    )
                payloads[member.name] = extracted.read()
                if len(payloads) == len(target_names):
                    break
        except tarfile.ReadError as error:
            raise StageKDataGateError(
                "Partial archive ended before all selected members were read"
            ) from error
    missing = sorted(target_names.difference(payloads))
    if missing:
        raise StageKDataGateError(
            f"Missing {len(missing)} selected archive members"
        )
    return payloads


def _decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise StageKDataGateError("Selected JPEG could not be decoded")
    return image


def parse_pklot_xml(
    payload: bytes,
    *,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = ET.fromstring(payload)
    polylines = []
    occupied = vacant = unknown = 0
    for space in root.findall("space"):
        contour = space.find("contour")
        if contour is None:
            raise StageKDataGateError("PKLot space is missing its contour")
        points = []
        for point in contour.findall("point"):
            x = float(point.attrib["x"])
            y = float(point.attrib["y"])
            if not (0 <= x <= width and 0 <= y <= height):
                raise StageKDataGateError("PKLot contour point is out of bounds")
            points.append([x / width, y / height])
        if len(points) < 3:
            raise StageKDataGateError("PKLot contour has fewer than 3 points")
        raw_status = space.attrib.get("occupied")
        if raw_status == "1":
            status = "occupied"
            occupied += 1
        elif raw_status == "0":
            status = "not occupied"
            vacant += 1
        else:
            status = "unknown"
            unknown += 1
        polylines.append(
            {
                "label": "parking_space",
                "points": [points],
                "closed": True,
                "filled": True,
                "occupancy_status": status,
                "space_id": int(space.attrib["id"]),
            }
        )
    polylines.sort(key=lambda item: item["space_id"])
    if not polylines:
        raise StageKDataGateError("PKLot XML contains no parking spaces")
    return polylines, {
        "known_slots": occupied + vacant,
        "occupied": occupied,
        "vacant": vacant,
        "unknown": unknown,
    }


def _load_prior_hashes(prior_manifest: Path) -> set[str]:
    with prior_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hashes = {row["image_sha256"] for row in rows}
    if not rows or "" in hashes:
        raise StageKDataGateError("Prior manifest has missing image SHA-256")
    return hashes


def _contact_sheet(
    thumbnails: list[tuple[str, np.ndarray]],
    columns: int = 10,
) -> bytes:
    thumb_width, thumb_height, label_height = 192, 108, 24
    rows = math.ceil(len(thumbnails) / columns)
    canvas = np.full(
        (rows * (thumb_height + label_height), columns * thumb_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, (label, image) in enumerate(thumbnails):
        row, column = divmod(index, columns)
        resized = cv2.resize(image, (thumb_width, thumb_height))
        y = row * (thumb_height + label_height)
        x = column * thumb_width
        canvas[y : y + thumb_height, x : x + thumb_width] = resized
        cv2.putText(
            canvas,
            label,
            (x + 3, y + thumb_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    encoded, payload = cv2.imencode(".png", canvas)
    if not encoded:
        raise StageKDataGateError("Could not encode Stage K contact sheet")
    return payload.tobytes()


def render_truth_contact_sheet(
    *,
    annotations_path: Path,
    source_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage K truth contact sheet: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnails = []
    records = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        image_path = source_root / record["local_path"]
        image = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise StageKDataGateError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        overlay = image.copy()
        for polygon in record["sample"]["parking_spaces"]["polylines"]:
            status = polygon["occupancy_status"]
            color = (
                (0, 0, 255)
                if status == "occupied"
                else (0, 180, 0)
                if status == "not occupied"
                else (0, 215, 255)
            )
            points = np.asarray(
                [
                    [
                        round(float(point[0]) * width),
                        round(float(point[1]) * height),
                    ]
                    for point in polygon["points"][0]
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [points], color)
            cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        thumbnails.append((str(record["sample_id"]), image))
    payload = _contact_sheet(thumbnails)
    output_path.write_bytes(payload)
    return {
        "images": len(records),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "legend": {
            "occupied": "red",
            "vacant": "green",
            "unknown": "yellow",
        },
    }


def _write_or_verify(path: Path, payload: bytes, *, reuse_existing: bool) -> None:
    if path.exists():
        if not reuse_existing:
            raise FileExistsError(f"Refusing to overwrite Stage K file: {path}")
        if not path.is_file() or path.read_bytes() != payload:
            raise StageKDataGateError(
                f"Existing Stage K file differs from regenerated content: {path}"
            )
        return
    path.write_bytes(payload)


def prepare_stage_k_candidate(
    *,
    archive_path: Path,
    prior_manifest: Path,
    output_root: Path,
    manifest_path: Path,
    annotations_path: Path,
    audit_path: Path,
    contact_sheet_path: Path,
    groups: list[CandidateGroup],
    official_archive_bytes: int,
    official_archive_url: str,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    for output in (
        manifest_path,
        annotations_path,
        audit_path,
        contact_sheet_path,
    ):
        if output.exists() and not reuse_existing:
            raise FileExistsError(f"Refusing to overwrite Stage K file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    existing_candidate = output_root.exists() and any(output_root.rglob("*"))
    if existing_candidate and not reuse_existing:
        raise FileExistsError(
            f"Refusing to overwrite Stage K candidate data: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    pairs, scan = scan_archive_pairs(archive_path)
    selected = select_candidate_pairs(pairs, groups)
    payloads = _extract_selected(
        archive_path=archive_path,
        selected=selected,
    )
    prior_hashes = _load_prior_hashes(prior_manifest)

    manifest_rows = []
    annotation_rows = []
    thumbnails = []
    selected_hashes: set[str] = set()
    totals = {
        "images": 0,
        "known_slots": 0,
        "occupied": 0,
        "vacant": 0,
        "unknown": 0,
    }
    for index, pair in enumerate(selected, start=1):
        image_payload = payloads[pair.image_member]
        xml_payload = payloads[pair.xml_member]
        image_hash = sha256_bytes(image_payload)
        if image_hash in selected_hashes:
            raise StageKDataGateError("Duplicate image SHA-256 in candidate")
        if image_hash in prior_hashes:
            raise StageKDataGateError(
                f"Candidate overlaps prior development data: {pair.stem}"
            )
        selected_hashes.add(image_hash)

        image = _decode_image(image_payload)
        height, width = image.shape[:2]
        if (width, height) != (1280, 720):
            raise StageKDataGateError(
                f"Unexpected PKLot image dimensions: {(width, height)}"
            )
        polylines, counts = parse_pklot_xml(
            xml_payload,
            width=width,
            height=height,
        )
        camera = pair.camera
        sample_id = f"pklot_stage_k_{index:03d}"
        image_relative = Path("images") / camera / pair.date / f"{pair.stem}.jpg"
        xml_relative = Path("xml") / camera / pair.date / f"{pair.stem}.xml"
        image_output = output_root / image_relative
        xml_output = output_root / xml_relative
        image_output.parent.mkdir(parents=True, exist_ok=True)
        xml_output.parent.mkdir(parents=True, exist_ok=True)
        if existing_candidate:
            if (
                not image_output.is_file()
                or image_output.read_bytes() != image_payload
                or not xml_output.is_file()
                or xml_output.read_bytes() != xml_payload
            ):
                raise StageKDataGateError(
                    "Existing candidate data does not match selected archive "
                    f"members: {pair.stem}"
                )
        else:
            image_output.write_bytes(image_payload)
            xml_output.write_bytes(xml_payload)

        manifest_rows.append(
            {
                "sample_id": sample_id,
                "role": "stage_k_candidate_no_predictions",
                "source": camera,
                "weather": pair.weather,
                "date": pair.date,
                "timestamp": f"{pair.stem.replace('_', 'T', 1)}Z",
                "group_id": f"{camera}/{pair.date}",
                "local_path": image_relative.as_posix(),
                "xml_path": xml_relative.as_posix(),
                "archive_image_member": pair.image_member,
                "archive_xml_member": pair.xml_member,
                "image_bytes": len(image_payload),
                "image_sha256": image_hash,
                "xml_bytes": len(xml_payload),
                "xml_sha256": sha256_bytes(xml_payload),
                "width": width,
                "height": height,
                **counts,
            }
        )
        annotation_rows.append(
            {
                "sample_id": sample_id,
                "role": "stage_k_candidate_no_predictions",
                "source": camera,
                "weather": pair.weather,
                "date": pair.date,
                "timestamp": f"{pair.stem.replace('_', 'T', 1)}Z",
                "local_path": image_relative.as_posix(),
                "sample": {
                    "parking_spaces": {
                        "polylines": polylines,
                    }
                },
            }
        )
        totals["images"] += 1
        for key in ("known_slots", "occupied", "vacant", "unknown"):
            totals[key] += counts[key]
        thumbnails.append((f"{sample_id} {pair.stem[11:]}", image))

    fields = list(manifest_rows[0])
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest_buffer,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(manifest_rows)
    annotation_text = "".join(
        json.dumps(row, separators=(",", ":")) + "\n"
        for row in annotation_rows
    )
    _write_or_verify(
        manifest_path,
        manifest_buffer.getvalue().encode("utf-8"),
        reuse_existing=reuse_existing,
    )
    _write_or_verify(
        annotations_path,
        annotation_text.encode("utf-8"),
        reuse_existing=reuse_existing,
    )
    _write_or_verify(
        contact_sheet_path,
        _contact_sheet(thumbnails),
        reuse_existing=reuse_existing,
    )

    audit = {
        "schema_version": 1,
        "candidate_id": STAGE_K_CANDIDATE_ID,
        "status": "candidate_prepared_no_predictions",
        "prediction_count": 0,
        "existing_candidate_data_reused_and_verified": existing_candidate,
        "selection_rule": (
            "sort timestamps within each predeclared camera/weather/date "
            "group, then choose 30 evenly spaced complete JPG/XML pairs; "
            "do not inspect occupancy labels for membership selection"
        ),
        "groups": [
            {
                "archive_camera": group.archive_camera,
                "camera": group.camera,
                "weather": group.weather,
                "date": group.date,
                "requested": group.count,
                "available_complete_pairs": sum(
                    pair.archive_camera == group.archive_camera
                    and pair.weather == group.weather
                    and pair.date == group.date
                    for pair in pairs
                ),
            }
            for group in groups
        ],
        "totals": totals,
        "prior_development_manifest": {
            "bytes": prior_manifest.stat().st_size,
            "sha256": sha256_file(prior_manifest),
            "image_hashes": len(prior_hashes),
        },
        "overlap": {
            "selected_unique_image_hashes": len(selected_hashes),
            "prior_image_sha256_overlap": len(
                selected_hashes.intersection(prior_hashes)
            ),
        },
        "source": {
            "dataset": "PKLot",
            "official_archive_url": official_archive_url,
            "official_archive_expected_bytes": official_archive_bytes,
            "local_partial_archive_bytes": archive_path.stat().st_size,
            "local_partial_archive_sha256": sha256_file(archive_path),
            "local_archive_is_complete": (
                archive_path.stat().st_size == official_archive_bytes
            ),
            "scan": scan,
        },
        "artifacts": {
            "manifest": {
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
            "annotations": {
                "bytes": annotations_path.stat().st_size,
                "sha256": sha256_file(annotations_path),
            },
            "contact_sheet": {
                "bytes": contact_sheet_path.stat().st_size,
                "sha256": sha256_file(contact_sheet_path),
            },
        },
        "manual_visual_review": {
            "status": "pending",
            "review_scope": "all 90 candidate thumbnails",
        },
        "gate": "pending_manual_visual_review_and_protocol_freeze",
    }
    _write_or_verify(
        audit_path,
        (json.dumps(audit, indent=2) + "\n").encode("utf-8"),
        reuse_existing=reuse_existing,
    )
    return audit


def verify_stage_k_gate_v2_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Verify the additive Stage K data-gate v2 evidence record."""
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != STAGE_K_GATE_V2_RECORD_ID:
        raise StageKDataGateError("Unexpected Stage K data-gate v2 record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise StageKDataGateError(
                f"Unknown Stage K data-gate artifact root: {root_name}"
            )
        path = roots[root_name] / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "passed": passed,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_sha256,
            }
        )
    return {
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
