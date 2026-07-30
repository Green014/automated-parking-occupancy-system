"""Bounded, non-overwriting access helpers for the official VIRAT release."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

API_ROOT = "https://data.kitware.com/api/v1"
VIDEOS_ORIGINAL_FOLDER_ID = "56f581ce8d777f753209ca43"
ANNOTATIONS_FOLDER_ID = "56f57e748d777f753209bed8"
_CLIP_PATTERN = re.compile(
    r"^(VIRAT_S_((\d{4})(\d{2}))(?:_[^.]+)?)\.mp4$"
)
_ANNOTATION_PATTERN = re.compile(
    r"^(VIRAT_S_((\d{4})(\d{2}))(?:_[^.]+)?)"
    r"\.viratdata\.(?:events|mapping|objects)\.txt$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class ViratItem:
    """Minimum immutable metadata required for a safe download."""

    item_id: str
    name: str
    size: int
    scene_id: str | None
    sequence_id: str | None


def parse_video_item(raw: dict[str, Any]) -> ViratItem | None:
    """Parse one Girder item, returning ``None`` for non-video entries."""

    name = str(raw.get("name", ""))
    match = _CLIP_PATTERN.fullmatch(name)
    if match is None:
        return None
    item_id = str(raw.get("_id", ""))
    size = raw.get("size")
    if not item_id or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"invalid VIRAT item metadata for {name!r}")
    return ViratItem(
        item_id=item_id,
        name=name,
        size=size,
        # Release 2.0 defines XXYY as the physical scene and ZZ as the
        # sequence. Six-digit prefixes must not be treated as independent
        # scenes.
        scene_id=match.group(3),
        sequence_id=match.group(4),
    )


def parse_annotation_item(raw: dict[str, Any]) -> ViratItem | None:
    """Parse an official event/mapping/object annotation item."""

    name = str(raw.get("name", ""))
    match = _ANNOTATION_PATTERN.fullmatch(name)
    if match is None:
        return None
    item_id = str(raw.get("_id", ""))
    size = raw.get("size")
    if not item_id or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"invalid VIRAT item metadata for {name!r}")
    return ViratItem(
        item_id=item_id,
        name=name,
        size=size,
        scene_id=match.group(3),
        sequence_id=match.group(4),
    )


def select_cross_scene_candidates(
    items: Iterable[ViratItem],
    *,
    max_candidates: int,
    max_total_bytes: int,
    max_item_bytes: int,
) -> list[ViratItem]:
    """Choose the smallest videos while allowing at most one per scene."""

    if max_candidates < 1 or max_total_bytes < 1 or max_item_bytes < 1:
        raise ValueError("candidate and byte limits must be positive")

    selected: list[ViratItem] = []
    seen_scenes: set[str] = set()
    total = 0
    for item in sorted(items, key=lambda candidate: (candidate.size, candidate.name)):
        if item.scene_id is None or item.scene_id in seen_scenes:
            continue
        if item.size > max_item_bytes or total + item.size > max_total_bytes:
            continue
        selected.append(item)
        seen_scenes.add(item.scene_id)
        total += item.size
        if len(selected) == max_candidates:
            break
    return selected


def select_named_items(
    items: Iterable[ViratItem],
    names: Iterable[str],
    *,
    max_total_bytes: int,
    max_item_bytes: int,
) -> list[ViratItem]:
    """Resolve exact official names and enforce declared byte limits."""

    requested = list(names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested names must be non-empty and unique")
    catalog = {item.name: item for item in items}
    missing = [name for name in requested if name not in catalog]
    if missing:
        raise ValueError("items not present in official catalog: " + ", ".join(missing))
    selected = [catalog[name] for name in requested]
    oversized = [item.name for item in selected if item.size > max_item_bytes]
    if oversized:
        raise ValueError("items exceed max_item_bytes: " + ", ".join(oversized))
    declared_total = sum(item.size for item in selected)
    if declared_total > max_total_bytes:
        raise ValueError(
            f"declared total {declared_total} exceeds budget {max_total_bytes}"
        )
    return selected


def fetch_folder_items(
    folder_id: str,
    *,
    page_size: int = 100,
    max_attempts: int = 3,
    retry_delay_s: float = 1.0,
    opener: Callable[..., BinaryIO] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Fetch a public Girder folder, retrying only transient request failures."""

    if (
        not folder_id
        or page_size < 1
        or max_attempts < 1
        or retry_delay_s < 0.0
    ):
        raise ValueError(
            "folder_id, positive page_size/max_attempts, and non-negative "
            "retry_delay_s are required"
        )
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = urlencode(
            {
                "folderId": folder_id,
                "limit": page_size,
                "offset": offset,
                "sort": "name",
                "sortdir": 1,
            }
        )
        url = f"{API_ROOT}/item?{query}"
        for attempt in range(1, max_attempts + 1):
            try:
                with opener(url, timeout=30) as response:
                    page = json.load(response)
                break
            except HTTPError as error:
                if (
                    error.code not in _RETRYABLE_HTTP_STATUSES
                    or attempt == max_attempts
                ):
                    raise
                sleeper(retry_delay_s * attempt)
            except (URLError, TimeoutError):
                if attempt == max_attempts:
                    raise
                sleeper(retry_delay_s * attempt)
        if not isinstance(page, list):
            raise ValueError("unexpected VIRAT catalog response")
        records.extend(page)
        if len(page) < page_size:
            break
        offset += len(page)
    return records


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_screening_manifest(
    payload: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    """Reconcile a screening manifest against non-versioned local videos."""

    errors: list[str] = []
    candidates = payload.get("candidates")
    screening = payload.get("screening")
    if not isinstance(candidates, list) or not isinstance(screening, dict):
        return {
            "valid": False,
            "checked_files": 0,
            "errors": ["manifest candidates and screening mappings are required"],
        }

    names = [str(record.get("name", "")) for record in candidates]
    if len(set(names)) != len(names):
        errors.append("candidate names must be unique")
    declared_bytes = sum(
        record.get("bytes", 0)
        for record in candidates
        if isinstance(record, dict)
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool)
    )
    if len(candidates) != screening.get("downloaded_video_count"):
        errors.append("downloaded_video_count does not match candidate count")
    if declared_bytes != screening.get("downloaded_bytes"):
        errors.append("downloaded_bytes does not match candidate byte sum")
    budget = screening.get("budget_bytes")
    if not isinstance(budget, int) or isinstance(budget, bool) or declared_bytes > budget:
        errors.append("declared bytes exceed or lack a valid screening budget")

    local_by_name: dict[str, list[Path]] = {}
    for path in data_root.rglob("*.mp4"):
        local_by_name.setdefault(path.name, []).append(path)

    checked = 0
    for record in candidates:
        if not isinstance(record, dict):
            errors.append("every candidate must be a mapping")
            continue
        name = str(record.get("name", ""))
        digest = str(record.get("sha256", "")).lower()
        size = record.get("bytes")
        try:
            parsed = parse_video_item(
                {
                    "_id": record.get("item_id"),
                    "name": name,
                    "size": size,
                }
            )
        except ValueError as error:
            errors.append(f"{name}: {error}")
            continue
        if parsed is None:
            errors.append(f"{name}: invalid official video name")
            continue
        if parsed.scene_id != record.get("physical_scene_id"):
            errors.append(f"{name}: physical scene does not follow XXYY")
        if parsed.sequence_id != record.get("sequence_id"):
            errors.append(f"{name}: sequence does not follow ZZ")
        if not _SHA256_PATTERN.fullmatch(digest):
            errors.append(f"{name}: invalid SHA-256")
            continue

        matches = local_by_name.get(name, [])
        if len(matches) != 1:
            errors.append(f"{name}: expected one local file, found {len(matches)}")
            continue
        path = matches[0]
        if path.stat().st_size != size:
            errors.append(f"{name}: local byte size mismatch")
            continue
        if sha256_file(path) != digest:
            errors.append(f"{name}: local SHA-256 mismatch")
            continue
        checked += 1

    return {
        "valid": not errors,
        "checked_files": checked,
        "declared_files": len(candidates),
        "declared_bytes": declared_bytes,
        "errors": errors,
    }


def _iter_response(response: BinaryIO, chunk_size: int) -> Iterator[bytes]:
    while chunk := response.read(chunk_size):
        yield chunk


def download_item(
    item: ViratItem,
    output_dir: Path,
    *,
    opener: Callable[..., BinaryIO] = urlopen,
    chunk_size: int = 1024 * 1024,
    max_attempts: int = 3,
    retry_delay_s: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Download one item atomically without replacing any existing file."""

    if Path(item.name).name != item.name:
        raise ValueError("item name must not contain a path")
    if chunk_size < 1 or max_attempts < 1 or retry_delay_s < 0.0:
        raise ValueError(
            "positive chunk_size/max_attempts and non-negative retry_delay_s "
            "are required"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / item.name
    partial = output_dir / f"{item.name}.part"

    if target.exists():
        if not target.is_file() or target.stat().st_size != item.size:
            raise FileExistsError(
                f"existing target does not match official size: {target}"
            )
        return {
            "status": "existing",
            "path": str(target),
            "size": item.size,
            "sha256": sha256_file(target),
        }
    if partial.exists():
        raise FileExistsError(f"partial file already exists: {partial}")

    for attempt in range(1, max_attempts + 1):
        written = 0
        digest = hashlib.sha256()
        try:
            with opener(
                f"{API_ROOT}/item/{item.item_id}/download",
                timeout=60,
            ) as response, partial.open("xb") as handle:
                for chunk in _iter_response(response, chunk_size):
                    written += len(chunk)
                    if written > item.size:
                        raise ValueError(
                            "download exceeded the declared official size"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
            if written != item.size:
                raise ValueError(
                    f"downloaded {written} bytes but official metadata declares "
                    f"{item.size}"
                )
            partial.rename(target)
            break
        except HTTPError as error:
            if partial.exists():
                partial.unlink()
            if (
                error.code not in _RETRYABLE_HTTP_STATUSES
                or attempt == max_attempts
            ):
                raise
            sleeper(retry_delay_s * attempt)
        except (URLError, TimeoutError):
            if partial.exists():
                partial.unlink()
            if attempt == max_attempts:
                raise
            sleeper(retry_delay_s * attempt)
        except Exception:
            if partial.exists():
                partial.unlink()
            raise

    return {
        "status": "downloaded",
        "path": str(target),
        "size": written,
        "sha256": digest.hexdigest(),
    }
