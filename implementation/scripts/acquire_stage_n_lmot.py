from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file


FORBIDDEN_COMPONENTS = {
    "train",
    "test",
    "lmot-real",
    "img_dark",
    "img_light",
}
APPROVED_ENTRIES = {"img_dark_rgb", "img_light_rgb", "gt", "seqinfo.ini"}


@dataclass(frozen=True)
class Member:
    name: str
    size: int
    directory: bool
    link: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and optionally extract a user-supplied LMOT validation "
            "archive. This tool performs no network download."
        )
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path)
    return parser.parse_args()


def _members(path: Path) -> list[Member]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                Member(row.filename, row.file_size, row.is_dir())
                for row in archive.infolist()
            ]
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            return [
                Member(row.name, row.size, row.isdir(), row.issym() or row.islnk())
                for row in archive.getmembers()
            ]
    raise ValueError("Only ZIP and TAR-family archives are supported")


def _approved_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    lower = [part.lower() for part in path.parts]
    forbidden = FORBIDDEN_COMPONENTS.intersection(lower)
    if forbidden:
        raise ValueError(f"Forbidden LMOT content {sorted(forbidden)}: {name}")
    if any(PurePosixPath(part).suffix.lower() in {".tif", ".tiff"} for part in path.parts):
        raise ValueError(f"TIFF acquisition is prohibited: {name}")
    try:
        val_index = lower.index("val")
    except ValueError as exc:
        raise ValueError(f"Archive member is not under val/: {name}") from exc
    relative = PurePosixPath(*path.parts[val_index:])
    if len(relative.parts) < 3:
        raise ValueError(f"Invalid validation member path: {name}")
    entry = relative.parts[2]
    if entry not in APPROVED_ENTRIES:
        raise ValueError(f"Unapproved validation entry {entry}: {name}")
    if entry == "seqinfo.ini" and len(relative.parts) != 3:
        raise ValueError(f"Invalid seqinfo.ini path: {name}")
    if entry == "gt" and len(relative.parts) > 3 and relative.parts[3] != "gt.txt":
        raise ValueError(f"Only gt/gt.txt is approved: {name}")
    return relative


def _extract_zip(path: Path, accepted: list[tuple[Member, PurePosixPath]], target: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for member, relative in accepted:
            if member.directory:
                (target / Path(*relative.parts)).mkdir(parents=True, exist_ok=True)
                continue
            destination = target / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member.name) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)


def _extract_tar(path: Path, accepted: list[tuple[Member, PurePosixPath]], target: Path) -> None:
    with tarfile.open(path, "r:*") as archive:
        by_name = {row.name: row for row in archive.getmembers()}
        for member, relative in accepted:
            if member.directory:
                (target / Path(*relative.parts)).mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(by_name[member.name])
            if source is None:
                raise RuntimeError(f"Could not read {member.name}")
            destination = target / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)


def main() -> None:
    args = parse_args()
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if args.inventory.exists():
        raise FileExistsError(f"Refusing to overwrite {args.inventory}")
    rows = _members(archive)
    if any(row.link for row in rows):
        raise ValueError("Archive links are prohibited")
    accepted: list[tuple[Member, PurePosixPath]] = []
    errors: list[str] = []
    for row in rows:
        try:
            accepted.append((row, _approved_relative(row.name)))
        except ValueError as exc:
            errors.append(str(exc))
    payload = {
        "schema_version": 1,
        "status": "approved_for_limited_extraction" if not errors else "blocked",
        "network_download_performed": False,
        "source_url": args.source_url,
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "members": len(rows),
            "uncompressed_bytes": sum(row.size for row in rows if not row.directory),
        },
        "approved_member_count": len(accepted),
        "errors": errors,
    }
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(
            "Archive blocked; inventory was written and nothing was extracted"
        )
    if args.extract_to is not None:
        target = args.extract_to.resolve()
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        target.mkdir(parents=True)
        if zipfile.is_zipfile(archive):
            _extract_zip(archive, accepted, target)
        else:
            _extract_tar(archive, accepted, target)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
