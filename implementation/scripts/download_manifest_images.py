from __future__ import annotations

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(row: dict[str, str], root: Path) -> dict[str, str | int]:
    target = root / row["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        with requests.get(row["download_url"], stream=True, timeout=60) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    return {
        "image_id": row["image_id"],
        "frame_index": row["frame_index"],
        "local_path": row["local_path"],
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download images listed in a dataset manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checksums-output", required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, row, Path(args.root)): row for row in rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % 50 == 0 or index == len(rows):
                print(f"Downloaded/verified {index}/{len(rows)}")
    completed.sort(key=lambda row: int(row["frame_index"]))
    output = Path(args.checksums_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "frame_index", "local_path", "bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(completed)


if __name__ == "__main__":
    main()
