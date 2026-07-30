from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_fiftyone_samples(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> Iterator[dict[str, Any]]:
    """Stream samples from a FiftyOne ``samples.json`` export."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        while '"samples"' not in buffer:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError("No samples array found")
            buffer += chunk
        samples_key = buffer.index('"samples"')
        array_start = buffer.index("[", samples_key)
        buffer = buffer[array_start + 1 :]

        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                sample, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError("Truncated samples array")
                buffer += chunk
                continue
            if not isinstance(sample, dict):
                raise ValueError("Expected each sample to be an object")
            yield sample
            buffer = buffer[end:]


def scalar_date(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("$date", "")
    return str(value)


def label_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("label", "")
    return str(value)
