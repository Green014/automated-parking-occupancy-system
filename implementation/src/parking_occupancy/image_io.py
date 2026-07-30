from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from paths that may contain non-ASCII characters."""

    path = Path(path)
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image to paths that may contain non-ASCII characters."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode image as {extension}")
    encoded.tofile(path)
