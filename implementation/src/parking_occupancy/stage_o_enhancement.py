from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stage_n_lmot import sha256_file
from .stage_o_low_light import StageOProtocolError


RETINEXFORMER_LOL_V2_REAL_SHA256 = (
    "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b"
)


def retinex_tensor_to_rgb(restored: Any) -> np.ndarray:
    """Convert an inference tensor without mutating inference-mode storage."""
    return (
        restored.squeeze(0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


class RetinexformerPreprocessor:
    """Official full-resolution LOL-v2-real network as an O2 preprocessor."""

    def __init__(
        self,
        *,
        repository: Path,
        weights: Path,
        device: str = "0",
    ) -> None:
        repository = repository.resolve()
        weights = weights.resolve()
        if not (repository / "LICENSE.txt").is_file():
            raise FileNotFoundError(repository / "LICENSE.txt")
        if not weights.is_file():
            raise FileNotFoundError(weights)
        actual_hash = sha256_file(weights)
        if actual_hash != RETINEXFORMER_LOL_V2_REAL_SHA256:
            raise StageOProtocolError(
                "Retinexformer LOL-v2-real checkpoint SHA-256 mismatch"
            )
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))

        import torch
        from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

        if not torch.cuda.is_available() or device.lower() in {"cpu", "mps"}:
            raise StageOProtocolError(
                "Frozen O2 diagnostic requires the local CUDA device"
            )
        self.torch = torch
        self.repository = repository
        self.weights = weights
        self.device = torch.device(f"cuda:{int(device)}")
        self.model = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=40,
            stage=1,
            num_blocks=[1, 2, 2],
        ).to(self.device)
        checkpoint = torch.load(
            weights, map_location="cpu", weights_only=False
        )
        state = checkpoint["params"]
        incompatible = self.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise StageOProtocolError(
                "Retinexformer checkpoint did not strict-load"
            )
        self.model.eval()
        self.calls = 0
        self.elapsed_seconds = 0.0
        self.state_dict_keys = len(state)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Retinexformer expects a BGR image")
        torch = self.torch
        tensor = (
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        started = time.perf_counter()
        with torch.inference_mode():
            restored = self.model(tensor)
        torch.cuda.synchronize(self.device)
        self.elapsed_seconds += time.perf_counter() - started
        self.calls += 1
        rgb = retinex_tensor_to_rgb(restored)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def metadata(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "method": "Retinexformer",
            "role": "preprocessing_only_diagnostic",
            "repository": str(self.repository),
            "repository_license": "MIT",
            "repository_license_sha256": sha256_file(
                self.repository / "LICENSE.txt"
            ),
            "weights": str(self.weights),
            "weights_sha256": sha256_file(self.weights),
            "state_dict_keys": self.state_dict_keys,
            "network": {
                "in_channels": 3,
                "out_channels": 3,
                "n_feat": 40,
                "stage": 1,
                "num_blocks": [1, 2, 2],
            },
            "precision": "float32",
            "full_resolution": True,
            "tiling": False,
            "calls": self.calls,
            "elapsed_seconds": self.elapsed_seconds,
            "mean_ms": (
                1000.0 * self.elapsed_seconds / self.calls
                if self.calls
                else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device),
        }
