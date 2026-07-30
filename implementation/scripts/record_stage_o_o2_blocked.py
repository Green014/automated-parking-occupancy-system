from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import read_image, sha256_file
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record the isolated GLARE/Retinexformer Stage O O2 preflight. "
            "This does not run D1 or detector evaluation."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    weights = args.weights.resolve()
    image_path = args.image.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for path in (repository, weights, image_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dependency_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output / "dependency_freeze.txt").write_text(
        dependency_freeze, encoding="utf-8"
    )

    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"Could not decode {image_path}")
    height, width = image.shape[:2]
    config = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": "O2",
        "role": "preprocessing_only_diagnostic",
        "primary": {
            "method": "GLARE",
            "status": "blocked_preflight",
            "official_required_environment": {
                "python": "3.8",
                "pytorch": "1.11",
                "cuda": "11.3",
                "native_extension": "models/ops/setup.py deformable convolution",
            },
            "local_tool_availability": {
                name: shutil.which(name)
                for name in ("python3.8", "conda", "nvcc", "cl")
            },
        },
        "only_allowed_fallback": {
            "method": "Retinexformer",
            "official_checkpoint": "LOL_v2_real",
            "network": {
                "in_channels": 3,
                "out_channels": 3,
                "n_feat": 40,
                "stage": 1,
                "num_blocks": [1, 2, 2],
            },
            "inference_policy": (
                "official full-resolution model path; no post-result tiling, "
                "threshold change, or alternate enhancer"
            ),
        },
        "detector_loaded": False,
        "detector_predict_called": False,
        "detector_track_called": False,
    }
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    sys.path.insert(0, str(repository))
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer

    model = RetinexFormer(
        in_channels=3,
        out_channels=3,
        n_feat=40,
        stage=1,
        num_blocks=[1, 2, 2],
    ).cuda()
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    state = checkpoint["params"]
    load_result = model.load_state_dict(state, strict=True)
    model.eval()
    tensor = (
        torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .div_(255.0)
        .cuda()
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_before = torch.cuda.mem_get_info()
    started = time.perf_counter()
    exception: RuntimeError | None = None
    try:
        with torch.inference_mode():
            _ = model(tensor)
        torch.cuda.synchronize()
    except RuntimeError as error:
        exception = error
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    expected_block = (
        exception is not None
        and "CUDA out of memory" in str(exception)
    )
    runtime = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": "O2",
        "status": (
            "blocked_retinexformer_full_resolution_cuda_oom"
            if expected_block
            else "unexpected_preflight_outcome"
        ),
        "formal_detector_only_evaluation_performed": False,
        "detector_loaded": False,
        "detector_predict_called": False,
        "detector_track_called": False,
        "enhancer_preflight_performed": True,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "repository": {
            "path": str(repository),
            "commit": commit,
            "license": "MIT",
            "license_path": str(repository / "LICENSE.txt"),
            "license_sha256": sha256_file(repository / "LICENSE.txt"),
        },
        "checkpoint": {
            "path": str(weights),
            "bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
            "state_dict_keys": len(state),
            "strict_load_missing_keys": list(load_result.missing_keys),
            "strict_load_unexpected_keys": list(load_result.unexpected_keys),
        },
        "input": {
            "path": str(image_path),
            "bytes": image_path.stat().st_size,
            "sha256": sha256_file(image_path),
            "width": width,
            "height": height,
        },
        "elapsed_seconds": elapsed,
        "cuda_memory_before": {
            "free_bytes": int(memory_before[0]),
            "total_bytes": int(memory_before[1]),
        },
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "exception_type": (
            None if exception is None else type(exception).__name__
        ),
        "exception_message": None if exception is None else str(exception),
        "blocked_reason": (
            "The only permitted fallback cannot enhance one native LMOT "
            "1800x1000 frame through its official full-resolution path on "
            "the local 6 GB GPU. No alternate enhancer or post-result tiled "
            "variant was tried."
            if expected_block
            else None
        ),
    }
    (output / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "failure_cases.json").write_text(
        json.dumps(
            [
                {
                    "method": "GLARE",
                    "phase": "isolated_environment_preflight",
                    "status": "blocked",
                    "reason": (
                        "Required Python 3.8/PyTorch 1.11/CUDA 11.3 native "
                        "extension toolchain is unavailable locally."
                    ),
                },
                {
                    "method": "Retinexformer",
                    "phase": "full_resolution_enhancer_preflight",
                    "status": runtime["status"],
                    "input_shape": [height, width],
                    "exception_type": runtime["exception_type"],
                    "exception_message": runtime["exception_message"],
                },
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not expected_block:
        raise RuntimeError(
            "Retinexformer preflight did not reproduce the expected CUDA OOM; "
            "inspect the non-overwriting evidence output."
        )
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
