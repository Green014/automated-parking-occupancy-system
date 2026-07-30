from __future__ import annotations

import argparse
import atexit
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .integrated_cli import DEFAULT_FINAL_INTEGRATED_CONFIG
from .stage_v_runner import build_backends
from .stage_w_member_reference import MemberReferenceBackend
from .stage_w_server import (
    StageWErrorProcessor,
    StageWProcessor,
    create_stage_w_app,
)
from .stage_w_ui_adapter import redact_source


STAGE_W_MODES = ("classic", "detection", "fusion", "member-reference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage W Flask dashboard for unified parking occupancy"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--mode", choices=STAGE_W_MODES, default="fusion")
    parser.add_argument("--d1-weights", type=Path)
    parser.add_argument("--e1b-checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FINAL_INTEGRATED_CONFIG,
    )
    parser.add_argument("--allow-custom-config", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack", "tracktrack"),
        default="none",
    )
    parser.add_argument("--tracker-config", type=Path)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--classic-threshold", type=float, default=0.30)
    parser.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--member-reference-root", type=Path)
    parser.add_argument("--member-reference-config", default="config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--allow-remote-bind", action="store_true")
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Process the input and write smoke artifacts without starting Flask.",
    )
    return parser


def _default_output() -> Path:
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"stage_w_dashboard_{identifier}"


def _build(args: argparse.Namespace):
    if args.mode == "member-reference":
        if args.member_reference_root is None:
            raise ValueError(
                "--member-reference-root is required for member-reference mode"
            )
        backend = MemberReferenceBackend(
            args.member_reference_root,
            config_name=args.member_reference_config,
        )
        classifier_enabled = bool(
            backend.config.get("slot_classifier", {}).get("enabled", False)
        )
        snapshot = {
            "method_identity": "member-reference",
            "frozen_parameters_changed": False,
            "member_reference": {
                "external_dependency": True,
                "audited_commit": backend.commit,
                "config_filename": args.member_reference_config,
                "public_redistribution_confirmed": False,
            },
            "temporal": {"enabled": False, "component": "member-defined"},
            "tracking": {
                "backend": "none" if classifier_enabled else "member-bytetrack",
                "default": "member-defined",
            },
            "artifacts": [],
        }
        return backend, snapshot
    backends, snapshot = build_backends(
        mode=args.mode,
        config_path=args.config.resolve(),
        d1_weights=args.d1_weights,
        e1b_checkpoint=args.e1b_checkpoint,
        device=args.device,
        tracker_backend=args.tracker,
        tracker_config_override=args.tracker_config,
        classifier_batch_size=args.classifier_batch_size,
        classic_threshold=args.classic_threshold,
        temporal_enabled=args.temporal,
        allow_custom_config=args.allow_custom_config,
    )
    return backends[args.mode], snapshot


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.classifier_batch_size <= 0:
        raise ValueError("classifier-batch-size must be positive")
    if args.tracker_config is not None and args.tracker == "none":
        raise ValueError("--tracker-config requires --tracker")
    if args.mode == "classic" and args.tracker != "none":
        raise ValueError("Classic mode does not support a vehicle tracker")
    if args.temporal and args.mode != "fusion":
        raise ValueError("E4 is available only for Fusion")
    if (
        args.host not in {"127.0.0.1", "localhost", "::1"}
        and not args.allow_remote_bind
    ):
        raise ValueError(
            "Remote binding requires explicit --allow-remote-bind"
        )
    output_dir = args.output_dir or _default_output()
    try:
        backend, snapshot = _build(args)
        processor: StageWProcessor | StageWErrorProcessor = StageWProcessor(
            source=args.input,
            slots_path=args.slots,
            backend=backend,
            mode=args.mode,
            config_snapshot=snapshot,
            output_dir=output_dir,
            max_frames=args.max_frames,
            warmup=args.warmup,
        )
    except Exception as exc:
        if args.no_serve:
            raise
        processor = StageWErrorProcessor(
            f"Backend unavailable: {exc}",
            mode=args.mode,
            source=args.input,
        )
    atexit.register(processor.stop)
    if args.no_serve:
        assert isinstance(processor, StageWProcessor)
        processor.run_blocking()
        result = processor.status()
        print(
            json.dumps(
                {
                    "stage": "W",
                    "health": result["health"],
                    "mode": args.mode,
                    "source": redact_source(args.input),
                    "frames": (
                        None
                        if result["frame_index"] is None
                        else int(result["frame_index"]) + 1
                    ),
                    "output_dir": str(output_dir),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if result["health"] == "error":
            raise SystemExit(1)
        return
    processor.start()
    app = create_stage_w_app(processor)
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
