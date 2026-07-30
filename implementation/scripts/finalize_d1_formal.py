from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.formal_training import finalize_existing_formal_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the completed Stage H v1 artifacts after the retained "
            "post-run callback audit failure. No model is loaded or run."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze-registry", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--launcher-stdout", required=True)
    parser.add_argument("--launcher-stderr", required=True)
    args = parser.parse_args()
    report = finalize_existing_formal_run(
        config_path=Path(args.config),
        freeze_registry_path=Path(args.freeze_registry),
        output_dir=Path(args.output_dir),
        launcher_stdout=Path(args.launcher_stdout),
        launcher_stderr=Path(args.launcher_stderr),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
