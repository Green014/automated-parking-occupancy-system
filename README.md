# Automated Parking Lot Occupancy and Tracking System

This repository contains the implementation and reproducibility material for a
fixed-camera parking occupancy project. It is a coursework research prototype.
It is not deployment-ready.

## Final workflow

```text
Video or image sequence
        |
        v
D1: parking-domain fine-tuned YOLOv8n vehicle detector
        |
        v
B1: confidence-weighted polygon overlap with one-to-one assignment
        |
        v
E1b: MobileNetV3-CBAM review of detector-negative slots
        |
        v
F2: asymmetric uncertainty-gated fusion
        |
        v
Per-slot occupancy states, events, metrics, and annotated video
```

The final default is `D1 -> B1 -> F2`, with temporal E4 and tracking disabled.
An optional research variant inserts TrackTrack after D1. The optional tracker
has not demonstrated a slot-occupancy improvement and is not the default.

In compact form, the final path is:

```text
D1 detector -> B1 one-to-one polygon mapping -> E1b/F2 -> occupancy output
```

## Inputs and outputs

Required inference inputs:

- a fixed-camera video or image sequence;
- a JSON parking-slot polygon map;
- D1 detector weights;
- an E1b classifier checkpoint.

The system does not automatically discover parking-slot locations. Ground truth
is optional for inference and required only when computing evaluation metrics.

Standard outputs include:

- `annotated.mp4`;
- `occupancy.csv`;
- `events.csv`;
- `detections.jsonl`;
- `summary.json`;
- `metrics.json` when truth is supplied;
- `runtime_metadata.json`.

## Repository layout

```text
implementation/
  src/parking_occupancy/  Main detector, mapping, fusion, tracking, and CLI code
  literature_core/src/    Runtime classifier, patch, metric, and temporal modules
  configs/                Frozen and generic experiment configurations
  scripts/                Training, evaluation, audit, and release entry points
  tests/                  Unit and integration tests
  data/                   Compact reports, manifests, and machine-readable evidence
```

`implementation/literature_core` is retained because the final runtime imports
code from that Python package. It does not contain the literature-review PDF
collection in this public repository.

## Installation

Python 3.10 to 3.13 is supported. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\implementation\literature_core
python -m pip install -e ".\implementation[integrated,dev]"
```

CUDA is optional for code inspection and tests. A CUDA-capable GPU is
recommended for practical video inference.

## Model assets

Model weights are project-produced artifacts and are intentionally not stored
in Git. Obtain the files from the project artifact owner and verify them before
reproducing frozen results:

| Asset | Suggested filename | Bytes | SHA-256 |
|---|---|---:|---|
| D1 detector | `D1_NDISPark_best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| E1b classifier | `E1b_CBAM_best.pt` | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

See
[`implementation/data/STAGE_U_1_MODEL_ASSETS.md`](implementation/data/STAGE_U_1_MODEL_ASSETS.md)
for the frozen asset boundary.

## Run the final workflow

After installation:

```powershell
parking-run-final `
  --input <video.mp4> `
  --slots <slot-map.json> `
  --d1-weights <D1_NDISPark_best.pt> `
  --e1b-checkpoint <E1b_CBAM_best.pt> `
  --tracker none `
  --output-dir <new-output-directory>
```

Add `--truth <slot-state.csv>` only when evaluation labels exist. Use
`--tracker tracktrack` only for the optional identity-enhanced variant.

The equivalent generic optional runner is:

```powershell
python implementation/scripts/run_p3_tt.py --help
```

The full CLI and experiment history are documented in
[`implementation/README.md`](implementation/README.md).

## Verification

Run the main implementation tests from `implementation`:

```powershell
cd implementation
$env:STAGE_U_PORTABLE_PACKAGE = "1"
$env:PARKING_PUBLIC_SOURCE_PACKAGE = "1"
python -m pytest -q
python -m pytest literature_core/tests -q
python -m compileall -q src scripts tests literature_core/src
```

The two environment variables classify tests that require local frozen
registries, datasets, model weights, or presentation media as explicit skips.
The test functions remain in the repository and can be enabled in a complete
local archive. Optional TrackEval tests also skip unless TrackEval is installed.

## Result boundary

The final external night comparison reports D1 with B1 + F2 at Macro F1
`0.706681`, with occupied recall `0.370927`. D1 remains the default, but this
result is not evidence of deployment readiness. The TrackTrack consumed-
development diagnostic produced no slot-level Macro F1 improvement and reduced
the median-frame FPS proxy from `33.521` to `14.215`.

Start with:

- [`implementation/data/FINAL_RESULTS_INDEX.md`](implementation/data/FINAL_RESULTS_INDEX.md)
- [`implementation/data/SYSTEM_RELEASE_INDEX.md`](implementation/data/SYSTEM_RELEASE_INDEX.md)
- [`implementation/data/STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md`](implementation/data/STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md)
- [`implementation/data/STAGE_T_TRACKTRACK_ENHANCED_VARIANT_REPORT.md`](implementation/data/STAGE_T_TRACKTRACK_ENHANCED_VARIANT_REPORT.md)

## Public release boundary

This repository excludes:

- literature-review papers and office documents;
- downloaded datasets and extracted images;
- model checkpoints;
- local experiment output directories;
- virtual environments and caches;
- large presentation videos and contact sheets;
- machine-specific absolute-path registries.

No license is granted by this repository at present. Public visibility is
provided for technical inspection, reproducibility review, and coursework
assessment. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for
dependency and redistribution notes.
