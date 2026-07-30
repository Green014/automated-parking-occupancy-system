# Automated Parking Lot Occupancy and Tracking System

This repository implements a research-grade parking-slot occupancy pipeline and
an optional identity-tracking variant. The final default is intentionally the
smallest configuration supported by the frozen external evidence:

`D1 detector -> B1 one-to-one polygon mapping -> E1b/F2 fusion -> occupancy`

E4 is off and tracking is `none` by default. TrackTrack is available only
through an explicit P3-TT command.

## Inputs

- a video, or the static-image evaluation inputs supported by the underlying
  tools;
- one parking-slot polygon file prepared once for each fixed camera;
- D1 detector weights;
- an E1b classifier checkpoint;
- optional frame-aligned slot truth for evaluation.

Model weights and datasets are intentionally not included in the portable
submission package.

## Outputs

The final video runtime writes:

- `annotated.mp4`
- `occupancy.csv`
- `events.csv`
- `detections.jsonl`
- `summary.json`
- `runtime_metadata.json`

P3-TT additionally writes `tracks.jsonl`. `metrics.json` is retained by the
generic P3-TT wrapper only when truth is supplied.

## Default system

Install the packages, then run:

```powershell
parking-run-final `
  --input <video> `
  --slots <slot-polygons.json> `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --output-dir <new-output-directory>
```

The default config is
`implementation/configs/p3_stage_r_recommended_default_20260729.yaml`.
Omitting optional flags preserves `temporal_enabled=false` and
`tracker_backend=none`.

## Controlled method comparison

Stage V.1 is a comparison tool, not the default runtime:

```powershell
parking-compare `
  --input <video-or-image-directory> `
  --slots <slot-polygons.json> `
  --mode compare `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --output-dir <new-output-directory>
```

It compares the independent Classic teaching baseline, D1+B1 Detection, and
D1+B1+E1b+F2 Fusion under controlled input/slot conditions. Frozen C1/C2
identity requires both the exact formal config SHA-256 and matching critical
parameters.

## Local dashboard

Stage W is a local demonstration interface around the unified backend:

```powershell
parking-dashboard `
  --input <video-path|camera-index|rtsp-url> `
  --slots <slot-polygons.json> `
  --mode fusion `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --output-dir <new-output-directory> `
  --host 127.0.0.1 `
  --port 5000
```

It is not a separate trained method. The existing four-frame loop is a
consumed interface demonstration with no truth and no accuracy or
continuous-video-validation claim.

## Optional generic P3-TT system

The portable generic command accepts any local video and does not require the
frozen VIRAT input:

```powershell
python implementation/scripts/run_p3_tt.py `
  --input <video> `
  --slots <slot-polygons.json> `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --source-id <camera-or-video-id> `
  --output-dir <new-output-directory>
```

Its flow is:

`D1 -> TrackTrack identity association -> B1 -> E1b/F2 -> occupancy + track IDs`

E4 is always off and TrackTrack is always explicitly enabled. Every call
constructs fresh tracker, event, and temporal state. Input, polygon, model, and
config SHA-256 values are recorded. Custom weights are permitted but are marked
`custom_weights=true` and cannot be compared with the frozen Stage T result.

The separate frozen Stage T experiment runner remains unchanged and continues
to enforce its consumed-development VIRAT hashes.

## Evidence summary

Stage S external low-light post-hoc component attribution:

| Detector | Final-compatible pipeline | Macro F1 |
|---|---|---:|
| D1 | B1 + F2 | 0.706681 |
| D1-LL | B1 + F2 | 0.666978 |

D1 remains the default. Its occupied recall is still only 0.370927, so the
system is not deployment-ready.

Stage T consumed-development diagnostic:

| Metric | TT0 | TT1 with TrackTrack |
|---|---:|---:|
| Macro F1 | 0.456797 | 0.456797 |
| Vacant recall | 0.000000 | 0.000000 |
| False-occupied rate | 1.000000 | 1.000000 |
| Median-frame FPS proxy | 33.521 | 14.215 |

TrackTrack produced identity-bearing output but did not improve slot occupancy
on this diagnostic and added substantial runtime cost. No slot-level occupancy
improvement is claimed.

## Demonstrations

- Default Stage S:
  `implementation/data/stage_s/demo/demo_main.mp4`
- Optional TrackTrack variant:
  `implementation/data/stage_t/demo/demo_tracktrack_optional.mp4`
- Clarified TrackTrack identity diagnostic:
  `implementation/data/stage_u_1/demo/demo_tracktrack_identity_diagnostic_presentation.mp4`

The frozen files decode as FMP4. Some PowerPoint/Windows combinations may
prefer H.264; this is a presentation compatibility consideration, not a model
or result issue. The frozen originals must not be replaced.

The Stage U.1 copy explicitly states that only one slot is evaluated, explains
the yellow TrackTrack-ID boxes and red/green predicted-state polygon, and marks
the known false-occupied failure from source frame 1660. It is a post-hoc
presentation copy of the frozen Stage T output, not a new experiment.

## Installation

Python 3.10–3.13 is supported by the package metadata.

```powershell
cd implementation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .\literature_core
.\.venv\Scripts\python.exe -m pip install -e ".[integrated,dashboard,dev]"
```

Official Stage N TrackEval reproduction additionally uses pinned commit
`12c8791b303e0a0b50f753af204249e622d0281a`. It is optional for the default and
generic P3-TT runtimes.

Required D1/E1b filenames, byte counts, hashes, and acquisition guidance are in
`implementation/data/STAGE_U_1_MODEL_ASSETS.md`. The source repository does
not bundle weights. The intended GitHub Release URL is pending; until a real
Release is published, supply each local path manually and verify its size and
SHA-256.

## Tests

Standard portable installation and tests, from `implementation`:

```powershell
python -m pytest tests -q
python -m pytest literature_core/tests -q
python -m compileall -q src scripts tests literature_core/src
python scripts/verify_stage_u_clean_package.py --no-record
```

Without TrackEval, its three official metric tests are explicitly skipped.
Stage U verifies the portable registry inside a clean package without datasets,
weights, virtual environments, or local runtime outputs.

To install and run the optional official TrackEval tests:

```powershell
python -m pip install -e ".[trackeval]"
python -m pytest tests/test_stage_n_lmot.py -m trackeval -q
```

The optional tests use TrackEval commit
`12c8791b303e0a0b50f753af204249e622d0281a`; they do not alter occupancy logic.

## Documentation

- [Final release index](FINAL_RELEASE_INDEX.md)
- [Historical system release index](implementation/data/SYSTEM_RELEASE_INDEX.md)
- [Stage S report](implementation/data/STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md)
- [Stage T report](implementation/data/STAGE_T_TRACKTRACK_ENHANCED_VARIANT_REPORT.md)
- [Stage U report](implementation/data/STAGE_U_PORTABLE_FINAL_RELEASE_REPORT.md)
- [Stage U.1 correction report](implementation/data/STAGE_U_1_FINAL_RELEASE_CORRECTION_REPORT.md)
- [Model asset requirements](implementation/data/STAGE_U_1_MODEL_ASSETS.md)
- [Stage W.3 public release index](implementation/data/STAGE_W_3_RELEASE_INDEX.md)
- [D1 model card](implementation/data/MODEL_CARD_D1.md)
- [E1b model card](implementation/data/MODEL_CARD_E1B.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Anonymous permission record](implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md)
- [Stage W.2 historical release index](implementation/data/STAGE_W_2_RELEASE_INDEX.md)

This is a reproducible local review candidate, not a deployment-ready parking
management product. Stage W.3 adds the full AGPL-3.0-only project license,
third-party notices, anonymous permission/provenance record, privacy-safe
source manifest, and independently prepared D1/E1b Release assets.
`public_release_published=false`: no remote push, GitHub Release, or public
asset URL is asserted by the preparation record. A local source commit does
not by itself mean that a public release exists.
