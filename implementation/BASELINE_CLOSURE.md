# Baseline Closure

Updated: 26 July 2026

This document closes the naming gap between the original baseline package and
the later `literature_core` experiments. It does not rename any historical
file, directory, CSV column, or frozen result.

The machine-readable source of truth for new runs is
`configs/baseline_methods.yaml`.

## Canonical naming

| ID | Canonical meaning | Detector settings | Mapping | Data role | Run policy |
|---|---|---|---|---|---|
| **B0** | YOLOv8 bounding-box-centre baseline | `yolov8n.pt`, `conf=0.025`, `imgsz=1280`, COCO vehicle classes 2/3/5/7 | Box centre inside slot polygon; greedy one-to-one assignment | Development baseline, or future data frozen under a new protocol | Runnable with `parking-run --method B0` |
| **B1** | YOLOv8 polygon-coverage baseline | Same as B0 | Confidence-weighted slot coverage, minimum coverage 0.40; greedy one-to-one assignment | Development baseline, or future data frozen under a new protocol | Runnable with `parking-run --method B1` |
| **E0** | Historical static external YOLOv8 coverage baseline | `yolov8n.pt`, `conf=0.025`, `imgsz=1280`, classes 2/3/5/7 | Confidence-weighted coverage of official CNR-EXT boxes, minimum coverage 0.40; one-to-one | Consumed once-only CNR-EXT static external holdout | Historical-only; use the frozen config and artifacts |
| **T0** | Raw YOLOv8 temporal comparator | `yolov8n.pt`, `conf=0.20`, `imgsz=640`, classes 2/3/5/7 | Confidence-weighted slot-polygon coverage, minimum coverage 0.30; one-to-one; no dwell or hysteresis | Temporal comparator on a future, separately frozen legal video protocol | Runnable with `parking-run --method T0` |

## Relationships and non-equivalences

- B0 and B1 intentionally share the same detector. Their controlled difference
  is centre mapping versus polygon coverage.
- E0 uses the same detector family and a coverage rule, but it is not a
  rename of B1. Its official axis-aligned CNR-EXT slot boxes, once-only
  external data role, frozen configuration, and evaluation runner are
  different.
- The historical temporal case-study CSV/JSON files call T0 `e0_raw`. That
  artifact key is retained for reproducibility. In reports written after this
  closure, the method is **T0 (historical artifact key `e0_raw`)**.
- The frozen temporal runner attached ByteTrack IDs to retained raw YOLOv8
  boxes because the same detector adapter also fed E5. T0 state decisions did
  not use those IDs, dwell, motion suppression, or hysteresis. The canonical
  T0 entry point therefore runs raw detection and mapping without tracking.
- The old baseline experiment named `proposed` is a historical engineering
  variant: YOLOv8 + ByteTrack IDs + polygon coverage + hysteresis. It is not
  the converged Part II main workflow and is not silently relabelled as F2.

## Historical artifact boundary

E0 is defined by
`literature_core/configs/external_holdout_frozen.yaml` and
`literature_core/outputs/cnrpark_ext_frozen_evaluation_20260725/`. CNR-EXT has
already been consumed and cannot select new thresholds, fusion gates, or
other parameters.

The T0 holdout result is embedded in the frozen VIRAT case-study outputs and
uses the historical key `e0_raw`. VIRAT 0503 has been viewed and consumed; it
cannot be presented as an untouched holdout for a new method.

No file under either frozen output directory was changed to create this
closure.

## Unified runnable entry point

From `implementation/`:

```powershell
.\.venv\Scripts\parking-run.exe `
  --input data\raw\video\camera01.mp4 `
  --slots data\annotations\camera01_slots.json `
  --method B1 `
  --output-dir outputs\camera01_b1_closure
```

Replace `B1` with `B0` or `T0`. A canonical `--method` run loads detector and
mapping settings from the registry and rejects command-line overrides of
`weights`, `conf`, `imgsz`, and coverage threshold. This prevents an
unlabelled configuration from being reported under a canonical ID.

The legacy `--experiment b0|b1|proposed|t0` interface remains available for
engineering compatibility. Such a run is custom/legacy unless its summary
contains a registered canonical method ID.

Every normal video run writes:

- `annotated.mp4`;
- `occupancy.csv`;
- `events.csv`;
- `summary.json`.

The summary records the canonical ID/name, registry path, data role, detector
metadata, mapping mode, one-to-one policy, coverage threshold, input and slot
map hashes, timings, and output paths. `--no-video` is an explicit diagnostic
override and intentionally omits `annotated.mp4`.

## Reporting rule

Tables and captions must use B0, B1, historical E0, and T0 exactly as defined
above. When citing an old file, include its old key or directory in
parentheses rather than renaming the artifact.
