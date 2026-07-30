# Stage P — Parking-Domain Retention and Final Low-Light Occupancy Evidence

Date: 29 July 2026

Protocol: `STAGE-P-PARKING-DOMAIN-RETENTION-20260729-01`

Stage P2 decision: **FAIL**

Final night parking occupancy data gate: **BLOCKED**

## Executive conclusion

Stage O's D1-LL is **not supported as the default parking-domain detector**.
Under identical frozen inference settings on locally available NDISPark,
D1-LL regressed relative to D1 on the consumed night box diagnostic, the
daylight training-resubstitution diagnostic, the combined labelled set, and
the consumed night count diagnostic. The pre-registered Stage P2 decision is
therefore `FAIL`.

The Stage O result remains valid only within its measured scope: D1-LL was
the selected low-light detector candidate on LMOT road-scene raw
detector-only evaluation. That LMOT result did not establish parking-domain
retention or parking-slot occupancy improvement. Stage P supplies negative
parking-domain retention evidence and leaves the existing P3 D1 default
unchanged.

No qualifying new night fixed-camera parking source with stable slot
polygons and per-slot occupied/vacant truth was found locally. Stage P4 was
not run, no P3-D1/P3-LL occupancy metrics were calculated, and
`p3_ll_integrated_runtime_defaults_20260729.yaml` was not created.

## Frozen comparison

Both models were evaluated with exactly the same settings:

- `ultralytics.YOLO.predict`, never `model.track`;
- `imgsz=640`, confidence `0.30`, NMS IoU `0.70`;
- `agnostic_nms=true`, `max_det=300`, class `[0]`;
- no augmentation, no rect mode, batch size 1;
- no threshold, sequence, image, or component reselection.

D1 used SHA-256
`0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`.
D1-LL used SHA-256
`99b658bba0ef117d3206b85fc982c81cc0b94839932bfb9e99780027bab1c5da`.
Each model made 259 `predict` calls. Runtime metadata records
`model_track_called=false`, `tracker_loaded=false`, and
`training_performed=false`.

The box metric named below is:

> confidence-truncated AP at the frozen confidence threshold of 0.30

It is not standard COCO AP and must not be compared directly with NDISPark
historical curves generated from a 0.001 confidence floor or with
paper-reported standard AP. NDISPark box truth contains only mapped vehicle
class 0 and has no ignore-region schema. No ground-truth-derived
ignored-class suppression was applied to either model.

## Stage P1 results

### Night validation — consumed-development diagnostic

This 30-image, 725-box subset was previously used in D1/Stage I development.
It is not an untouched test.

| Model | Precision | Recall | Truncated AP50 @ 0.30 | Truncated AP50-95 @ 0.30 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.9015 | 0.9090 | 0.8946 | 0.6454 | 659 | 72 | 66 |
| D1-LL | 0.8647 | 0.8290 | 0.8009 | 0.5296 | 601 | 94 | 124 |
| D1-LL − D1 | -0.0368 | -0.0800 | -0.0937 | -0.1159 | -58 | +22 | +58 |

D1-LL did not reproduce its LMOT low-light advantage in this parking-domain
diagnostic. It produced fewer true positives, more false positives, and
nearly twice as many false negatives.

### Daylight train — training-resubstitution diagnostic

All 112 images and 2,577 boxes trained both D1 and D1-LL. These numbers
measure fit/retention on seen training data, not generalization.

| Model | Precision | Recall | Truncated AP50 @ 0.30 | Truncated AP50-95 @ 0.30 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.9018 | 0.9158 | 0.9042 | 0.6619 | 2,360 | 257 | 217 |
| D1-LL | 0.8761 | 0.8231 | 0.8042 | 0.5411 | 2,121 | 300 | 456 |
| D1-LL − D1 | -0.0257 | -0.0927 | -0.1000 | -0.1208 | -239 | +43 | +239 |

Even on seen daylight parking images, D1-LL showed substantial recall and AP
regression.

### All labelled NDISPark images — descriptive pooled/micro result

The aggregate sums GT, prediction, TP, FP, and FN counts before computing
pooled rates. Per-camera macro values are retained separately in
`metrics.json`.

| Model | Precision | Recall | Truncated AP50 @ 0.30 | Truncated AP50-95 @ 0.30 | GT | Pred | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.9017 | 0.9143 | 0.9040 | 0.6580 | 3,302 | 3,348 | 3,019 | 329 | 283 |
| D1-LL | 0.8736 | 0.8243 | 0.8034 | 0.5378 | 3,302 | 3,116 | 2,722 | 394 | 580 |
| D1-LL − D1 | -0.0282 | -0.0899 | -0.1006 | -0.1202 | 0 | -232 | -297 | +65 | +297 |

### Night test count — consumed post-hoc diagnostic

The 117 night test images have count truth but no box truth. Count MAE/RMSE
are not detector AP, box precision/recall, or occupancy metrics.

| Model | Count MAE | Count RMSE | Mean predicted | Mean truth |
|---|---:|---:|---:|---:|
| D1 | 1.5299 | 2.2322 | 10.7265 | 11.9829 |
| D1-LL | 2.0427 | 3.0028 | 10.6239 | 11.9829 |
| D1-LL − D1 | +0.5128 | +0.7706 | -0.1026 | 0 |

### Descriptive runtime

The run used the local NVIDIA GeForce RTX 3060 Laptop GPU, Python 3.12.13,
PyTorch 2.13.0+cu130, Ultralytics 8.4.104, and no A100. Measured end-to-end
model-loop rates were 21.22 FPS for D1 and 29.26 FPS for D1-LL, with peak
allocated CUDA memory of 49,754,112 and 46,016,000 bytes respectively.
These values include warm-up/order effects and are descriptive only; runtime
was not a selection criterion.

## Stage P2 decision

The rule was frozen before predictions. D1-LL failed the required night
Recall-or-AP50 improvement check and the night AP50-95 safety check. It also
failed daylight Recall/AP retention, combined AP50-95 retention, and the
allowed count-MAE increase. The resulting status is **FAIL**.

This means:

- D1 remains the frozen P3 default;
- D1-LL is retained only as the Stage O selected low-light detector
  candidate and as a documented negative parking-retention result;
- the result must not trigger threshold changes, more low-light model search,
  or a P3-LL default configuration.

## Stage P3/P4 final occupancy gate

The local audit found no eligible final night parking occupancy source:

- NDISPark has fixed night images and vehicle boxes/counts, but no parking
  polygons or per-slot truth and is already consumed;
- PKLot has static slot truth but is daylight, non-continuous, and consumed;
- Grand Bassin lacks verified night vacant/transition truth and is consumed;
- VIRAT local cases are daytime, consumed, and include a known geometry
  association failure;
- the Stage O P3 smoke repeats a consumed image and has no truth.

Therefore the final data gate is **BLOCKED**. Interface-only Stage O evidence
is not occupancy evidence. Stage P4 outputs such as annotated video,
occupancy/events CSVs, and transition metrics were intentionally not
generated.

The minimum material needed to unblock P4 is:

1. a path to a new night/low-light fixed-camera parking video or ordered
   frame directory;
2. stable slot polygons with source dimensions and pixel coordinates;
3. per-frame or timestamped per-slot states using occupied, vacant, and
   optional unknown (excluded) labels;
4. FPS/timestamps for transition evaluation;
5. source/scene description and permission or license.

Schema-only examples are frozen in
`STAGE_P_FINAL_NIGHT_PARKING_DATA_GATE_20260729.yaml`; they are not fabricated
real labels.

## Evidence classification

### Experimentally supported

- At the frozen 0.30 operating threshold, D1 outperformed D1-LL on the
  available NDISPark night box, daylight resubstitution, combined box, and
  night count diagnostics.
- D1-LL does not pass the pre-registered parking-domain retention rule.
- Both models were compared through raw `YOLO.predict` with identical
  settings and no tracker or retraining.

### Consumed-development or retrospective diagnostic

- all NDISPark results in this report;
- daylight train results are additionally training-resubstitution evidence;
- night validation boxes and night test counts were previously consumed.

### Interface-only evidence

- Stage O's P3 smoke remains interface-only and is not reused as evaluation.
- Stage P created no new P3 interface result because a smoke cannot resolve
  the occupancy evidence gate.

### Blocked final evaluation

- P3-D1 versus P3-LL slot occupancy and temporal transition comparison;
- final-night Macro F1, occupied/vacant recall, false-free/false-occupied
  rates, per-frame occupancy count error, and signed transition error.

### Not supported

- that LMOT detection gains imply parking occupancy gains;
- that D1-LL generally improves parking-domain detection;
- that D1-LL should replace D1 in the P3 defaults;
- any low-light slot occupancy or transition-performance claim;
- comparison of the reported confidence-truncated AP with standard COCO AP.

## Reproducibility and outputs

The additive formal output directory is
`implementation/outputs/stage_p_ndispark_retention_20260729_v1`. It contains
raw predictions, per-image statistics, per-model and comparison metrics,
runtime metadata, source-image hash verification, qualitative case records,
and a D1/D1-LL contact sheet. Existing Stage L, M, N, and O files were not
rewritten.

The qualitative sheet uses green for available ground-truth boxes and red
for frozen-threshold predictions. Its examples are selected by a fixed
per-image error comparison and are illustrative retrospective evidence, not
an additional selection set.

Repository-wide validation completed with 203/203 implementation tests and
83/83 `literature_core` tests passing (286/286 combined). The 16 Stage P
tests are included in that total. Full `compileall`, `git diff --check`, the
additional whitespace scan for untracked Stage P files, and the 16-file
formal output contract all passed. The unchanged Stage O registry reverified
213/213 artifacts with registry SHA-256
`efbbb63b77aefae00c1c4758b8df1dd463c2fd4a54bc32626ebf441073b87660`.

The final Stage P registry contains 39 non-recursive records: implementation,
tests, protocol/gates/report, formal outputs, frozen model weights, and bound
input manifests/configuration. The registry is intentionally not a member of
itself; its final SHA-256 is reported in the delivery handoff.
