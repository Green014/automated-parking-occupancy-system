# Stage I detector evaluation report

## Outcome

Stage I is complete under the frozen NDISPark protocol. D1 is the selected
detector for the parking pipeline because it led the consumed development
validation on the predeclared primary metric, mAP@0.5:0.95, and on recall.
The separate count-only test is a retained negative result: under the one
shared development-selected confidence rule, D2—not D1—had the lowest MAE.
No post-test threshold change or model reselection was made.

## Frozen development comparison

All models used the same 30 night validation images, 725 human vehicle boxes,
image size 640, NMS IoU 0.7, maximum 300 detections, single canonical
`vehicle` class, and the same local RTX 3060. This split is consumed
development validation, not an untouched test.

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ms/image | FPS |
|---|---:|---:|---:|---:|---:|---:|
| D0 COCO-pretrained YOLOv8n | 0.59394 | 0.51648 | 0.55852 | 0.28468 | 25.818 | 38.733 |
| D1 NDISPark-fine-tuned YOLOv8n | **0.88153** | **0.84160** | **0.89910** | **0.64969** | 26.474 | 37.773 |
| D2 YOLO-World zero-shot | 0.76736 | 0.61655 | 0.72963 | 0.39704 | 26.390 | 37.893 |

D1 improved mAP@0.5:0.95 by 0.36501 and recall by 0.32512 relative to D0
at similar framework pipeline speed. This comparison result differs slightly
from the training-loop best-epoch diagnostic because Stage I uses a common
canonical evaluator and class mapping for D0/D1/D2.

The confidence floor of 0.001 exists to construct PR/AP curves. At that floor
D1 reaches the common `max_det=300` cap on these dense frames; the declared
operating rule removes low-confidence predictions before counting and
qualitative analysis. The cap and floor are retained rather than hidden.

## Detector selection and shared counting rule

Before any test prediction, D1 was selected using:

1. higher mAP@0.5:0.95;
2. higher recall as tie-break;
3. recorded FPS as the deployment constraint.

A single counting confidence was selected from
`0.05, 0.10, ..., 0.95` using only the same consumed development data. The
objective minimized the mean count MAE across every D0/D1/D2 image; ties
would use lower worst-model MAE and then the higher threshold. The selected
confidence was 0.10 and applies identically to all three detectors. Per-model
thresholds and test-count tuning were prohibited.

## Count-only test

The frozen NDISPark official test has 117 night images, six cameras, and
vehicle counts only. It has no vehicle-box truth, so detector mAP, box
precision, box recall, FP boxes, and FN boxes are not reported on this split.

| Method | MAE | RMSE | Mean predicted | Mean true | ms/image | FPS |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 2.99145 | 5.30119 | 11.16239 | 11.98291 | 9.217 | 108.492 |
| D1 | 3.46154 | 6.78800 | 14.62393 | 11.98291 | 7.855 | 127.315 |
| D2 | **2.58974** | **4.98631** | 10.09402 | 11.98291 | 60.887 | 16.424 |

D1 over-counted on average under the shared 0.10 rule, while D2 under-counted
but achieved the lowest absolute and squared count errors. This does not
invalidate the pre-test D1 box-metric selection; it shows that improved box
AP does not automatically produce the best raw count calibration. D2's count
advantage also carries an approximately 7.8x latency cost relative to D1 in
this run.

Per-camera MAE preserves the six official source groups:

| Camera | D0 | D1 | D2 |
|---|---:|---:|---:|
| 60 | 3.55556 | 4.72222 | **3.11111** |
| 64 | 4.90323 | 5.38710 | **3.80645** |
| 69 | 1.73684 | 1.52632 | **1.10526** |
| 73 | **1.94444** | 4.27778 | 2.94444 |
| 78 | 1.06250 | **0.50000** | 0.75000 |
| 83 | 3.26667 | **2.60000** | 2.86667 |

The machine-readable count reports include MAE and RMSE per camera. MAPE is
not used, and count metrics are never described as detection mAP.

## Qualitative evidence

FP/FN overlays use only the consumed validation split where human box truth
exists. Green denotes a matched prediction, red an unmatched prediction, and
orange an unmatched truth box at IoU 0.5. The exported representative,
false-positive, and false-negative montages were visually reviewed.

The montages show D1's large recall gain, especially in dense rows, but also
show duplicate and low-confidence false positives exposed by the shared
0.10 rule. The file named `night_occlusion_candidate_comparison.png` uses the
densest annotated frame only as a dense/overlap review candidate. NDISPark
provides no official occlusion tag here, so it is not labeled occlusion ground
truth.

The formal D1 `results.png` is the training-curve artifact. Each detector's
Stage I directory contains PR, precision, recall, F1, raw confusion matrix,
normalized confusion matrix, JSONL detections, metrics, and runtime metadata.

## Retained failure and reproducibility

The first count-run directory is retained with only its preflight record. It
failed before the first model prediction because Ultralytics attempted to
write a non-writable user settings directory. The v2 run changed only
`YOLO_CONFIG_DIR`, directing settings into the new output directory; data,
weights, image size, NMS, and confidence remained frozen.

The full record is
`data/comparisons/stage_i_detector_evaluation_20260727.yaml`. Generated output
is ignored by Git and is verified by size and SHA-256. Historical CNR-EXT,
PKLot, VIRAT, temporal, and training outputs were not modified.

The initially typed rounded `frozen_at/completed_at` values are explicitly
non-authoritative. Their correction sidecar,
`data/comparisons/stage_i_timestamp_correction_20260727.yaml`, records actual
filesystem timestamps and confirms the required order: selection at
20:50:47, protocol file at 20:51:41, and test result at 20:54:03 (UTC+08:00).
No configuration, prediction, metric, or generated artifact changed during
this metadata correction.

Final verification passed:

- implementation tests: 87/87;
- literature-core tests: 82/82;
- Python AST parse: 137/137 files;
- YAML/JSON parse: 37/37 YAML and 7/7 JSON files;
- Stage I selected artifacts: 24/24;
- historical static frozen artifacts: 17/17 plus 4,081-frame and
  144,965-slot count checks;
- frozen temporal artifacts: 11/11;
- `git diff --check`: passed (line-ending conversion warnings only).

## Scientific boundary and next gate

D1 is selected for P1, while D0 and D2 remain the P0/P2 comparators. Stage J
must use identical B1 polygon mapping and output contracts to isolate the
detector change. A new untouched slot-occupancy test is not currently
available, so Stage J may run development integration and prepare the gate but
must not claim a final slot Macro-F1 improvement.
