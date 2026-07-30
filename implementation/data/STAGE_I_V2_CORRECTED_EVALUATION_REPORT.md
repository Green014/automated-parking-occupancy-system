# Stage I-v2 corrected detector evaluation

Stage I v1 remains the immutable historical experiment. Stage I-v2 is a
methodology correction on the already consumed NDISPark development
validation, followed by one explicitly post-hoc sensitivity analysis on the
already consumed NDISPark count-only test. It is not a new untouched test.

## Corrections

Stage I v1 ran NMS by source class and then mapped `car`, `motorcycle`, `bus`,
and `truck` to one project class. That ordering could retain two highly
overlapping source-class boxes for one vehicle. Stage I-v2 explicitly passes
`agnostic_nms=true` to Ultralytics before canonicalization and applies a
project-side class-agnostic safety check before emitting class 0 `vehicle`.

Stage I v1 also used one development-selected confidence threshold, 0.10, for
all detectors. This is retained as a controlled diagnostic but is not the
primary deployment operating point. Stage I-v2 selects each detector's
threshold on exactly the same 30 development images and the same
`0.05:0.05:0.95` grid. The objective is lowest count MAE, with lower RMSE and
then higher threshold as tie-breaks:

| Method | Development-selected confidence | Development count MAE | RMSE |
|---|---:|---:|---:|
| D0 | 0.10 | 6.8000 | 9.5673 |
| D1 | 0.30 | 1.7000 | 2.7264 |
| D2 | 0.10 | 4.9667 | 7.1204 |

The common-threshold diagnostic still selects 0.10. Its role is explicitly
`controlled_sensitivity_only`.

## Corrected development comparison

At the retained `max_det=300` arm:

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Cap-hit images | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.63669 | 0.53379 | 0.59954 | 0.30672 | 4/30 | 676,204,032 B |
| D1 | **0.88590** | **0.84606** | **0.90653** | **0.65577** | 30/30 | 652,165,120 B |
| D2 | 0.79014 | 0.62069 | 0.74025 | 0.40295 | 5/30 | 2,248,857,600 B |

D1 remains the detector selected by the predeclared mAP@0.5:0.95-then-recall
rule. This selection was made before the post-hoc count sensitivity.

The paired `max_det=1000` arm kept the same D1 > D2 > D0 ranking:

| Method | mAP@0.5:0.95 at 300 | mAP@0.5:0.95 at 1000 | Cap hits at 300 | Cap hits at 1000 |
|---|---:|---:|---:|---:|
| D0 | 0.30672 | 0.30372 | 4/30 | 0/30 |
| D1 | 0.65577 | 0.65243 | 30/30 | 30/30 |
| D2 | 0.40295 | 0.40172 | 5/30 | 0/30 |

The frozen decision therefore retains 300. D1 remaining saturated on all 30
images even at 1000 is a limitation of evaluating at the very low 0.001
confidence floor, not evidence that a larger deployment cap is beneficial.
The 1000 arm did not improve the ranking or primary metrics.

## Consumed-test post-hoc sensitivity

All v2 settings, evidence hashes, and operating points were frozen before the
single corrected prediction pass on the 117-image count-only split. The output
is labelled `consumed_test_posthoc_sensitivity`; no detector was reselected and
no threshold, NMS, or max-det setting was changed afterward.

| Method | Common 0.10 MAE | Common RMSE | Per-model MAE | Per-model RMSE |
|---|---:|---:|---:|---:|
| D0 | 2.96581 | 5.32291 | 2.96581 | 5.32291 |
| D1 | 3.45299 | 6.74949 | **1.52991** | **2.23224** |
| D2 | 2.58974 | 4.98288 | 2.58974 | 4.98288 |

The common-threshold outcome remains a negative result for D1 and is very
close to v1. Under the independently development-calibrated operating points,
D1's over-counting is substantially reduced. Because this split was already
viewed in Stage I v1, the latter is only a post-hoc corrected sensitivity
result. It does not create a new test claim and was not used to revise the
method.

The post-hoc pass predicted once at confidence floor 0.001 and computed both
regimes offline from the same detections. Its latency therefore includes
processing the low-confidence candidate set and is not a clean deployment
latency measurement at thresholds 0.10 or 0.30.

## Difference from Stage I v1

| Source | Stage I v1 | Stage I-v2 |
|---|---|---|
| NMS | source-class NMS before class merge | class-agnostic NMS before class merge |
| Count operating point | common 0.10 | common 0.10 diagnostic plus D0 0.10 / D1 0.30 / D2 0.10 primary points |
| max_det evidence | 300 only | paired development sensitivity at 300 and 1000; 300 retained |
| NDISPark test role | originally run as count-only test, now consumed | explicitly post-hoc/consumed-test sensitivity |
| Downstream model selection | D1 before test | D1 remains frozen before post-hoc result |

Count MAE/RMSE measure whole-image vehicle counts. They are not detector mAP,
parking-slot occupancy accuracy, or slot Macro F1.

## Evidence and reproducibility

- Frozen configs:
  `configs/detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml`,
  `configs/detector_comparison_stage_i_v2_maxdet1000_frozen_20260727.yaml`,
  and `configs/stage_i_v2_posthoc_count_frozen_20260727.yaml`.
- Source/output registry:
  `data/comparisons/stage_i_v2_corrected_evaluation_20260727.yaml`.
- Ignored generated outputs:
  `outputs/detector_comparison_stage_i_v2_maxdet300_20260727_v2/`,
  `outputs/detector_comparison_stage_i_v2_maxdet1000_20260727_v1/`, and
  `outputs/detector_count_test_stage_i_v2_posthoc_20260727_v1/`.
- Verification:
  `outputs/stage_i_v2_artifact_verification_20260727_v1.json`, 44/44 passed.

The prematurely timed-out max-det=300 execution directory contains only its
preflight and is retained as negative engineering evidence. The completed
retry uses a new v2 output directory. No v1 output was overwritten.
