# Stage L Integrated Literature Workflow Report

Date: 28 July 2026

Protocol: `P3-INTEGRATED-LITERATURE-WORKFLOW-20260728-01`

## 1. Objective and method

Stage L tested whether the previously separate Part I-derived components can
form one runnable parking-occupancy workflow:

```text
fixed camera frame
  -> D1 NDISPark-fine-tuned YOLOv8n
  -> B1 polygon coverage with one-to-one assignment
  -> E1b MobileNetV3-CBAM review of detector-negative slots
  -> F2 asymmetric uncertainty gate
  -> E4 asymmetric EMA and hysteresis on continuous video
  -> optional E5 ByteTrack moving-vehicle suppression
  -> per-slot states, events, detections and annotated video
```

F2 is detector-first. A mapped vehicle is occupied and E1b cannot overturn
it. A detector-negative slot is uncertain and E1b recovers it only when the
frozen occupied probability is at least 0.76. This is a local modification
combining known components, not a reproduction of one source paper.

YOLO-World remains the already tested D2 detector-replacement variant. It is
not added as an uncalibrated parallel vote.

## 2. Literature traceability

| Component | Part I relationship | Stage L role |
|---|---|---|
| D1 YOLOv8n | YOLO-based detection and fine-tuning category | Primary vehicle detector |
| B1 mapping | APSD-OC detector-to-slot geometry concept | Converts vehicle boxes to slot evidence |
| E1b | Improved MobileNetV3 plus paper-inspired CBAM | Reviews only detector-negative slots |
| F2 | Local integration of D1, B1 and E1b | Asymmetric miss-recovery gate |
| E4 | Temporal-stability design motivated by tracking literature | Smooths short score changes |
| E5 ByteTrack | Supporting identity/motion module; not TrackTrack reproduction | Suppresses moving tracks and confirms stationary tracks |
| D2 YOLO-World | Direct use of the CVPR 2024 model | Alternative detector already evaluated in P2 |

Automatic slot discovery from APSD-OC, TrackTrack/MOTIP reproduction and
specialized low-light/RAW networks remain outside the implemented scope.

## 3. Static results

The development run reused the 27 cached Stage J D1 detections and did not
rerun a detector. Those exact 27 PKLot images also supplied E1b's historical
camera-split train/development/test patches, so this run is a functional
integration check, not independent generalization evidence.

The more informative retrospective run used the 90 Stage K images. These
images had zero hash overlap with Stage J and were not used to train E1b.
However, P0/P1/P2 results had already been viewed, so the P3 extension is not
an untouched test and was not used for tuning.

| Method | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied |
|---|---:|---:|---:|---:|---:|
| P1: D1 + B1 | 0.808398 | 0.598044 | 0.984148 | 0.401956 | 0.015852 |
| E1b classifier only | **0.992226** | 0.983531 | **0.998382** | 0.016469 | **0.001618** |
| P3 static gate | 0.987061 | **0.995368** | 0.982853 | **0.004632** | 0.017147 |

Relative to P1, P3 recovered 772 occupied slots and introduced four new
false-occupied decisions among detector-negative slots. It improved 78 of 90
images, tied on 12 and lost on none. The paired mean per-image Macro F1 gain
was 0.191884 with a 95% image-bootstrap interval of
0.158389 to 0.227038.

The camera-macro F1 increased from 0.824835 for P1 to 0.978063 for P3, and
every camera improved:

| Camera | P1 Macro F1 | P3 Macro F1 |
|---|---:|---:|
| PUCPR | 0.735789 | 0.994114 |
| UFPR04 | 0.871630 | 0.954711 |
| UFPR05 | 0.867086 | 0.985363 |

The E1b-only ablation prevents overclaiming. E1b had higher balanced Macro F1
than P3. The paired P3-minus-E1b mean was -0.014315 with a 95% interval of
-0.022769 to -0.006230. P3 instead offers an operational trade-off: it
reduces false-free errors, which are risky when a system tells a driver that
an occupied space is available, but accepts more false-occupied errors.

P3 classified 3,829 of 5,040 total Stage K slots; the detector confirmed the
other 1,211. Its mean incremental classifier time was 118.47 ms per image.
Classifying the additional detector-positive slots for E1b-only required
another 66.90 ms per image. The static timing excludes the already cached D1
detector run, whose Stage K mean detector latency was 37.28 ms.

## 4. Continuous-video result

The complete P3 workflow ran on all 1,974 frames of the consumed VIRAT 0502
case, with one manually reviewed slot and one departure at absolute frame
1660. All variants missed the departure:

| Method | Macro F1 | Occupied recall | Vacant recall | Departure |
|---|---:|---:|---:|---|
| P1 raw | 0.456072 | 1.000000 | 0.000000 | Missed |
| P3 gate | 0.456072 | 1.000000 | 0.000000 | Missed |
| P3 + temporal | 0.456072 | 1.000000 | 0.000000 | Missed |
| P3 + tracking + temporal | 0.456072 | 1.000000 | 0.000000 | Missed |

D1+B1 remained detector-positive in all 314 post-departure frames. Therefore
the detector-negative E1b branch was never called. ByteTrack produced 1,833
stationary-track-confirmed frames, 44 moving-track-suppressed frames and 97
plain detector-confirmed frames. Temporal filtering retained occupied through
the short suppressions.

Visual review at frame 1700 showed the cause: the oblique, broad slot polygon
overlaps an adjacent stationary vehicle after the labelled vehicle departs.
B1 assigns that neighbouring detection to the slot, and ByteTrack then makes
the wrong association temporally stable. Editing the polygon or thresholds
after seeing the result was prohibited and was not performed.

The steady-state full video path averaged 28.27 ms per frame, or 35.37 FPS,
on the local RTX 3060. That case contains one slot and never invokes E1b, so
it must not be used as a full-lot classifier-throughput claim.

## 5. Interpretation

Stage L demonstrates that the Part I components can be connected into a
complete, modular and runnable workflow. It also shows that more modules do
not guarantee better generalization:

- P3 strongly improves the D1+B1 static detector pipeline on the same parking
  domain and is particularly effective at reducing false-free errors.
- E1b alone remains the best balanced static method in the retrospective
  ablation.
- The continuous failure originates upstream in oblique-view slot geometry.
  A classifier fallback cannot act while the detector branch is confidently
  but incorrectly positive, and tracking can stabilize the wrong assignment.

For the coursework, P3 should be presented as the integrated proposed system
with a transparent trade-off and a real negative video case, not as a
universally superior model. A future geometry experiment should compare
perspective-normalized/core-region assignment on a newly selected development
video and then freeze it before any new test scene. VIRAT 0502 must not be
reused to tune that change.

## 6. Evidence

- Configuration: `configs/stage_l_integrated_workflow_frozen_20260728.yaml`
- Static development: `outputs/stage_l_p3_static_development_20260728_v1/`
- Static retrospective: `outputs/stage_l_p3_static_retrospective_20260728_v1/`
- E1b-only ablation:
  `outputs/stage_l_e1b_classifier_ablation_retrospective_20260728_v1/`
- Continuous case:
  `outputs/stage_l_p3_video_virat0502_20260728_v1/`
- Absolute-frame video analysis:
  `outputs/stage_l_p3_video_virat0502_20260728_v1/metrics_v2_absolute_frames.json`
