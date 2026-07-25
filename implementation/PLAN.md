# Part II Implementation Plan

Project: **Automated Parking Lot Occupancy and Tracking System**

Prepared: 24 July 2026

## 1. Coursework alignment

Part II requires a working computer-vision system implemented with Python and
OpenCV, either by improving an existing method or by combining known
algorithms. The system must be evaluated with task-appropriate metrics and the
report must explain both the differences from existing systems and the reasons
for any negative result.

This project satisfies the combination requirement through the following
pipeline:

```text
video
  -> OpenCV decoding and frame handling
  -> pretrained YOLOv8 vehicle detection
  -> optional ByteTrack identity association
  -> predefined parking-slot polygon mapping
  -> confidence-aware temporal hysteresis
  -> occupied/vacant states, overlays, logs, and evaluation artifacts
```

YOLO is only the vehicle detector. A YOLO vehicle box is **not** a vacant-space
prediction and is **not** yet a slot-occupancy prediction. The explicit
detection-to-slot mapping stage converts vehicle detections into slot states.
A vehicle in a driving lane must remain unassigned, while a vehicle whose
geometry matches a parking-slot polygon may cause that slot to be marked
occupied.

## 2. Problem statement

A fixed parking-lot camera produces frames containing parked vehicles, moving
vehicles, shadows, partial occlusions, and perspective distortion. A generic
vehicle detector can locate vehicles but cannot determine which predefined
parking spaces are occupied. Frame-independent decisions also flicker when a
vehicle is briefly missed or passes near a slot. The problem is therefore to
convert noisy vehicle detections from continuous video into stable,
slot-specific occupied/vacant states while preserving real-time performance
and recording explainable evidence for evaluation.

## 3. Objectives

1. Read parking-lot video with OpenCV and run pretrained YOLOv8n/s without
   custom training for the initial baseline.
2. Provide an OpenCV polygon annotation tool and a versioned slot-map format.
3. Implement and compare centre-point and polygon-overlap detection-to-slot
   mapping.
4. Integrate ByteTrack for persistent vehicle IDs without training a new
   tracker.
5. Add confidence-aware temporal hysteresis to reduce slot-state flicker.
6. Produce occupied/vacant overlays, per-frame state logs, transition events,
   timing summaries, and rendered output video.
7. Evaluate classification, detection, tracking, temporal stability, and
   runtime with metrics appropriate to each task.
8. Use video-level splits and never randomly mix adjacent frames across
   training, validation, and test sets.

## 4. Scope and non-goals

The first implementation is a single fixed-camera system with manually defined
slot polygons. YOLO-World, MOTIP, TrackTrack, DiffMOT, AODRaw, multi-camera
fusion, automatic slot discovery, and edge deployment are literature-supported
extensions only. They are not dependencies of the first evaluated system.

The first implementation establishes a no-training detector baseline before
any fine-tuning. The development error analysis has now passed the fine-tuning
gate, so a video-level-split annotation package is prepared. Model training
remains blocked until its machine preannotations have been manually corrected.

## 5. Experimental conditions

| ID | Detector | Mapping | Tracking | Temporal filter |
|---|---|---|---|---|
| B0 | Pretrained YOLOv8n/s | Bounding-box centre inside slot polygon | No | No |
| B1 | Same detector and thresholds as B0 | Box/slot polygon-overlap score | No | No |
| Proposed | Same detector and thresholds as B0/B1 | Polygon overlap | ByteTrack | Confidence-aware hysteresis |

The detector model, image size, confidence threshold, class list, source
videos, and slot polygons will be held constant across B0, B1, and Proposed.
Only the components named in the table may change.

## 6. Data plan

The data audit is recorded in `data/DATASET_MANIFEST.md`.

### 6.1 Primary occupancy evaluation

- Use PKLot and/or CNRPark-EXT because both have explicit open licenses,
  fixed views, visible parking geometry, and slot-level annotations.
- Treat PKLot's five-minute captures as an image sequence, not as
  frame-contiguous tracking data.
- Group all frames from the same camera/date sequence in one split. No
  adjacent or near-duplicate frames may cross split boundaries.
- Use a small, manually verified video set for temporal evaluation. Each clip
  stays wholly in exactly one of development, validation, or test.

### 6.2 Detection and robustness

- Use NDISPark for daytime/night-time vehicle detection and qualitative failure
  analysis because it provides vehicle boxes/masks and an explicit open-data
  license.
- Detection metrics are only computed on frames with vehicle-box ground truth.

### 6.3 Tracking

- Use a licensed continuous-video source for engineering tests.
- Report IDF1/HOTA only if the selected test clip has persistent track-ID
  ground truth. Otherwise report slot flicker, transition latency, track
  fragmentation diagnostics, and qualitative trajectories without presenting
  them as IDF1/HOTA.

## 7. Evaluation protocol

### 7.1 Slot-level binary classification

Primary metric: slot-level occupied-class F1.

Also report the 2x2 confusion matrix, accuracy, precision, recall,
false-free rate (`FN / actual occupied`), and false-occupied rate
(`FP / actual vacant`). Results must include per-video scores and a
micro-averaged aggregate so that one long clip cannot silently hide a failed
scene.

### 7.2 Detection

On box-annotated data, report precision-recall curves, AP, mAP@0.5,
mAP@0.5:0.95, and representative false positives/false negatives. These
metrics describe the vehicle detector and must not be relabelled as
slot-occupancy metrics.

### 7.3 Temporal behaviour

- Flicker count: a predicted state change not supported by a ground-truth
  state change within the tolerance window.
- Flicker rate: unsupported changes per slot-minute.
- Transition latency: elapsed time from a ground-truth transition to the first
  stable matching predicted state.
- Report both entry and exit latency distributions (median, p90, and maximum).

### 7.4 Tracking

IDF1 and HOTA are conditional on track-ID ground truth. If that ground truth is
not available, the report will explicitly state that the metrics were not
computed.

### 7.5 Runtime

Report end-to-end FPS, mean/p50/p95 frame latency, detector latency, mapping
latency, and output/rendering latency. Record the model, input resolution,
device, GPU, OpenCV, PyTorch, and Ultralytics versions. Warm-up frames are
reported separately from steady-state timing.

## 8. Fine-tuning gate

Fine-tuning is allowed only after a frozen pretrained baseline has been
evaluated on the validation videos.

Proceed to custom annotation and YOLOv8n/s fine-tuning if both conditions hold:

1. At least 10% of validation occupied-slot errors are confirmed by manual
   review to originate from missed or severely mislocalized vehicles rather
   than bad polygons/mapping; and
2. Changing the confidence threshold or using YOLOv8s does not remove the
   failure without an unacceptable false-positive or runtime cost.

All custom labels will be split by video before training. The test videos
remain untouched until the final frozen comparison.

Current gate result: passed. Detector miss/severe localization accounted for
120 of 137 initial PKLot false-free errors (87.6%); YOLOv8s was slower and did
not recover YOLOv8n recall. See
`metadata/FINETUNING_GATE_ASSESSMENT.md`.

## 9. Implementation phases

### Phase A - reproducible B0 baseline

- Project environment, dependency lock snapshot, hardware report.
- Versioned polygon schema and loader.
- OpenCV polygon annotation utility.
- YOLOv8 vehicle detector adapter.
- Centre-point assignment, annotated video, CSV/JSON logs.
- Geometry/configuration tests and one licensed-data smoke run.

Exit criterion: B0 processes a video from start to finish and emits a readable
output video, slot-state log, and timing summary.

### Phase B - B1 geometry

- Convex polygon-overlap computation with OpenCV.
- One-to-one best-evidence assignment and configurable overlap threshold.
- B0/B1 comparison on exactly the same detections.

Exit criterion: unit tests cover partial overlap, boundary points, moving-lane
vehicles, and competing slots.

### Phase C - Proposed temporal system

- ByteTrack integration through the detector/tracker adapter.
- Confidence-aware exponential evidence and separate occupied/vacant
  hysteresis thresholds.
- Transition event log and track-to-slot diagnostics.

Exit criterion: deterministic synthetic tests show reduced one-frame flicker
and bounded entry/exit delay.

### Phase D - automatic evaluation

- Slot-level evaluator, confusion matrix, PR curve, false-free/false-occupied
  rates, flicker rate, transition latency, and per-video aggregation.
- Detection evaluator on box-annotated data.
- FPS/latency benchmark and saved run metadata.
- Failure-case exporter that copies frame references and prediction evidence.

Exit criterion: one command recreates tables and plots for B0, B1, and
Proposed from frozen prediction logs.

### Phase E - report and viva evidence

- System diagram, algorithm descriptions, parameter table, and ablation table.
- Explain why YOLO detection requires ROI mapping.
- Discuss failures even if Proposed does not improve every metric.
- Package clean Python code, configuration, tests, and exact run commands.

## 10. Schedule to submission

| Date | Deliverable |
|---|---|
| 24 July | Audit, plan, dataset manifest, B0 scaffold and smoke test |
| 25-26 July | B1 overlap mapping, annotation pass, unit tests |
| 27-28 July | ByteTrack and temporal filter; freeze experiment configs |
| 29-30 July | Ground-truth pass and automated evaluation |
| 31 July | Failure analysis and final reruns |
| 1 August | IEEE report figures/tables and reproducibility check |
| 2 August | Project report submission |
| 3 August | Code submission package and viva run-through |

## 11. Reproducibility rules

- Every run saves the source video ID, source checksum where practical, slot-map
  checksum, configuration, package versions, model weights name, device, and
  random seed.
- Raw data are not committed unless their license and size permit it.
- Derived videos and model weights stay under ignored output/cache paths.
- `literature_review/` is read-only for Part II work.
- A test failure blocks experiment reporting until resolved or documented.

## 12. Execution status

| Work item | Status | Evidence |
|---|---|---|
| Hardware/environment audit | Complete | `metadata/HARDWARE_REPORT.md` |
| Licensed-data audit | Complete for current sources | `data/DATASET_MANIFEST.md` |
| B0/B1 implementation | Complete | geometry tests and PKLot report |
| ByteTrack + temporal implementation | Complete | Grand Bassin report |
| Automated slot evaluation | Complete | JSON, confusion, PR and error outputs |
| NDISPark detector evaluation | Complete | 30 night validation images, 725 manual boxes |
| PKLot development selection | Complete | 27 images, 1,505 known slot labels |
| Continuous stability ablation | Complete | 793 frames, seven checked bus bays |
| Transition-latency test | Pending suitable verified transitions | Current real clip has none |
| IDF1/HOTA | Not applicable to current data | No persistent track-ID truth |
| Fine-tuning labels | Prepared, manual correction pending | `data/finetune/` |
| Final untouched test | Pending | Must follow annotation/freeze step |
