# Grand Bassin Continuous-Video Stability Report

Run date: 24 July 2026

## Audited source and truth scope

- source sequence:
  `stream_gbassinexch1.stream_2023-02-16_16-34-07.264`;
- 793 ordered 1280x720 frames at 2 FPS, reconstructed with OpenCV;
- duration: 396.5 seconds;
- every manifest frame decoded successfully;
- license: CC BY-NC-SA 4.0;
- seven manually checked, continuously occupied bus-bay polygons;
- 5,551 positive slot-frame labels.

The subset contains no manually verified vacant slots and no verified
parking-space arrival/departure. It is therefore a stability/false-free
ablation, not a complete binary-classification test. Precision, false-occupied
rate, and AP are not informative because the truth has no negatives.

## Frozen comparison

All methods use pretrained YOLOv8n, confidence 0.025, image size 1280, the
same video and the same seven polygons. B1 and Proposed use overlap 0.40.
Temporal metrics exclude six initialization frames; classification includes
all frames.

| Method | Recall | Slot F1 | False-free | Unsupported flickers | Flicker / slot-minute | Steady FPS |
|---|---:|---:|---:|---:|---:|---:|
| B0: centre | 0.586 | 0.739 | 0.414 | 395 | 8.60 | 30.37 |
| B1: overlap | 0.625 | 0.769 | 0.375 | 422 | 9.19 | 32.71 |
| Proposed | **0.824** | **0.903** | **0.176** | **59** | **1.29** | 21.06 |

Relative to B1, Proposed reduced unsupported flicker by 86.0% and reduced the
false-free rate by 53.0%, while steady-state throughput fell by 35.6%. The
21.06 FPS result still exceeds the 2 FPS source rate and includes OpenCV
visualization plus MP4 writing.

## Implementation correction found by the experiment

The first tracker integration used the boxes returned by
`Ultralytics.model.track()`. Unconfirmed detections were absent from those
outputs, which unintentionally made ByteTrack a second detector gate and
dropped Proposed recall to 0.098. The corrected adapter:

1. runs YOLO once;
2. passes those detections to ByteTrack;
3. attaches IDs to matched detections; and
4. retains unmatched YOLO detections for slot mapping.

This preserves the intended relationship: tracking adds temporal identity but
does not redefine whether YOLO detected a vehicle.

The low-confidence detector setting also required a validation-scale
hysteresis threshold. A cached sweep selected occupied 0.020 and vacant 0.005
with rise alpha 0.60 and fall alpha 0.15. This setting is provisional because
the stability subset has no vacant slots; it must be checked on a mixed-class
validation video before any final claim.

## Metric exclusions

- Transition latency: not observed; median/p90/maximum are `null`.
- IDF1/HOTA: not reported because the selected source has no persistent
  human track IDs.
- Detector mAP: not reported on Grand Bassin because its COCO boxes are
  machine-generated preannotations rather than manual gold labels.

These exclusions are deliberate task/ground-truth alignment, not missing
software functionality.
