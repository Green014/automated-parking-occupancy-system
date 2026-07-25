# Literature-Core Experiment Plan

## Frozen split rule

The camera-level PKLot pilot split is fixed before model training:

| Split | Camera | Images | Slot instances | Use |
|---|---|---:|---:|---|
| Train | PUCPR | 9 | 900 | Fit MobileNetV3 head |
| Development | UFPR04 | 9 | 247 known (252 raw) | Select thresholds/fusion weights |
| Pilot test | UFPR05 | 9 | 358 known (360 raw) | One final evaluation after freezing |

Unknown slot labels are excluded. No PKLot frame is used for temporal metrics.
The pilot test is not described as globally untouched because the same images
were part of an earlier baseline development experiment.

## Post-hoc three-camera rotation

After the original Fold A result was observed, two additional camera rotations
were frozen as a robustness study:

| Fold | Train | Development selection | Test |
|---|---|---|---|
| A | PUCPR | UFPR04 | UFPR05 |
| B | UFPR04 | UFPR05 | PUCPR |
| C | UFPR05 | PUCPR | UFPR04 |

Each fold trains its own four-epoch classifier with the same seed and training
settings. E1/E2 thresholds and E3 weights/threshold are selected only on that
fold's development camera. The study is post-hoc and does not replace the
original Fold A interpretation or produce one universally selected
configuration.

## Experiment matrix

| ID | Branches | Mapping | Temporal | Primary purpose |
|---|---|---|---|---|
| E0 | Existing pretrained YOLOv8 | Polygon coverage | None | Closed-set detector baseline on the same pilot split |
| E1 | Adapted MobileNetV3 | Direct slot patch | None | Slot classifier |
| E2 | YOLO-World | Polygon coverage | None | Open-vocabulary detector branch |
| E3 | MobileNetV3 + YOLO-World | Weighted evidence fusion | None | Test complementarity |
| E4 | E3 | Weighted fusion | Hysteresis | Restricted positive-only Grand Bassin run completed; a three-sequence audit found no valid local mixed/transition truth |
| E5 | E3 + track evidence | Weighted fusion | Optional | Only after a tracker affects occupancy and suitable truth exists |

Detector-level domain check (separate from E0-E5 slot metrics):

| ID | Model | Data | Evaluation |
|---|---|---|---|
| D0 | YOLOv8 vehicle classes | NDISPark validation | 30 night images / 725 manual boxes, one vehicle class |
| D1 | YOLO-World vehicle prompts | NDISPark validation | Same images, truth, image size, and single-class protocol |

## Development selection

1. Select E1 and E2 binary thresholds only on UFPR04.
2. Sweep `w_cls` from 0 to 1 and set `w_det = 1 - w_cls`.
3. For every weight, sweep the fusion threshold on UFPR04.
4. Select highest macro F1; break ties by lower false-free rate and then the
   simpler 0.5 threshold.
5. Freeze the selected settings and evaluate UFPR05 once.
6. Save all development sensitivities and branch probabilities.

## Metrics

- Slot classifier: occupied precision/recall/F1, vacant recall, balanced
  accuracy, macro F1, confusion matrix.
- Detector: precision, recall, mAP@0.5, mAP@0.5:0.95 only on manual box truth
  such as NDISPark.
- Final slot state: macro F1, occupied recall, vacant recall,
  false-occupied rate, false-free rate.
- Continuous sequence: unsupported flicker per slot-minute, transition
  instability, entry/exit transition latency, stable-state time, FPS.
- Tracking: IDF1/HOTA only with persistent human identity truth.

## Negative-result rules

- E3 is not declared better unless the frozen pilot-test result improves the
  selected primary metric.
- PKLot cannot support E4/E5 claims.
- Grand Bassin's current positive-only labels cannot support vacant recall or
  false-occupied rate.
- Machine preannotations are never used as detector ground truth.
- Candidate motion or detector dropouts are never promoted to transition
  truth without one complete marked bay and a human-visible arrival/departure.
- Failure to download a checkpoint is an environment limitation, not an
  algorithm result.
