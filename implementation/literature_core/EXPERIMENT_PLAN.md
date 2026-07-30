# Literature-Core Experiment Plan

## Historical split and corrected data role

The camera-level PKLot pilot split is fixed before model training:

| Split | Camera | Images | Slot instances | Use |
|---|---|---:|---:|---|
| Train | PUCPR | 9 | 900 | Fit MobileNetV3 head |
| Development | UFPR04 | 9 | 247 known (252 raw) | Select thresholds/fusion weights |
| Fold A evaluation partition | UFPR05 | 9 | 358 known (360 raw) | Internal camera-transfer diagnostic |

Unknown slot labels are excluded. No PKLot frame is used for temporal metrics.
All 27 images are method-development data because every camera has been used
by the baseline or the post-hoc rotation. Historical `test` field names remain
in saved artifacts for compatibility; they do not denote an external final
test.

## Post-hoc three-camera internal development rotation

After the original Fold A result was observed, two additional camera rotations
were frozen as a robustness study:

| Fold | Train | Development selection | Test |
|---|---|---|---|
| A | PUCPR | UFPR04 | UFPR05 |
| B | UFPR04 | UFPR05 | PUCPR |
| C | UFPR05 | PUCPR | UFPR04 |

Each fold trains its own four-epoch classifier with the same seed and training
settings. E1/E2 thresholds and E3 weights/threshold are selected only on that
fold's development camera. The study is post-hoc internal method development;
it neither produces an external holdout result nor one universally selected
configuration.

## Experiment matrix

| ID | Branches | Mapping | Temporal | Primary purpose |
|---|---|---|---|---|
| E0 | Existing pretrained YOLOv8 | Polygon coverage | None | Closed-set detector baseline on the same pilot split |
| E1a | Adapted standard MobileNetV3 | Direct slot patch | None | Standard slot-classifier baseline |
| E1b | Paper-inspired MobileNetV3 | Direct slot patch | None | CBAM/LeakyReLU6 component ablation; never called exact reproduction |
| E2 | YOLO-World | Polygon coverage | None | Open-vocabulary detector branch |
| E3a | E1a + YOLO-World | Raw weighted evidence fusion | None | Historical fusion baseline |
| E3b | E1a + YOLO-World | Separately calibrated, non-negative logistic fusion | None | Proposed interpretable unified fusion |
| E4 | Frozen E3b | Calibrated fusion | EMA/hysteresis | Executed as a two-sequence departure case study; negative external result |
| E5 | YOLOv8n + ByteTrack | Track-to-slot gate | Stationary/moving and dwell rules | Executed as a two-sequence departure case study; did not beat T0 raw comparator |

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

## Calibrated fusion selection and freeze

1. Treat one historical held-camera prediction set per PKLot camera as
   camera-grouped out-of-fold development data.
2. Fit separate monotonic Platt-style mappings to the classifier score and to
   detector `confidence x coverage` evidence.
3. Fuse calibrated log-odds with a logistic model constrained to
   non-negative branch coefficients.
4. In each development diagnostic, fit calibration, fusion, and threshold on
   two complete cameras and evaluate the third; never split slots randomly.
5. Fit the deployable `configs/proposed_fusion.yaml` once on all camera-OOF
   development predictions.
6. Do not change this configuration after any CNR-EXT prediction is viewed.

## External holdout protocol

CNRPark+EXT was selected from its official project page. The CNR-EXT subset
has an explicit ODbL-1.0 license, nine cameras, 4,081 full frames and 144,965
slot labels. Official slot geometry in the full-frame archive is
axis-aligned bounding boxes rather than precise polygons; this limitation
must be stated in every comparison.

The frozen protocol is stored in `configs/external_holdout_frozen.yaml`.
It uses every available CNR-EXT full frame once, retains complete
camera/datetime groups, and computes confidence intervals by resampling
complete image groups rather than individual slots. CNR-EXT is not used for
training, calibration, model choice, fusion weights, or thresholds. If an
official archive cannot be acquired or validated, no final metric is
substituted; `DATASET_ACCESS_BLOCKER.md` must record the exact blocker.

## Phase A/B continuation boundary

The completed CNR-EXT run is now a consumed once-only external evaluation.
No new threshold, weight, calibration, temporal parameter, or method choice
may use CNR-EXT.

The immutable-artifact manifest
`data/manifests/frozen_artifacts_20260725.yaml` hashes the method configs,
three PKLot split files, E0/E1a/E1b/E2 models, training summaries, Fold A
selected parameters, CNR inputs, and external metrics. Re-verification on 26
July passed all 17 artifact hashes and both result-count checks.

The next temporal source is governed by
`configs/temporal_protocol_pending.yaml`:

1. The user accepted the VIRAT Usage Agreement on 26 July 2026; no personal
   details are stored in Git.
2. The initial bounded subset contains 21 videos / 961,643,821 bytes. A
   targeted `0503` extension added five videos, bringing the total to 26
   videos / 1,605,720,653 video bytes plus 15,925,504 annotation bytes. Both
   manifests record item IDs, sizes, and SHA-256 values.
3. Confirm complete fixed bays, occupied/vacant states, and at least one
   human-visible arrival/departure in every selected sequence.
4. Prefer development and holdout from distinct camera/scene groups.
5. If one long video must be split, record half-open frame intervals and a
   non-zero guard; no adjacent or random frames may cross the boundary.
6. Lock the holdout ID, interval, and source SHA-256 before using development
   truth for any parameter.
7. Only a protocol that validates source/truth paths, file hashes, decoded
   video bounds, polygons, complete intervals, mixed counts, and transitions
   as `ready_for_experiment: true` authorizes mixed-class E4/E5. Fusion V2 has
   the additional prerequisite that E5 first behave reliably on development.

Current gate outcome: `0502` is frozen as development (last occupied 1659,
first vacant 1660), and the subsequently found `0503` clip is the distinct-
scene once-only holdout (last occupied 1549, first vacant 1550). The holdout
identity, hash, polygon, truth, and `temporal_e4_e5_frozen.yaml` were locked
before any model output was inspected. Official `XXYY` scene grouping remains
mandatory; six-digit `XXYYZZ` sequence prefixes are never independent folds.

E4 and E5 were then executed without retuning. Neither exceeded T0 on the
holdout, and E5 had zero vacant recall on development. Therefore the
detector-primary/classifier-fallback Fusion V2 gate remains closed. No
Grand Bassin, CNR-EXT, repeated still image, or unlicensed video was substituted
for this result.

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

- E3 is not declared better unless a frozen external result improves the
  selected primary metric; internal camera rotations remain development.
- PKLot cannot support E4/E5 claims.
- Grand Bassin's current positive-only labels cannot support vacant recall or
  false-occupied rate.
- Machine preannotations are never used as detector ground truth.
- Candidate motion or detector dropouts are never promoted to transition
  truth without one complete marked bay and a human-visible arrival/departure.
- Failure to download a checkpoint is an environment limitation, not an
  algorithm result.

## D1 formal-training resource gate

`GPU-GATE-NDISPARK-D1-20260727-01` passed from the executed Stage F evidence
without training or prediction. The selected formal experiment is
`D1-NDISPARK-FT-20260727-01`: fresh COCO-pretrained YOLOv8n, NDISPark official
train plus consumed development validation, image size 640, physical batch 4,
nominal batch 64, 16 post-warm-up accumulation steps, 50 maximum epochs,
patience 10, and seed 20260727.

The local 6 GiB RTX 3060 is selected. Paid/remote GPU use, A100 use, batch 8,
larger image sizes, additional seeds, and hyperparameter search are outside
this frozen run. NDISPark count-only test, CNR-EXT, PKLot, and VIRAT remain
unavailable to training and epoch selection.

Stage H outcome: the one frozen seed completed 47 epochs and stopped under
patience 10. Epoch 37 is the selected D1 checkpoint. The initial runner's
post-run resource callback audit failed after training completed and is
retained; existing artifacts were recovered without retraining.

Stage I outcome: D1 was selected on consumed development validation before
test prediction, using mAP@0.5:0.95 and recall. One confidence of 0.10 was
selected from the declared development grid by aggregate D0/D1/D2 count MAE
and frozen for all three models. On the count-only test D2 had the lowest MAE;
this negative D1 result is retained without post-test threshold or detector
changes. The test has no box truth, so no test detector mAP or FP/FN box claim
is permitted.

Stage I-v2 retained v1, corrected class merging with class-agnostic NMS,
calibrated each detector on the same development grid, and retained
`max_det=300` after a 300/1000 development sensitivity check.

Stage J then froze `P-COMP-PKLOT-DEV-STAGEJ-20260727-01` before prediction and
connected D0/D1/D2 to the same B1 geometry and logging contract. P0/P1/P2
used the Stage I-v2 development thresholds without slot-truth selection.
Their consumed-development Macro F1 values were 0.768040, 0.825723 and
0.735168. All negative errors and the cold-start runtime confound are
retained.

The original Stage K local-inventory blocker is retained. An additive v2 gate
later admitted 90 complete PKLot JPG/XML pairs from three previously unused
camera/date groups, with zero image-hash overlap against Stage J. The frozen
P0/P1/P2 test was executed once on 5,034 known slot labels. Its output may not
be used for threshold changes, detector reselection or a new fusion
development cycle.
