# Literature-Core Results

This file is updated only with executed commands. No placeholder value is
presented as an experiment result.

Run date: 25-26 July 2026 (Asia/Shanghai)

| Check | Result |
|---|---|
| Existing baseline unit tests | **28/28 passed** after baseline closure |
| Literature-core unit tests | **82/82 passed** after metric correction |
| Source compilation | `python -m compileall` passed |
| Python/OpenCV available | Yes |
| CUDA / RTX 3060 available | Yes, 6 GiB |
| torchvision MobileNetV3 API | Available |
| Ultralytics YOLO-World API | Available |
| Local MobileNetV3 pretrained checkpoint | Official weight acquired and hash verified |
| Local YOLO-World/CLIP checkpoints | Official weights acquired and hashes verified |
| Existing baseline outputs overwritten | No |
| Continuous mixed/transition truth found locally | Yes; one departure sequence each in distinct VIRAT scenes `0502` and `0503` |
| Frozen artifact audit | **17/17 static and 11/11 temporal SHA-256 checks passed**; declared result sizes, 4,081 frames, and 144,965 slot records matched |
| VIRAT bounded screening | 26 official videos / 1,605,720,653 video bytes; **0502 development and 0503 holdout truth verified** |
| Temporal protocol gate | Frozen, artifact-verified, and `ready_for_experiment: true`; Fusion V2 remains closed |

## Baseline execution check

The original B0 entry point was run on the existing two-frame
`pklot_smoke.mp4`, with its output redirected into this independent
literature-core output tree. It processed 100 slots over two frames. The first
frame included model initialization, so end-to-end throughput was only
0.349 FPS; the one measured post-warm-up frame took 17.656 ms (56.638 FPS),
including 15.123 ms detection and 2.083 ms mapping. One post-warm-up frame is
not a benchmark and is reported only as an environment check.

Evidence:
`outputs/phase0_baseline_smoke/summary.json`.

## Evaluation correction and baseline closure

The temporal evaluator previously searched only at or after each
ground-truth transition. If a prediction changed early and stayed in the
target state, the search rediscovered that already-active state at the truth
frame and reported a misleading zero latency.

The corrected evaluator matches the nearest observed prediction change into
the target state that satisfies the frozen `stable_frames` rule. Each truth
event now records prediction-minus-truth signed error in frames and seconds,
entry/exit direction, and one of `early`, `on_time`, `delayed`, or `missed`.
Early matches are excluded from the backward-compatible non-negative latency
summary. Unsupported flicker and extra transition-window changes remain
separate.

Unit tests cover on-time, early, delayed, completely missed, briefly correct
but unstable, and multiple-jump predictions. The full `literature_core`
suite passed 82/82.

Baseline reporting is now closed by
`../configs/baseline_methods.yaml` and `../BASELINE_CLOSURE.md`:

- B0 is YOLOv8 + bounding-box-centre mapping;
- B1 is YOLOv8 + polygon coverage;
- E0 is the historical frozen static CNR-EXT YOLOv8 coverage result; and
- T0 is the raw YOLOv8 temporal comparator (historical artifact key
  `e0_raw`).

The common `parking-run --method B0|B1|T0` entry point writes
`annotated.mp4`, `occupancy.csv`, `events.csv`, and `summary.json`. A
deterministic fake-video integration test verified the complete artifact
contract, and the full baseline suite passed 28/28. No historical metric or
frozen output was recomputed or modified for this code/documentation closure.

## Adapted MobileNetV3 training

The standard torchvision MobileNetV3-Small backbone was initialized from
ImageNet, frozen, and trained with a two-class head for four pilot epochs:

- seed 20260725, batch 32, AMP, CUDA;
- 900 PUCPR training patches (449 vacant, 451 occupied);
- 247 UFPR04 development patches;
- 358 UFPR05 patches not loaded by the training loop;
- best epoch 2, development-selected threshold 0.61;
- development confusion matrix `[[121, 1], [1, 124]]`;
- development macro F1 0.991902;
- elapsed training time 15.907 s.

This is a feasibility/pilot training run, not the 500-epoch modified
MobileNetV3 reported in [2]. Evidence:
`outputs/mobilenet_pilot/training_summary.json` and
`outputs/mobilenet_pilot/training_log.jsonl`.

## YOLO-World validation

The official `yolov8s-worldv2.pt` adapter was successfully prompted with
`car`, `truck`, `bus`, and `motorcycle` on one local PKLot image. At confidence
0.025 and image size 1280 it returned 154 raw detections in 7.794 s including
model and text-encoder loading. Visual inspection confirmed broad vehicle
coverage but also low-confidence duplicate/confused labels (for example a
single vehicle receiving car/bus/truck candidates). This justifies retaining
raw detections and tuning the slot-level threshold rather than claiming that
prompting alone solves occupancy.

Evidence: `outputs/yolo_world_smoke/detections.json` and
`outputs/yolo_world_smoke/annotated.jpg`.

## Frozen E0-E3 PKLot pilot ablation

All thresholds and weights were selected on UFPR04 only. Frozen settings were:

- E0: existing coverage rule;
- E1 threshold 0.61;
- E2 threshold 0.08;
- E3 `w_cls=0.5`, `w_det=0.5`, threshold 0.37.

UFPR05 was then evaluated once without participating in selection.

| Experiment | UFPR04 macro F1 | UFPR05 macro F1 | UFPR05 occupied recall | UFPR05 vacant recall | UFPR05 false-free rate |
|---|---:|---:|---:|---:|---:|
| E0 YOLOv8 + polygon | 0.987847 | 0.980446 | 0.977901 | 0.983051 | 0.022099 |
| E1 MobileNetV3 | 0.991902 | 0.977647 | 0.983425 | 0.971751 | 0.016575 |
| E2 YOLO-World + polygon | 0.991900 | **0.983240** | 0.966851 | **1.000000** | 0.033149 |
| E3 weighted fusion | **1.000000** | 0.980446 | 0.961326 | **1.000000** | 0.038674 |

UFPR05 confusion matrices (`[[TN, FP], [FN, TP]]`) were:

- E0 `[[174, 3], [4, 177]]`;
- E1 `[[172, 5], [3, 178]]`;
- E2 `[[177, 0], [6, 175]]`;
- E3 `[[177, 0], [7, 174]]`.

The principal result is mixed. E2 improved macro F1 over E0 by 0.002794 and
eliminated false-occupied cases in this internal Fold A evaluation partition,
but had more false-free errors. E3 perfectly fitted its selection partition
yet did not improve on E0 on the held-camera partition and increased
false-free errors. This is a probability-scale calibration and
threshold-transfer failure. The saved overlap analysis confirms complementary
cases in which MobileNetV3 and YOLO-World each corrected examples missed by
the other.

Diagnostic accumulated branch times for the same run were 2.420 s
(classifier), 7.747 s (YOLO-World), and 1.056 s (YOLOv8). These paths use
different batching/loading patterns and are not presented as a fair latency
benchmark.

Evidence:
`outputs/pklot_ablation/metrics.json`,
`outputs/pklot_ablation/selected_parameters.json`,
`outputs/pklot_ablation/development_sensitivity.csv`,
`outputs/pklot_ablation/branch_threshold_sensitivity.json`,
`outputs/pklot_ablation/branch_probabilities.csv`, and
`outputs/pklot_ablation/raw_detections.jsonl`.

## Frozen PKLot error attribution

The saved probabilities were analyzed after evaluation without changing a
model, threshold, or weight. UFPR05 contained 20 slots for which at least one
method was wrong; no slot was wrong for all four methods.

| Split | E0 errors | E1 errors | E2 errors | E3 errors | Any method wrong |
|---|---:|---:|---:|---:|---:|
| UFPR04 development | 3 | 2 | 2 | 0 | 6 |
| UFPR05 pilot holdout | 7 | 8 | 6 | 7 | 20 |

On UFPR05, E1 and E2 were both correct on 344/358 slots, E1 alone was correct
on 6, and E2 alone was correct on 8. Fusion produced a correct result while
at least one branch was wrong in 9 cases, but it also failed in 2 cases where
both branches were individually correct. The failure is therefore partly a
probability-scale/threshold interaction rather than complete absence of
information from both branches.

Evidence: `outputs/pklot_error_analysis/summary.json`,
`outputs/pklot_error_analysis/error_rows.csv`, and
`outputs/pklot_error_analysis/test_error_montage.jpg`.

## Post-hoc three-camera internal development rotation

Two additional camera rotations were executed after the original Fold A
result had been observed. Every fold kept its evaluation camera out of that
fold's training and parameter selection, but all 27 images have now
participated somewhere in method development. These are internal
camera-grouped development results, not an external final test. The table
reports camera-equal, not sample-weighted, macro F1.

| Fold | Train | Development | Test | E0 | E1 | E2 | E3 |
|---|---|---|---|---:|---:|---:|---:|
| A | PUCPR | UFPR04 | UFPR05 | 0.980446 | 0.977647 | **0.983240** | 0.980446 |
| B | UFPR04 | UFPR05 | PUCPR | 0.951070 | 0.964405 | 0.823569 | **0.992222** |
| C | UFPR05 | PUCPR | UFPR04 | 0.987847 | 0.971652 | 0.991900 | **0.995950** |
| Unweighted mean | - | - | three cameras | 0.973121 | 0.971235 | 0.932903 | **0.989539** |
| Population std | - | - | three cameras | 0.015882 | 0.005414 | 0.077391 | 0.006608 |

E3 beat E0 in two folds and tied it in one, with a mean macro-F1 delta of
+0.016418. Its mean occupied/vacant recalls were 0.986370/0.992813, compared
with E0's 0.966026/0.980214. Mean false-free and false-occupied rates fell
from 0.033974/0.019786 to 0.013630/0.007187.

The broader internal result is more positive for fusion than Fold A alone,
but it does not establish one deployable universal configuration.
Development-selected
E3 classifier weight varied from 0.30 to 0.50 and the threshold from 0.23 to
0.38. E2 was particularly camera-sensitive: it won against E0 in two folds
but fell to 0.823569 on PUCPR, yielding the largest standard deviation
(0.077391). Training-camera sample counts also differ substantially, so these
three folds are a robustness diagnostic rather than a large-sample estimate.

Evidence: `outputs/cross_camera/summary/summary.json`,
`outputs/cross_camera/summary/fold_results.csv`, and per-fold training and
ablation directories under `outputs/cross_camera/`.

## E3b camera-grouped calibrated fusion development

Historical held-camera predictions were reinterpreted as one out-of-fold
development prediction set per camera. For each leave-one-camera-out fold,
separate monotonic Platt-style calibrators were fitted to the MobileNet score
and the YOLO-World `confidence x slot coverage` evidence. A non-negative
logistic model then fused the calibrated log-odds. Calibration, fusion
coefficients, and the decision threshold used only the other two cameras;
there was no slot-level random split.

| Held camera | Unified E3a Macro F1 | E3b Macro F1 | E3b occupied recall | E3b vacant recall |
|---|---:|---:|---:|---:|
| PUCPR | **0.993333** | 0.986665 | 0.995565 | 0.977728 |
| UFPR04 | 0.991900 | **0.995950** | 1.000000 | 0.991803 |
| UFPR05 | **0.988827** | 0.969262 | 0.939227 | 1.000000 |
| Camera-equal mean | **0.991353** | 0.983959 | 0.978264 | 0.989844 |

E3b is therefore a negative Macro-F1 result on these development cameras:
calibration did not outperform the simpler E3a rule. It did, however, improve
the meaning of the branch scores before fusion:

| Held camera | Classifier Brier raw -> calibrated | Classifier ECE raw -> calibrated | Detector evidence Brier raw -> calibrated | Detector evidence ECE raw -> calibrated | E3b Brier / ECE |
|---|---:|---:|---:|---:|---:|
| PUCPR | 0.093158 -> 0.031308 | 0.229905 -> 0.091520 | 0.394347 -> 0.203548 | 0.440065 -> 0.236107 | 0.011576 / 0.036305 |
| UFPR04 | 0.053667 -> 0.011180 | 0.176405 -> 0.057656 | 0.201400 -> 0.062598 | 0.304752 -> 0.186042 | 0.002155 / 0.008622 |
| UFPR05 | 0.051772 -> 0.023627 | 0.152161 -> 0.055773 | 0.211569 -> 0.067754 | 0.308461 -> 0.206952 | 0.019571 / 0.025176 |

This distinction matters: `P_det` is not described as a native probability.
It is detector confidence multiplied by the fraction of a slot covered by
the assigned box. The tracked `configs/proposed_fusion.yaml` freezes the
all-development fit at classifier/detector calibration slopes
11.008170/11.642228, non-negative fusion coefficients 1.494789/1.049488,
intercept -0.156385, and occupied threshold 0.67. No external holdout result
was used in that choice.

Evidence:
`outputs/calibrated_fusion_development/metrics.json`,
`outputs/calibrated_fusion_development/predictions.csv`,
`outputs/calibrated_fusion_development/reliability_curves.csv`,
`outputs/calibrated_fusion_development/threshold_sensitivity.csv`,
`outputs/calibrated_fusion_development/weighted_fusion_sensitivity.csv`, and
`configs/proposed_fusion.yaml`.

## Standard versus paper-inspired MobileNetV3 development ablation

The source paper was checked again before implementation. E1a remains the
adapted torchvision MobileNetV3-Small. E1b is explicitly a paper-inspired
transfer-learning adaptation: it supplements each pretrained SE block with
an identity-initialized CBAM. Shallow LeakyReLU6 was also tested separately
and jointly. BSConv was not added because a verifiable conversion that
preserves pretrained weights was not established.

Thresholds in this table were selected on the internal UFPR04 development
camera and applied to the internal UFPR05 evaluation camera. The 358 timed
patches followed a 64-patch GPU warm-up on an RTX 3060 Laptop GPU.

| Variant | Parameters | Train time (s) | Internal Macro F1 | Occupied recall | Vacant recall | Model ms/patch |
|---|---:|---:|---:|---:|---:|---:|
| E1a standard | 1,519,906 | **18.901** | 0.977647 | 0.983425 | 0.971751 | 1.980 |
| LeakyReLU6 only | 1,519,906 | 27.404 | 0.966443 | **0.988950** | 0.943503 | **1.752** |
| E1b CBAM supplement | 1,976,245 | 29.268 | **0.986033** | 0.977901 | **0.994350** | 2.220 |
| CBAM + LeakyReLU6 | 1,976,245 | 32.631 | 0.972053 | 0.983425 | 0.960452 | 2.045 |

The component ablation does not support claiming that every paper component
helps this small transfer-learning regime. LeakyReLU6 reduced Macro F1, and
the combined variant was worse than CBAM alone. An earlier direct
SE-replacement prototype also produced a poor development result; making its
gate nearly identity caused FP16 overflow and NaN loss. Those output
directories were retained. The selected E1b instead preserves the pretrained
SE path and adds a zero-initialized CBAM whose initial mapping is exactly the
identity.

Evidence:
`outputs/mobilenet_variant_ablation/`,
`outputs/mobilenet_variant_evaluation/metrics.json`,
`outputs/mobilenet_variant_evaluation/probabilities.csv`, and
`outputs/mobilenet_variant_evaluation/threshold_sensitivity.csv`.

## Once-only CNR-EXT external holdout

The official CNRPark+EXT metadata and full-frame archive were downloaded from
the project GitHub release linked by the official site. The CNR-EXT subset is
licensed ODbL-1.0. Before any external prediction, all thresholds, model
checkpoints, prompts, fusion parameters, camera inclusion, and the bootstrap
unit were frozen in `configs/external_holdout_frozen.yaml`.

Integrity checks found 4,081 unique labeled frames, 144,965 slot labels, nine
official camera geometry files, zero missing slot geometries, and zero
missing/corrupt/wrong-size labeled frames. The official geometry consists of
axis-aligned slot boxes, not precise polygons. Boxes were scaled by the
published 2592x1944 to 1000x750 ratio. The 197 additional archive frames
without matching metadata labels were excluded rather than assigned inferred
labels.

| Method | Macro F1 (95% image-group bootstrap CI) | Occupied recall | Vacant recall | False-free | False-occupied |
|---|---:|---:|---:|---:|---:|
| E0 YOLOv8 + official boxes | **0.966766** (0.965044-0.968463) | **0.948134** | 0.989617 | **0.051866** | 0.010383 |
| E1a standard MobileNetV3 | 0.894361 (0.889323-0.899455) | 0.817333 | 0.987333 | 0.182667 | 0.012667 |
| E1b paper-inspired CBAM | 0.910801 (0.906046-0.915660) | 0.843191 | 0.992433 | 0.156809 | 0.007567 |
| E2 YOLO-World + official boxes | 0.963589 (0.961703-0.965410) | 0.946948 | 0.984090 | 0.053052 | 0.015910 |
| E3a raw weighted fusion | 0.921187 (0.917048-0.925342) | 0.861455 | 0.993362 | 0.138545 | 0.006638 |
| E3b calibrated logistic fusion | 0.909022 (0.904370-0.913586) | 0.837187 | **0.995737** | 0.162813 | **0.004263** |

The external result does not support E3b as the best overall method. E0 has
the highest Macro F1, and E2 is close but its confidence intervals remain
below E0's. E1b improves over E1a by 0.016441 Macro F1 and reduces both error
rates, supporting the CBAM supplement as the stronger classifier variant in
this external domain. Both fusion variants trade occupied recall for high
vacant recall. E3b has the lowest false-occupied rate, but its false-free
rate is 0.162813 and its Macro F1 is below E3a.

Calibration transfer is also mixed:

| Score | Brier | ECE |
|---|---:|---:|
| Classifier raw | **0.074689** | **0.044488** |
| Classifier after PKLot calibration | 0.099345 | 0.119239 |
| Detector evidence raw | 0.187201 | 0.297406 |
| Detector evidence after PKLot calibration | **0.059467** | **0.148274** |
| E3a raw weighted | 0.101619 | 0.183965 |
| E3b calibrated logistic | **0.078317** | **0.080631** |

Thus the detector calibrator transferred usefully, while the classifier
calibrator did not. E3b improved calibration over raw E3a but did not improve
its frozen classification threshold. This is direct external evidence of
probability-calibration and threshold-transfer limitations; no external
threshold sweep or follow-up parameter selection was performed.

After excluding eight warm-up frames, the complete six-method path timed
4,073 frames at 2.299 FPS (434.904 ms/frame) on the RTX 3060 Laptop GPU.
This includes patch extraction, both MobileNet variants, YOLOv8, YOLO-World,
both mappings, and both fusions.

Evidence:
`data/manifests/cnrpark_ext_external_holdout.yaml`,
`configs/external_holdout_frozen.yaml`,
`outputs/cnrpark_ext_frozen_evaluation_20260725/metrics.json`,
`outputs/cnrpark_ext_frozen_evaluation_20260725/predictions.csv`,
`outputs/cnrpark_ext_frozen_evaluation_20260725/reliability_curves.csv`, and
`outputs/cnrpark_ext_frozen_evaluation_20260725/frame_results.jsonl`.

## NDISPark night-domain detector comparison

The same 30 NDISPark validation images and 725 manual boxes were evaluated at
image size 1280. NDISPark's converted labels use COCO class 2 for every
vehicle, so YOLOv8 classes `car/motorcycle/bus/truck` and all four YOLO-World
prompts were collapsed to one vehicle class. This is a detector-level
validation comparison, not slot occupancy.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference ms/image |
|---|---:|---:|---:|---:|---:|
| YOLOv8n vehicle classes | 0.866350 | 0.721379 | 0.833638 | 0.434042 | **12.457** |
| YOLO-World prompts | **0.874733** | **0.780166** | **0.867063** | **0.476857** | 29.802 |

YOLO-World improved recall by 0.058786 and mAP@0.5:0.95 by 0.042815 under
this common night-domain protocol, while taking roughly 2.39 times the model
inference time. The pre-existing baseline report's YOLOv8 values differ
because that earlier run filtered predictions to COCO `car` only and did not
use this four-class-to-one comparison; its files remain unchanged.

Ultralytics names the collapsed class using dataset class 0 in its generated
batch plots, so the blue labels read `person` even though the evaluated boxes
are the collapsed vehicle predictions. This is a plotting-name artifact of
`single_cls=True`, not a claim that the model detected people; the numerical
protocol and JSON record this explicitly.

Evidence: `outputs/ndispark_detection_comparison/comparison.json`, plus the
two Ultralytics subdirectories containing PR curves, confusion matrices, and
label/prediction batches.

## End-to-end video smoke

The new pipeline processed two frames and 100 slots per frame, retaining raw
YOLO-World boxes, detection-to-slot mapping, `P_cls`, `P_det`, raw fused
`P_occ`, EMA-filtered `P_occ_filtered`, raw state, and final state. The run
took 9.241 s (0.216 FPS) including model/CLIP initialization. Because it has
only two repeated/static frames and no mixed-class transition truth, this is
an integration check, not temporal-accuracy or production-throughput
evidence.

Evidence: `outputs/pipeline_smoke/summary.json`,
`outputs/pipeline_smoke/occupancy.csv`, and
`outputs/pipeline_smoke/detections.jsonl`.

## Grand Bassin restricted temporal-domain check

The frozen PKLot development parameters were applied without retuning to 793
ordered Grand Bassin frames and seven manually checked continuously occupied
bus bays (5,551 positive slot-frames). Generic temporal defaults were also
fixed before this run. Because no verified vacant slot or transition exists,
this is a positive-only external-domain/stability check, not a complete E4
evaluation.

| Output | Occupied recall | Positive-only F1 | False-free | Post-warm-up changes | Flicker / slot-minute |
|---|---:|---:|---:|---:|---:|
| Frozen E1 classifier | **0.581877** | **0.735679** | **0.418123** | 286 | 6.229806 |
| Frozen E2 detector | 0.465682 | 0.635447 | 0.534318 | 134 | 2.918860 |
| Raw frozen E3 | 0.491443 | 0.659017 | 0.508557 | 203 | 4.421855 |
| E3 + generic hysteresis | 0.336336 | 0.503370 | 0.663664 | **5** | **0.108913** |

This is a strong negative result. Hysteresis reduced state changes by 97.5%
but reduced recall by 0.155107, so the more stable output was substantially
less correct. The pipeline processed 793 frames at 7.147 FPS, above the 2 FPS
source rate.

The branch diagnostics explain the failure. Mean `P_det` was 0.200-0.392 for
the three left bus bays but only 0.006-0.045 for the four right bays.
Classifier probabilities were also weaker on the right. Equal fusion diluted
the remaining classifier evidence, and the generic hysteresis ON threshold
0.58 was incompatible with right-bay fused medians of 0.125-0.346. These
parameters must not be silently retuned on this positive-only sequence.

The pre-existing full baseline report obtained recall 0.824 and flicker 1.29
with its corrected tracking pipeline, but its very low hysteresis thresholds
were selected using this same positive-only source. It is therefore useful
engineering context, not a fair frozen-parameter comparison.

Evidence: `outputs/grand_bassin_frozen/summary.json`,
`outputs/grand_bassin_frozen/occupancy.csv`, and
`outputs/grand_bassin_frozen/evaluation/metrics.json`.

## Grand Bassin continuous-truth candidate audit

The absence of verified negative and transition labels was tested rather than
assumed. Three ordered Grand Bassin sequences were searched: 793, 361, and
195 samples at 2 FPS (1,349 samples, 674.5 nominal seconds by `N/FPS`). All
195 holdout images were made locally available and checksum-recorded.

The fixed-location candidate finder returned 16 hypotheses containing 34
proposed changes. Transition-centred and uniformly sampled full/ROI contact
sheets produced 803 candidate-frame appearances for human adjudication.
Every automated hypothesis was rejected:

| Rejection class | Candidates |
|---|---:|
| Circulation/access vehicle rather than a fixed slot | 8 |
| Complete slot unavailable at an image boundary | 3 |
| Marked-row target with no human-visible state change | 3 |
| Queued/overlapping vehicle, no independent bay | 1 |
| Not a parking space | 1 |

Seven additional targeted hypotheses were also rejected. Focused views showed
an apparent bus departure to be road traffic, an apparent holdout departure
to be a queued/overlapping vehicle, two apparent vacancies to be no-parking
hatching, one to be a circulation lane, and another to contain a dark
vehicle/glare. Broad coordinate-grid checks of the remaining parking rows
found occupied marked bays but no defensible vacant polygon.

Accordingly, no mixed-class or transition ground truth and no corresponding
E4 result were manufactured. The invalid provisional mixed-slot files were
removed; rejected review images were retained under explicit `rejected_*`
names as audit evidence.

Evidence: `TRANSITION_AUDIT.md`,
`data/annotations/grand_bassin_transition_candidate_adjudication.csv`,
`data/annotations/grand_bassin_rejected_manual_hypotheses.csv`, and
`outputs/candidate_search/*_review/`.

## Phase A freeze verification and Phase B data gate

The existing methods were frozen without rerunning or altering any scientific
output. A new read-only verifier checked 17 tracked references covering:

- generic and executed configurations;
- all three PKLot camera split files;
- E0, E1a, E1b, and E2 model artifacts;
- E1a/E1b training summaries and the Fold A selected parameters;
- the CNR-EXT metadata/full-frame archives; and
- the once-only external metrics file.

Every SHA-256 matched. The saved metrics also reported exactly 4,081 complete
image groups and 144,965 slot records, equal to the manifest. The verifier
wrote its new report to
`outputs/phase_a_freeze_audit_20260726_v3/verification.json`; it did not modify
the old results.

The continuous-video audit now has a legal, locally verified pair of distinct
physical scenes. The user accepted the VIRAT agreement and 26 official videos
totaling 1,605,720,653 video bytes were hash-recorded and screened. `0502` is
the development sequence: frame 1659 is the last occupied frame and frame 1660
(55.389 s) is the first vacant frame. `0503` is the locked once-only holdout:
frame 1549 is the last occupied frame and frame 1550 (51.718 s) is the first
vacant frame. Both truth files contain one fixed polygon, continuous half-open
frame intervals, both classes, and one verified departure.

Five event-prioritized `0503` clips and their official event/mapping/object
annotations were screened. Three had no valid marked-slot state change, and
one contained departures only from an unmarked curb/edge row. The fifth,
`VIRAT_S_050300_09_001789_001858.mp4`, contains a departure from a fixed
marked slot followed by 507 vacant frames. Screening stopped when this
holdout was locked; no model output was inspected before the identity, hash,
polygon, truth interval, and E4/E5 configuration were frozen.

DLP requires a raw-video request and uses a drone; EPFL currently
exposes only non-consecutive ground-truth frames; ISLab-PVD has no explicit
dataset license located; and LMOT is not parking-slot data and its official
repository still states that release is forthcoming.

`configs/temporal_protocol_pending.yaml` retains its historical filename for
command compatibility but now has `status: frozen` and returned
`ready_for_experiment: true`. The validator checked source/truth existence,
source SHA-256, decoded dimensions and frame count, selected frame bounds,
polygon bounds, interval coverage, occupied/vacant counts, one transition per
partition, and distinct scene IDs. E4/E5 were therefore authorized. Fusion V2
remained prohibited until E5 could first demonstrate reliable development
behavior.

Evidence: `DATASET_AUDIT.md`, `DATASET_ACCESS_BLOCKER.md`,
`data/manifests/temporal_dataset_audit_20260726.yaml`,
`data/manifests/virat_screening_20260726.yaml`,
`data/manifests/virat_0503_targeted_screening_20260726.yaml`,
`data/annotations/virat_0502_departure_truth.yaml`,
`data/annotations/virat_0503_departure_truth.yaml`,
`outputs/virat_screening_verification_20260726/verification.json`,
`outputs/phase_b_protocol_audit_20260726_v7/validation.json`,
`outputs/virat_0503_targeted_verification_20260726/verification.json`, and
`outputs/phase_a_freeze_audit_20260726_v3/verification.json`.

## Temporal evaluation status

Frozen E4/E5 case studies were run on every frame after excluding a
predeclared 30-frame warm-up from metrics. The methods were:

- T0 raw: YOLOv8n plus polygon mapping, without temporal filtering
  (historical artifact key `e0_raw`);
- E3b raw: the frozen calibrated logistic fusion;
- E4: E3b plus pre-registered asymmetric EMA/hysteresis; and
- E5: YOLOv8n + ByteTrack + one-to-one track-to-slot mapping, stationary-motion
  suppression, and 15-frame occupied/vacant dwell. E5 contains no classifier
  fallback and is not Fusion V2.

| Partition | Method | Macro F1 | Occupied recall | Vacant recall | False-free rate | False-occupied rate |
|---|---|---:|---:|---:|---:|---:|
| `0502` development | T0 raw | 0.456072 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| `0502` development | E3b raw | 0.456072 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| `0502` development | E4 | 0.456072 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| `0502` development | E5 | 0.453626 | 0.990184 | 0.000000 | 0.009816 | 1.000000 |
| `0503` holdout | T0 raw | **0.954119** | 0.951974 | **1.000000** | 0.048026 | **0.000000** |
| `0503` holdout | E3b raw | 0.940952 | 0.980921 | 0.883629 | 0.019079 | 0.116371 |
| `0503` holdout | E4 | 0.834272 | **0.990132** | 0.599606 | **0.009868** | 0.400394 |
| `0503` holdout | E5 | 0.919147 | 0.912500 | **1.000000** | 0.087500 | **0.000000** |

These are negative E4/E5 results. On development, all methods remained
occupied after the true departure because an adjacent/stationary detection
continued to overlap the fixed polygon; E5 mapped that evidence to persistent
track 3. On holdout, T0 raw remained strongest. E4 reduced raw E3b flicker
(29 to 11 unsupported changes) but retained false occupied states too long.
E5 reduced changes to two but first declared vacant at frame 1491, 59 frames
before the true frame 1550, and did not initialize occupied until frame 104.
Consequently, a nominal zero exit-latency value would be misleading and is not
presented as a tracking success.

Post-warm-up timing covered 1,944 development frames and 2,027 holdout frames,
well over the 100-frame requirement:

| Path | Development FPS | Holdout FPS |
|---|---:|---:|
| E4 (classifier + YOLO-World + E3b + hysteresis) | 14.608 | 15.010 |
| E5 (YOLOv8n + ByteTrack + track gate) | 31.778 | 34.248 |
| Combined audit runner | 10.008 | 10.436 |

No per-frame bootstrap interval is reported because each partition contains
only one video group. The results are case studies, not evidence of
cross-video tracking generalization. E5 failed its development reliability
gate, so Fusion V2 remains closed. IDF1 and HOTA remain unreported because no
identity ground truth was created. CNR-EXT remains a consumed once-only static
evaluation and is excluded from all new tuning.

The executed outputs and configuration are frozen by
`data/manifests/temporal_case_study_frozen_20260726.yaml`; all 11 listed
artifacts passed SHA-256 and size verification.

- The existing baseline's Proposed configuration underperformed B0 on the
  repeated-frame static smoke clip because tracker gating/start-up delay
  reduced occupied recall. The baseline adapter was subsequently corrected to
  retain unmatched detections.
- Existing Grand Bassin labels contain only continuously occupied slots, so
  its strong reduction in false-free/flicker cannot establish vacant-space or
  transition performance.
- E1a is standard MobileNetV3-Small. E1b is only a paper-inspired CBAM
  supplement; it is not the paper's exact CBAM/LeakyReLU6/BSConv model.
- YOLO-World's general/open-vocabulary training does not guarantee performance
  for tiny overhead vehicles.
- The 27 PKLot images were already used during earlier baseline development.
  All three PKLot cameras are therefore method-development data, even when a
  camera is held out inside one rotation. None is a final benchmark.

## D1 smoke and local GPU gate

The frozen `D1-NDISPARK-SMOKE-20260727-01` run completed three pretrained
epochs at 640/batch 4 on the local RTX 3060 Laptop GPU. It changed all three
training losses, updated the checkpoint hash, validated all 30 consumed
development images, and reported no NaN, OOM, silent batch reduction, or
material dataloader wait. Its peak Torch reserved memory was 767,557,632 bytes
(0.715 GiB) on 6,441,926,656 total bytes. The smoke validation metrics are
diagnostics and the smoke checkpoint is not formal D1.

Stage G then performed calculation only. For a maximum 50-epoch run at the
same 640/batch-4 setting, measured overhead plus epochs 2-3 gives a 2.43-minute
central estimate; the slowest smoke epoch gives 4.35 minutes, and a doubled
slowest-epoch stress bound gives 8.52 minutes. These are extrapolations rather
than a measured formal-run duration.

The local resource gate passed. Physical batch 4 is retained as the largest
allowed directly executed batch, with nominal batch 64 and 16 post-warm-up
accumulation steps. The 6 GiB local device is the recommended minimum and the
formal run is frozen as `D1-NDISPARK-FT-20260727-01`. Paid/remote GPU rental
is not justified, rental duration is zero, and an A100 is unnecessary.
Stage G accessed no test data and ran no model prediction.

## Formal D1 training

`D1-NDISPARK-FT-20260727-01` completed one fixed-seed local run from the
original COCO-pretrained YOLOv8n. It stopped at epoch 47 under the frozen
patience of 10, selecting epoch 37:

| Checkpoint | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| D1 best, epoch 37 | 0.93708 | 0.88339 | 0.94478 | 0.67556 |
| D1 last, epoch 47 | 0.93105 | 0.89396 | 0.94946 | 0.67232 |

These values are consumed-development diagnostics from the training loop.
They are not an untouched test result and are not yet the canonical
D0/D1/D2 comparison. The best weight SHA-256 is
`0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`.

Training took 299.473 seconds by the epoch CSV (0.083 hours as reported by
Ultralytics). The rounded progress-log peak was approximately 0.814 GiB on
the local 6 GiB RTX 3060. No NaN, OOM, or automatic batch reduction occurred.

The first runner retained a post-run callback audit failure after training and
final best-checkpoint validation completed. No rerun was performed. Offline
recovery loaded no model, all 14 recorded artifacts passed size/SHA-256
verification, and the failure remains part of the result record. NDISPark
count-only test and all slot-occupancy data remained unaccessed.

## Stage I D0/D1/D2 evaluation

The frozen canonical detector comparison used all 30 NDISPark night
development-validation images and 725 human vehicle boxes:

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---:|---:|---:|---:|---:|
| D0 COCO-pretrained YOLOv8n | 0.59394 | 0.51648 | 0.55852 | 0.28468 | 38.733 |
| D1 NDISPark-fine-tuned YOLOv8n | **0.88153** | **0.84160** | **0.89910** | **0.64969** | 37.773 |
| D2 YOLO-World zero-shot | 0.76736 | 0.61655 | 0.72963 | 0.39704 | 37.893 |

D1 was selected by the predeclared mAP@0.5:0.95-then-recall rule before any
test prediction. It improved mAP@0.5:0.95 by 0.36501 and recall by 0.32512
over D0 at similar framework pipeline speed.

One shared count confidence, 0.10, was then selected on development data from
the fixed `0.05:0.05:0.95` grid. The objective minimized mean count MAE across
D0/D1/D2 rather than selecting a separate threshold for each detector.

The frozen rule was applied once to the 117-image, six-camera official
count-only test:

| Method | MAE | RMSE | Mean predicted | Mean true | FPS |
|---|---:|---:|---:|---:|---:|
| D0 | 2.99145 | 5.30119 | 11.16239 | 11.98291 | 108.492 |
| D1 | 3.46154 | 6.78800 | 14.62393 | 11.98291 | 127.315 |
| D2 | **2.58974** | **4.98631** | 10.09402 | 11.98291 | 16.424 |

This is a retained negative result for D1: better development box AP did not
produce the lowest count error under the common rule. D1 tended to over-count,
while D2 under-counted but had the lowest MAE/RMSE at substantially lower
speed. The D1 selection and threshold were not changed after test access.

The test split contains no box truth, so no detector mAP, box precision, box
recall, FP-box, or FN-box result is reported there. FP/FN montages instead use
the consumed development boxes. Visual review confirms D1's recall gain and
also shows low-confidence/duplicate detections in dense rows at the shared
0.10 threshold. The dense night image is only an overlap/occlusion review
candidate because no official occlusion tag exists.

The initial count runner failed before its first prediction because the
Ultralytics user settings directory was not writable. Its preflight-only
directory remains. The successful v2 changed only the settings-directory
location. All 24 selected Stage I artifacts passed SHA-256 and size
verification. See `../data/DETECTOR_EVALUATION_REPORT.md`.

## Stage I-v2 corrected evaluation

Stage I v1 remains frozen. The corrected development evaluation applies
class-agnostic NMS before the D0/D2 source classes are merged into project
class 0 `vehicle`. On the same consumed 30-image/725-box validation, the
`max_det=300` results were:

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Cap-hit images |
|---|---:|---:|---:|---:|---:|
| D0 | 0.63669 | 0.53379 | 0.59954 | 0.30672 | 4/30 |
| D1 | **0.88590** | **0.84606** | **0.90653** | **0.65577** | 30/30 |
| D2 | 0.79014 | 0.62069 | 0.74025 | 0.40295 | 5/30 |

A paired development-only `max_det=1000` arm preserved the D1 > D2 > D0
ranking and slightly reduced rather than improved all three mAP@0.5:0.95
values. D1 still hit the cap on 30/30 images. The preregistered decision
therefore retained 300 and records low-floor candidate saturation as a
limitation.

The common-threshold diagnostic remains 0.10. Independent calibration on the
same development membership and grid selected D0 0.10, D1 0.30, and D2 0.10.
The objective was minimum development count MAE, with lower RMSE and higher
threshold as tie-breaks.

One corrected pass over the already consumed count-only split was frozen
before execution and is labelled post-hoc sensitivity:

| Method | Common 0.10 MAE | Per-model calibrated MAE |
|---|---:|---:|
| D0 | 2.96581 | 2.96581 |
| D1 | 3.45299 | **1.52991** |
| D2 | 2.58974 | 2.58974 |

D1's post-hoc calibrated result supports the diagnosis that v1's common 0.10
operating point amplified its over-counting. It does not create a new
untouched test, did not trigger model reselection or retuning, and does not
measure per-slot occupancy. All 44 critical v2 artifacts passed size/SHA-256
verification. See `../data/STAGE_I_V2_CORRECTED_EVALUATION_REPORT.md`.

## Stage J P0/P1/P2 slot-occupancy integration

The corrected Stage I-v2 detector settings were frozen before any slot
prediction as `P-COMP-PKLOT-DEV-STAGEJ-20260727-01`. P0, P1 and P2 differ
only in D0, D1 and D2 detector evidence; all use the same B1 0.40 polygon
coverage, one-to-one assignment, class-agnostic NMS, `max_det=300` and no
temporal stabilization. Detector thresholds came from the earlier Stage I-v2
development calibration (0.10, 0.30 and 0.10).

The 27 local PKLot images were already consumed by previous development.
Their 1,505 known slot labels therefore support integration diagnostics only:

| Pipeline | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied |
|---|---:|---:|---:|---:|---:|
| P0 = D0 + B1 | 0.768040 | 0.566711 | 0.991979 | 0.433289 | 0.008021 |
| P1 = D1 + B1 | **0.825723** | **0.671070** | 0.990642 | **0.328930** | 0.009358 |
| P2 = D2 + B1 | 0.735168 | 0.504624 | **1.000000** | 0.495376 | **0.000000** |

P1 led this fixed comparison, but the remaining 249 false-free errors are a
material negative result. P2's zero false-occupied errors came with 375
false-free errors. Detector selection remained D1 because it was selected
before this comparison; no Stage J metric changed a model or parameter.

Each pipeline wrote a 27-frame annotated montage, 1,512 occupancy rows, a
header-only events file, 27 detection records, metrics, summary and runtime
metadata. The montage is non-contiguous and is not temporal evidence. All 27
registered source/output artifacts passed size and SHA-256 verification.

The preregistered read-only grouped analysis did not rerun predictions. P1
versus P0 was 6 wins, 2 ties and 19 losses by image. Its mean paired
Macro-F1 difference was -0.020935 with 95% image-group bootstrap interval
[-0.080493, 0.044616]. P1 therefore leads the pooled consumed-development
comparison, but the gain is camera-dependent and was not confirmed there as a
stable per-image improvement.

Run-inclusive mean throughput is confounded by sequential lazy loading and
first CUDA initialization. Median end-to-end frame latency was 34.105 ms for
P0, 31.974 ms for P1 and 32.668 ms for P2; the anomalous cold-start means are
retained rather than rewritten. Stage I-v2 remains the controlled source for
detector runtime comparisons.

## Stage K untouched PKLot slot-occupancy evaluation

The original local-inventory blocker is preserved, but an additive v2 gate
later passed before predictions after 90 complete JPG/XML pairs were recovered
from the partial official PKLot archive. The three selected camera/date groups
contain 90 unique image hashes with zero overlap against Stage J, 5,034 known
slot labels and six excluded unknown labels. Membership within each group was
timestamp-sorted and evenly spaced. Raw and truth-overlay contact sheets were
reviewed before `P-COMP-PKLOT-TEST-STAGEK-20260727-01` was frozen.

The frozen P0/P1/P2 pipelines were executed once without retraining, threshold
changes, geometry changes or post-result detector selection:

| Pipeline | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied | Slot AP |
|---|---:|---:|---:|---:|---:|---:|
| P0 = D0 + B1 | 0.785612 | 0.544519 | 0.992883 | 0.455481 | 0.007117 | 0.718263 |
| P1 = D1 + B1 | **0.808398** | **0.598044** | 0.984148 | **0.401956** | 0.015852 | **0.739705** |
| P2 = D2 + B1 | 0.796548 | 0.559444 | **0.997735** | 0.440556 | **0.002265** | 0.729244 |

The pooled table alone is insufficient. Camera-macro F1 is 0.841099 for P0,
0.824835 for P1 and 0.849168 for P2. P1 improved PUCPR but was lower than P0
on UFPR04 and UFPR05. Across 90 images, P1 versus P0 was 17 wins, 11 ties and
62 losses; the paired mean difference was -0.040322 with 95% interval
[-0.068804, -0.009457]. P2 versus P0 was 33 wins, 34 ties and 23 losses, with
paired mean 0.035781 and interval [0.009939, 0.062788].

D1 remains the candidate selected before slot-test access, but Stage K does
not support an across-camera P1 superiority claim. The test result did not
replace D1 with P0 or P2.

Date and weather tables were produced afterward from the frozen occupancy CSV
files without model prediction or parameter selection. Each camera has one
selected date, and cloudy weather occurs only for UFPR04, so those strata are
confounded with camera and are descriptive only.

Stage K binds 43/43 main artifacts, 9/9 stratified-analysis artifacts and
11/11 data-gate v2 artifacts. Full interpretation is in
`../data/STAGE_K_FINAL_REPORT.md`.

## Stage L integrated Part I workflow

Stage L connected D1, B1, E1b, asymmetric uncertainty gating, temporal
hysteresis and optional ByteTrack without modifying Stage J/K. On the 90
previously consumed Stage K images, P3 reached 0.987061 Macro F1 versus
0.808398 for P1. It improved 78 images, tied 12 and lost none; the paired
mean per-image gain was 0.191884 with 95% interval
[0.158389, 0.227038].

The fixed E1b-only ablation was higher at 0.992226 Macro F1. P3 instead
reduced false-free rate from E1b's 0.016469 to 0.004632 while increasing
false-occupied rate from 0.001618 to 0.017147. P3 is therefore a
recall-oriented operational trade-off, not an unconditional replacement for
E1b.

The complete 1,974-frame VIRAT 0502 run produced a negative result. D1+B1
remained detector-positive in all 314 post-departure frames, so E1b was never
called; ByteTrack confirmed the adjacent stationary vehicle selected by the
oblique slot polygon, and all four variants missed the departure. Full
interpretation is in `../data/STAGE_L_INTEGRATED_WORKFLOW_REPORT.md`.

## Stage O raw detector-only low-light adaptation

Stage O is an additive consumed-development diagnostic on the four released
LMOT validation sequences. Unlike Stage N-v2/v3, it calls `YOLO.predict`
without loading a tracker. All arms retain `imgsz=640`, confidence 0.30, NMS
IoU 0.70, class-agnostic NMS, `max_det=300`, the unified
car/motorcycle/bus/truck class and the frozen person/bicycle suppression rule.
LMOT has no slot polygons or occupied/vacant truth, so this section reports
vehicle detection only and cannot establish a parking-occupancy Macro F1
change.

The primary pooled/micro result sums all GT, predictions, TP, FP and FN and
uses one confidence ordering with isolated sequence/frame keys:

| Method | Dark precision | Dark recall | Dark AP50 | Dark AP50-95 | Dark TP | Dark FP | Dark FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| O0: frozen D1, original sRGB | 0.722928 | 0.034694 | 0.034954 | 0.018189 | 2,390 | 916 | 66,497 |
| O1: sequence-gated Gamma/CLAHE | 0.454643 | 0.043216 | 0.034143 | 0.017125 | 2,977 | 3,571 | 65,910 |
| O2: pretrained Retinexformer preprocessing | 0.581802 | 0.019307 | 0.017284 | 0.011517 | 1,330 | 956 | 67,557 |
| O3: supervised D1-LL | 0.725393 | 0.259222 | 0.230516 | 0.116695 | 17,857 | 6,760 | 51,030 |

O1 is a negative engineering result: its small recall increase comes with a
large precision loss and lower AP50/AP50-95. O3 used one ordinary supervised
fine-tune from D1, paired sequence-grouped LMOT train sRGB boxes, and the
existing NDISPark daylight parking-training split. The frozen patience-five
rule stopped the single run after epoch 9, with epoch 4 best. No consistency
loss, hyperparameter search, LMOT-validation threshold tuning or tracker
tuning was performed.

O2 is also a negative result: full-resolution pretrained Retinexformer
preprocessing reduces dark recall, AP50 and AP50-95 below O0 and runs at
1.198 whole-run frames/s. O3 alone passes every predeclared eligibility
check. It is selected as the D1-LL detector-side candidate, raising dark
pooled recall from 0.034694 to 0.259222 and AP50 from 0.034954 to 0.230516
without a precision loss. The truth-free four-frame P3 interface smoke
completed with the unchanged B1/E1b/F2/E4 path and no tracker; its metrics
status is `not_computed_no_truth`, so it adds no occupancy-performance claim.

The completed P3 remains the system main path. Stage O replaces only the
detector side if a candidate passes the predeclared rule. B1, E1b, F2 and E4
stay frozen. Without a new nighttime fixed-camera parking dataset with
per-slot truth, the selected detector receives only a truth-free interface
smoke; no low-light occupancy or event improvement is reported.

## Stage P parking retention and Stage Q-v2 external occupancy

Stage P remains a consumed-development parking-domain diagnostic. Its frozen
decision is `FAIL`; D1-LL is not eligible to replace D1 from that evidence.

Stage Q-v2 subsequently acquired the official UPM-GTI Test archive and froze
376 low-light images from 17 previously unused Test sequences before model
output. The 21 polygons were aligned with the official numbering and
human-confirmed before the two formal runs. Both methods used the same
confidence 0.30, NMS IoU 0.70, B1 0.40 one-to-one mapping, E1b/F2 threshold
0.76 and E4 parameters, with no tracker:

| Metric | P3-D1 | P3-D1-LL |
|---|---:|---:|
| Macro F1 | **0.664318** | 0.617484 |
| Occupied precision | **0.368530** | 0.269572 |
| Occupied recall | 0.446115 | **0.457393** |
| Vacant recall | **0.914060** | 0.860665 |
| False-free | 0.553885 | **0.542607** |
| False-occupied | **0.085940** | 0.139335 |
| Accuracy | **0.866768** | 0.819909 |
| Count MAE | **1.329787** | 1.909574 |

D1-LL recovers nine additional occupied labels but introduces 379 additional
false-occupied labels. It therefore has a small recall advantage and a
material precision, vacant-recall, Macro-F1 and count-error regression. D1
remains the default, D1-LL remains a secondary frozen comparison, and Stage
P2 remains `FAIL`.

The source lacks reliable timestamps/FPS. The 90 ground-truth transitions
are reported in frame units only: D1 matched 39 and D1-LL matched 41, with a
median four selected frames for non-early matched stabilization in both
methods. No seconds latency is claimed. This is one-camera external
low-light slot-occupancy evidence, not detector AP, tracker robustness or a
universal generalization result. Full interpretation is in
`../data/STAGE_Q_V2_UPM_GTI_EXTERNAL_EVALUATION_REPORT.md`.
