# Literature-Core Results

This file is updated only with executed commands. No placeholder value is
presented as an experiment result.

Run date: 25-26 July 2026 (Asia/Shanghai)

| Check | Result |
|---|---|
| Existing baseline unit tests | **23/23 passed** after all work |
| Literature-core unit tests | **67/67 passed** |
| Source compilation | `python -m compileall` passed |
| Python/OpenCV available | Yes |
| CUDA / RTX 3060 available | Yes, 6 GiB |
| torchvision MobileNetV3 API | Available |
| Ultralytics YOLO-World API | Available |
| Local MobileNetV3 pretrained checkpoint | Official weight acquired and hash verified |
| Local YOLO-World/CLIP checkpoints | Official weights acquired and hashes verified |
| Existing baseline outputs overwritten | No |
| Continuous mixed/transition truth found locally | No; 16 automated and 7 targeted hypotheses were rejected |
| Frozen artifact audit | **17/17 SHA-256 checks passed**; 4,081 frames and 144,965 slot records matched |
| VIRAT bounded screening | 24 official videos / 1,430,406,456 video bytes; **0502 frame truth verified, 0503 targeted clips rejected** |
| Temporal protocol gate | Schema valid; deliberately **not experiment-ready** because no second physical scene passed |

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

The continuous-video audit did not produce a legal, locally verified pair of
development/holdout sequences. The user accepted the VIRAT agreement and 24
official videos totaling 1,430,406,456 video bytes were hash-recorded and
screened. The `0502` candidate now has a fixed polygon and frame-level truth:
frame 1659 is the last occupied frame and frame 1660 (55.389 s) is the first
vacant frame. It remains an unassigned candidate and was not used for tuning.

Three event-prioritized `0503` clips and their official event/mapping/object
annotations were then screened. An e06 clip and one e05 clip had no vehicle
slot-state change. A longer clip with three e05 events included two subsequent
vehicle departures, but both originated from an unmarked curb/edge row with
no complete fixed slot polygon; the third vehicle remained parked. These are
negative screening outcomes, not model results. Two more e05 downloads were
attempted but the official catalog API returned HTTP 502 twice, so `0503`
screening is explicitly non-exhaustive. No second physical scene passed.

DLP requires a raw-video request and uses a drone; EPFL currently
exposes only non-consecutive ground-truth frames; ISLab-PVD has no explicit
dataset license located; and LMOT is not parking-slot data and its official
repository still states that release is forthcoming.

`configs/temporal_protocol_pending.yaml` passed validation but returned
`ready_for_experiment: false`. A frozen protocol can no longer become ready
from valid-looking fields alone: the validator now checks source/truth
existence, source SHA-256, decoded dimensions and frame count, selected frame
bounds, polygon bounds, interval coverage, occupied/vacant counts, and at
least one transition in each partition. This is the intended result: E4, E5,
and Fusion V2 remain prohibited until a second physical scene and a
scene-level frozen split are complete.

Evidence: `DATASET_AUDIT.md`, `DATASET_ACCESS_BLOCKER.md`,
`data/manifests/temporal_dataset_audit_20260726.yaml`,
`data/manifests/virat_screening_20260726.yaml`,
`data/manifests/virat_0503_targeted_screening_20260726.yaml`,
`data/annotations/virat_0502_departure_truth.yaml`,
`outputs/virat_screening_verification_20260726/verification.json`,
`outputs/phase_b_protocol_audit_20260726_v6/validation.json`,
`outputs/virat_0503_targeted_verification_20260726/verification.json`, and
`outputs/phase_a_freeze_audit_20260726_v3/verification.json`.

## Temporal evaluation status

A restricted positive-only E4-style check has been run, but full E4/E5 remain
unavailable. PKLot captures are not frame-contiguous, and Grand Bassin has no
verified negatives or transitions. Vacant recall, false-occupied rate,
transition latency, mixed-class flicker, IDF1, and HOTA remain unclaimed until
a continuous sequence with suitable human truth is provided. CNR-EXT remains
a consumed once-only static evaluation and is excluded from all new tuning.

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
