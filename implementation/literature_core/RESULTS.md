# Literature-Core Results

This file is updated only with executed commands. No placeholder value is
presented as an experiment result.

Run date: 25 July 2026 (Asia/Shanghai)

| Check | Result |
|---|---|
| Existing baseline unit tests | **23/23 passed** after all work |
| Literature-core unit tests | **37/37 passed** |
| Source compilation | `python -m compileall` passed |
| Python/OpenCV available | Yes |
| CUDA / RTX 3060 available | Yes, 6 GiB |
| torchvision MobileNetV3 API | Available |
| Ultralytics YOLO-World API | Available |
| Local MobileNetV3 pretrained checkpoint | Official weight acquired and hash verified |
| Local YOLO-World/CLIP checkpoints | Official weights acquired and hashes verified |
| Existing baseline outputs overwritten | No |
| Continuous mixed/transition truth found locally | No; 16 automated and 7 targeted hypotheses were rejected |

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
eliminated false-occupied cases in this holdout, but had more false-free
errors. E3 perfectly fitted the development set yet did not improve on E0 in
test macro F1 and increased false-free errors. Therefore this pilot does not
support a claim that weighted fusion generalizes better than either branch;
it shows development-set over-selection and non-complementary errors on
UFPR05.

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

## Post-hoc three-camera rotation

Two additional train/development/test camera rotations were executed after the
original Fold A holdout result had been observed. Every fold still kept its
test camera out of training and parameter selection. The table reports
camera-equal, not sample-weighted, macro F1.

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

The broader result is more positive for fusion than Fold A alone, but it does
not establish one deployable universal configuration. Development-selected
E3 classifier weight varied from 0.30 to 0.50 and the threshold from 0.23 to
0.38. E2 was particularly camera-sensitive: it won against E0 in two folds
but fell to 0.823569 on PUCPR, yielding the largest standard deviation
(0.077391). Training-camera sample counts also differ substantially, so these
three folds are a robustness diagnostic rather than a large-sample estimate.

Evidence: `outputs/cross_camera/summary/summary.json`,
`outputs/cross_camera/summary/fold_results.csv`, and per-fold training and
ablation directories under `outputs/cross_camera/`.

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

## Temporal evaluation status

A restricted positive-only E4-style check has been run, but full E4/E5 remain
unavailable. PKLot captures are not frame-contiguous, and Grand Bassin has no
verified negatives or transitions. Vacant recall, false-occupied rate,
transition latency, mixed-class flicker, IDF1, and HOTA remain unclaimed until
a continuous sequence with suitable human truth is provided.

- The existing baseline's Proposed configuration underperformed B0 on the
  repeated-frame static smoke clip because tracker gating/start-up delay
  reduced occupied recall. The baseline adapter was subsequently corrected to
  retain unmatched detections.
- Existing Grand Bassin labels contain only continuously occupied slots, so
  its strong reduction in false-free/flicker cannot establish vacant-space or
  transition performance.
- Standard MobileNetV3-Small is not the paper's CBAM/BSConv/LeakyReLU6 model.
- YOLO-World's general/open-vocabulary training does not guarantee performance
  for tiny overhead vehicles.
- The 27 PKLot images were already used during earlier baseline development.
  The UFPR05 result is camera-disjoint for this run, but it is a pilot
  holdout—not a globally untouched final benchmark.
