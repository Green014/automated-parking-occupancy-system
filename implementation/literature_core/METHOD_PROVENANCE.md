# Method Provenance

All provenance labels describe this repository's implementation, not merely
the paper category.

| Module | Part I source | Code/weights source | License | Project change | Status |
|---|---|---|---|---|---|
| E1a slot patch classification | Yuldashev et al. [2] | torchvision `mobilenet_v3_small`; ImageNet weights from PyTorch | BSD-style PyTorch/torchvision terms; ImageNet weights subject to their dataset terms | OpenCV perspective-normalized 224x224 slot patch, standard MobileNetV3-Small, two-class head, ImageNet transfer learning | **Adaptation**, not exact reproduction |
| E1b paper-inspired classification | Yuldashev et al. [2] for LeakyReLU6/CBAM/BSConv direction; Woo et al. for CBAM | Local PyTorch modules layered on torchvision | Project code plus the runtime terms above | Identity-initialized channel/spatial CBAM supplements pretrained SE; shallow capped LeakyReLU6 is separately switchable; BSConv is not implemented | **Paper-inspired local adaptation**, not exact reproduction |
| Open-vocabulary detection | Cheng et al., YOLO-World [6] | Original: `AILab-CVC/YOLO-World`; runtime adapter: Ultralytics `YOLOWorld`; `yolov8s-worldv2.pt` | Original repository GPL-3.0; Ultralytics package/weights AGPL-3.0 for this academic workflow | Prompts restricted to `car`, `truck`, `bus`, `motorcycle`; raw box and confidence retained | **Adaptation** |
| Detection-to-slot mapping | APSD-OC [1] supports linking detections and slot geometry; baseline provides an existing OpenCV implementation | Implemented locally with `cv2.intersectConvexConvex` | Project code | Confidence-weighted fraction of each slot polygon covered by a prompted object box, with one-to-one greedy assignment | **Inspired/local integration** |
| E3a weighted evidence fusion | Combination motivated by [2] and [6] | Implemented locally | Project code | Development-selected normalized weighted sum of raw `P_cls`, detector evidence, and optional `P_track`; every component is logged | **Local system integration / baseline** |
| E3b calibrated logistic fusion | Probability calibration literature; branch choices motivated by [2] and [6] | Implemented locally with deterministic NumPy IRLS | Project code | Separate monotonic Platt-style branch calibration followed by non-negative logistic fusion of calibrated log-odds; camera-grouped fitting only | **Local innovation / proposed** |
| Temporal hysteresis | Existing baseline; tracking motivation from TrackTrack [7] | Reimplemented in the independent package | Project code | Asymmetric EMA with separate occupied/vacant thresholds; early/on-time/delayed/missed stable transitions are separated from ordinary flicker using signed timing error | **Adaptation/reuse** |
| E5 track-aware gate | Tracking motivation from TrackTrack [7], but no code copied | Local project code plus Ultralytics ByteTrack runtime | Project code; Ultralytics runtime terms apply | YOLOv8 detections receive ByteTrack IDs, then deterministic one-to-one slot assignment, stationary-motion suppression, and occupied/vacant dwell; synthetic tests precede a frozen negative case study | **Local paper-inspired adaptation**, not TrackTrack reproduction |
| P3 integrated uncertainty gate | Combines the Part I YOLO category, APSD-OC-inspired geometry, E1b MobileNetV3-CBAM, and tracking/temporal motivation | Local Stage L code using the existing D1, B1, E1b, E4, and optional E5 artifacts | Project code; upstream model/runtime terms still apply | D1 is primary; E1b reviews only detector-negative slots at the frozen 0.76 threshold; E4/E5 are continuous-video support | **Local integration / meaningful modification**, not an exact paper reproduction |
| Stage O raw detector-only diagnostic | LMOT [8] motivates paired light/dark robustness measurement | Local `YOLO.predict` adapter; frozen D1/D1-LL weights | Project code plus Ultralytics runtime; LMOT data CC BY-NC 4.0 | Separates detector boxes from Stage N tracker-emitted boxes; fixed thresholds, pooled/micro primary metrics and per-sequence macro diagnostics | **Local controlled diagnostic**, not parking-occupancy evaluation |
| O1 brightness-gated Gamma/CLAHE | General image-processing baseline; no paper result transferred | OpenCV operations implemented locally | Project code plus OpenCV terms | One brightness decision per sequence/source; frozen Gamma/CLAHE tuple; no per-frame parameter adaptation | **Engineering baseline**, not a paper algorithm |
| O2 pretrained low-light preprocessing | GLARE [10] primary feasibility route; Retinexformer [11] predeclared fallback | Official upstream repositories/checkpoints in an isolated environment | GLARE Apache-2.0; Retinexformer MIT; upstream model/data terms apply | Preprocessing-only diagnostic in front of unchanged D1; no enhancer fine-tuning and no reproduction of the papers' detector experiments | **Direct pretrained preprocessing use with local adapter** |
| O3 D1-LL detector adaptation | LMOT [8] paired sRGB boxes motivate low-light detector adaptation | Frozen formal D1 initialization; local Ultralytics fine-tune on grouped LMOT train plus NDISPark daylight | Derived weights inherit applicable LMOT CC BY-NC 4.0, NDISPark ODC-By and Ultralytics terms | One ordinary supervised run; paired sequence-level split, deterministic sampling, no consistency loss or hyperparameter search | **Project-specific supervised adaptation**, not LTrack reproduction |
| Automatic slot discovery | APSD-OC [1] | No official implementation confirmed | Not applicable | No homography/DBSCAN reproduction is claimed | **Not implemented / future work** |
| External holdout data | Amato et al., CNRPark+EXT | Official `cnrpark.it` page and `fabiocarrara/deep-parking` GitHub release | ODbL-1.0 stated by the official page | CNR-EXT occupancy metadata joined to official camera boxes and released full frames; boxes are scaled, not converted into claimed precise polygons | **Third-party data, once-only external evaluation** |
| Conditional temporal data | Oh et al., VIRAT | Official `viratdata.org` and Kitware collection | VIRAT individual Usage Agreement; restricted redistribution and PII duties | User acceptance is recorded; 26 official clips were checksum-verified; two distinct-scene single-slot departure truths are project-created and support a limited frozen case study | **Third-party data with local truth; not a native VIRAT occupancy benchmark** |

## Code and runtime addresses

- torchvision MobileNetV3-Small API:
  https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- YOLO-World authors' repository:
  https://github.com/AILab-CVC/YOLO-World
- Ultralytics YOLO/YOLO-World runtime reference:
  https://docs.ultralytics.com/reference/models/yolo/model/
- APSD-OC and the Improved MobileNetV3 paper: no official implementation was
  confirmed during this audit, so no unofficial repository is presented as
  authoritative.
- TrackTrack: no implementation was copied or reproduced. It only motivates
  the E5 tracking direction; the implemented gate is a simpler local rule
  system and is reported under that narrower label.
- VIRAT Ground Release 2.0:
  https://viratdata.org/
- VIRAT Usage Agreement:
  https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf
- Dragon Lake Parking candidate:
  https://sites.google.com/berkeley.edu/dlp-dataset

## Baseline naming provenance

The cross-package baseline names are closed in
`../configs/baseline_methods.yaml` and explained in
`../BASELINE_CLOSURE.md`:

- B0 is the local YOLOv8 bounding-box-centre mapping baseline.
- B1 is the local YOLOv8 polygon-coverage baseline.
- E0 is the historical frozen static CNR-EXT YOLOv8 coverage baseline.
- T0 is the raw YOLOv8 temporal comparator. Frozen temporal artifacts retain
  their original `e0_raw` key.

These are reporting/configuration identities for local project code. They are
not claims that this repository was forked from an external parking-management
system.

## Temporal data provenance boundary

The user personally accepted the VIRAT agreement on 26 July 2026. The project
then downloaded 26 unmodified official video items (1,605,720,653 video bytes)
for bounded screening. The targeted `0503` extension also downloaded the
official event/mapping/object files for five clips. Item IDs, SHA-256 values,
video properties, event-selection evidence, and negative decisions are
recorded in `data/manifests/virat_screening_20260726.yaml` and
`data/manifests/virat_0503_targeted_screening_20260726.yaml`.

The official Release 2.0 document defines filename digits `XXYY` as the
physical scene and `ZZ` as the sequence. The local acquisition helper enforces
that grouping, explicit byte budgets, recorded agreement acceptance, atomic
non-overwriting downloads, bounded retry of transient 429/5xx/network errors,
and checksum verification. Those safeguards and the parking-suitability
decisions are project adaptations, not VIRAT methods or labels.

DLP is deferred because the raw video requires a request and drone motion may
break fixed ROIs. EPFL's current official page provides only non-consecutive
ground-truth archives, ISLab-PVD lacks an explicit dataset license, and LMOT
lacks parking-slot truth and a released licensed dataset.

The two eligible VIRAT clips have project-created truth. `0502` development
uses frame 1660 as first vacant; `0503` holdout uses frame 1550. Both contain a
fixed polygon and half-open occupied/vacant intervals. The holdout was locked
before any model output. Polygon coordinates, interval labels, transition
adjudication, scene-level split, track-to-slot assignment, motion threshold,
and dwell rules are local project adaptations; none is a native VIRAT label or
a Part I paper implementation.

The frozen E4/E5 run is explicitly a two-video, one-slot-per-video departure
case study. E5 did not beat T0 on holdout and failed to recognize vacancy on
development, so it cannot support a general tracking-improvement claim.

## Frozen runtime artifacts

| Artifact | Local role | SHA-256 |
|---|---|---|
| `mobilenet_v3_small-047dcff4.pth` | torchvision ImageNet initialization | `047DCFF4ADDEF86EA5BC2EFF13C9614DC11F47AB1160D0A71A25E7DB994F4E1F` |
| `yolov8s-worldv2.pt` | Ultralytics YOLO-World detector | `9B2C17AB6124A913E9B3A5C170617920D91B0F01111A8479DA69F00E2CF27792` |
| `ViT-B-32.pt` | CLIP text encoder cache used by the adapter | `40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF` |

The generated two-class `best.pt` checkpoint is an experiment output rather
than a third-party artifact. Its SHA-256 is
`B67A8A5217902601FDF7BCEDF64F739AB03E688AFE86A386E56106F9F968651E`;
its training settings and data split are stored inside the checkpoint and
repeated in `outputs/mobilenet_pilot/training_summary.json`.

Post-hoc camera-rotation classifier checkpoints:

- Fold B SHA-256:
  `5E8903987E2225EAE301ECC19DE6260891CCD33CFF70D532C38B708747A5932C`;
- Fold C SHA-256:
  `92110D18DB5587AF0C6766219BDD5A8B4C15920E37A7D186E5885CB8CD5697E3`.

Their split membership, best epoch, test-not-loaded count, and training
settings are stored in the corresponding
`outputs/cross_camera/*_classifier/training_summary.json` files.

The selected architecture-ablation checkpoints are also local outputs:

- E1a standard:
  `outputs/mobilenet_variant_ablation/standard/best.pt`,
  SHA-256 `600C90CCB271FB1DB3E39F87E500866E81C11781A8BB5A03A4285BE1CAF276B4`;
- E1b paper-inspired CBAM supplement:
  `outputs/mobilenet_variant_ablation/cbam_supplement/best.pt`,
  SHA-256 `F6966DABE0801F221CC6E67B9EE117AF1B06C93A7E34C96D25771572616DDBE3`.

Their component flags, parameter counts, training time, and selected internal
development threshold are embedded in each checkpoint and its adjacent
`training_summary.json`.

## Stage E detector-comparison integration

The new D0/D1/D2 comparison layer is a local project implementation in
`../src/parking_occupancy/detector_comparison.py`; it is not a fork of an
external parking-management repository. Ultralytics supplies YOLO/YOLO-World
model loading, prediction, NMS, and the `ap_per_class` metric primitive. The
project supplies the frozen method registry, file/hash preflight, source-class
filtering, mapping to the common `vehicle=0` class, one-to-one IoU matching,
non-overwriting artifact contract, and comparison reports.

This adapter is necessary because the frozen Stage D data uses class 0 while
COCO-pretrained D0 emits vehicle classes 2, 3, 5, and 7. Passing those COCO IDs
to Ultralytics validation as a dataset class filter would remove class-0
truth. The project therefore filters predictions first and evaluates the
canonical predictions against unchanged truth. The comparison remains blocked
until a Stage F/H D1 weight and its recorded SHA-256 are supplied.

The Stage F smoke runner in
`../src/parking_occupancy/training_smoke.py` is also a local integration. It
uses Ultralytics for pretrained model loading, augmentation, optimization,
validation, plotting, and checkpoint serialization. The project adds frozen
input/hash gates, non-overwriting outputs, epoch/batch callback measurements,
NaN/OOM and loss checks, checkpoint hash verification, failure retention, and
the machine-readable smoke summary. The successful smoke checkpoint is
explicitly not the formal D1 model.

The Stage G GPU decision implementation in
`../src/parking_occupancy/gpu_decision.py` is likewise local project code.
It reads the frozen smoke summary, validates that the executed 640/batch-4
protocol completed without OOM or batch reduction, and makes transparent
runtime and memory projections. The memory heuristic scales measured Torch
reserved memory by batch ratio, squared image-size ratio, and a declared 1.25
safety factor. The 960/1280 values are planning estimates, not measured
benchmarks. The formal configuration remains at the execution-validated
640/batch-4 condition and uses Ultralytics 8.4.104's nominal-batch
accumulation behavior. No external GPU estimator, remote benchmark, paid
service, or generated performance result was used.

The Stage H formal runner in
`../src/parking_occupancy/formal_training.py` is a local orchestration and
audit layer around Ultralytics 8.4.104. Ultralytics supplies YOLOv8n transfer
learning, augmentation, optimization, early stopping, validation, curve
generation, and checkpoint serialization. The project layer enforces the
frozen configuration and SHA-256 registry, checks the prepared one-class
dataset and D0 initialization, prohibits output overwrite, records resources,
and verifies all retained artifacts.

The first formal runner completed training before its post-run resource audit
failed: Ultralytics calls `on_fit_epoch_end` again while validating the final
best checkpoint, which the reused Stage F callback guard counted as an extra
training epoch after early stopping. The failure and traceback are preserved.
The callback now requires an active training epoch. Existing artifacts were
finalized without loading or running a model, and no second formal seed was
created. The rounded 0.814 GiB progress-log memory maximum is therefore
reported with its precision limitation; no exact callback peak or dataloader
wait measurement is invented.

## Stage I detector selection, counting, and qualitative layer

`../src/parking_occupancy/stage_i_evaluation.py` is local project code, not a
fork of an external parking system. It consumes the frozen canonical
D0/D1/D2 outputs, selects the detector with the preregistered mAP@0.5:0.95 and
recall ordering, and searches one declared confidence grid using development
counts only. Its shared-threshold objective, artifact/hash gates,
non-overwriting count runner, per-camera count aggregation, and qualitative
montage selection are project-specific orchestration.

Ultralytics still supplies detector loading, preprocessing, inference, and
NMS. The project adapter maps D0 COCO vehicle classes, D1 class 0, and D2 text
prompt classes into one `vehicle` class under identical conditions. The
count-only test evaluation uses the official count field through the local
MAE/RMSE implementation. It does not infer vehicle boxes from counts and does
not call count errors mAP.

Qualitative TP/FP/FN matching is computed at IoU 0.5 only on the consumed
development validation where human vehicle boxes exist. The selected dense
night frame is a visual overlap/occlusion candidate, not an official
occlusion annotation. The formal D1 training curve remains the Ultralytics
`results.png` output; Stage I references and freezes it rather than recreating
or altering it.

The count result is deliberately not repackaged as a D1 improvement. D2 had
the lowest MAE under the shared rule, while D1 remained the pre-test
box-metric selection. Neither result supports a claim about slot occupancy
Macro F1.

Stage I-v2 remains the same local project evaluation layer and is not an
official Ultralytics parking-management fork. Ultralytics 8.4.104 supplies
model loading, preprocessing, prediction, and its NMS implementation. The
project now explicitly passes `agnostic_nms=true` and also enforces a local
class-agnostic canonicalization safety check before all allowed source
categories become class 0 `vehicle`.

The v2 threshold-grid aggregation, per-model MAE/RMSE selection with
deterministic tie-breaks, max-det saturation audit, peak CUDA-memory capture,
two-root artifact verifier, and consumed-test boundary checks are local
implementations. The one post-hoc count pass predicts at the frozen 0.001
floor and derives both threshold regimes from the same detection log. It does
not train D1, change Ultralytics weights, infer boxes from count truth, or use
test counts for threshold/model selection.

## Stage J detector-to-slot integration

`../src/parking_occupancy/stage_j_occupancy.py`,
`../scripts/run_stage_j_occupancy.py` and
`../scripts/verify_stage_j_artifacts.py` are local project implementations,
not forks of Ultralytics Parking Management or another parking repository.
Ultralytics supplies D0/D1/D2 model loading and prediction. OpenCV supplies
image/video I/O, polygon intersection primitives and drawing.

The local layer verifies image, protocol and weight hashes; canonicalizes all
allowed source classes to project class 0 `vehicle`; applies the existing B1
one-to-one confidence-weighted polygon mapping; exports identical per-method
logs; computes slot metrics and camera/weather strata; and refuses an existing
output directory. It does not alter the D1 checkpoint, retrain any model or
add E1b/temporal fusion.

The Stage J protocol reuses Stage I-v2 development-selected detector
thresholds without reading slot truth for parameter selection. Its 27 PKLot
images are labelled `consumed_development`. The ordered videos are explicitly
montages and their event files are header-only. The original Stage K
local-inventory blocker is retained as the state at Stage J closure.

## Stage K untouched slot evaluation

Stage K is an additive local evaluation layer built from the same project
adapter and B1 mapping, not a new external parking-system fork. Before any
Stage K prediction, 90 complete official PKLot JPG/XML pairs were selected
from three previously unused camera/date groups by timestamp-sorted evenly
spaced sampling. Their hashes have zero overlap with Stage J. Official XML
polygons and occupied/vacant attributes supply slot truth; six unknown labels
are retained and excluded.

The P0/P1/P2 checkpoints, thresholds and mapping rules were frozen before the
single test execution. The result did not retrain D1 or reselect a detector.
Subsequent date/weather tables read only the frozen occupancy CSV files.
Because each camera contributes one date and cloudy weather occurs only for
UFPR04, those strata are descriptive and confounded with camera.

## Stage M official baseline and TrackTrack execution

Stage M does not copy or claim to reproduce the TrackTrack author's codebase.
The exact relationship is **paper method executed through Ultralytics
implementation**. The local adapter calls Ultralytics 8.4.104
`model.track(..., persist=True, tracker=<frozen TrackTrack YAML>)`, which
selects `ultralytics.trackers.track_tracker.TRACKTRACK` through the installed
tracker registration. The installed registration file, implementation,
upstream configuration and local frozen configuration are all recorded by
SHA-256. The author repository remains the paper-code provenance source; the
Ultralytics package is the code actually executed.

`OS0-Controlled` executes the official
`ultralytics.solutions.ParkingManagement` object without modifying
`site-packages`. Ultralytics supplies model loading, TrackTrack, official
centre-point-in-polygon occupancy totals and annotations. The local project
adapter supplies source-bound state reset, per-slot replay of the identical
centre-point rule, an assertion that the replay total matches the official
total, metrics and the seven-file export contract. The replay is logging and
evaluation adaptation, not a replacement occupancy algorithm.

T0--T3 are local controlled integrations: T0 retains the Stage L asymmetric
D1+B1+E1b gate without E4 or a tracker; T1 adds the unchanged E4 temporal
filter; T2 uses the unchanged Stage L ByteTrack parameters; T3 substitutes
the frozen TrackTrack configuration. D1, B1, E1b, thresholds and shared
inference settings remain fixed. T2/T3 tracking identities are supporting
evidence, not an additional detector or proof of slot occupancy.

The Stage M PKLot repeated-image run is an interface smoke with no truth.
AODRaw remains detector-only and licence-blocked, and LMOT can support only a
validation tracking diagnostic because it lacks parking-slot polygons and
occupied/vacant truth. Neither box AP nor HOTA/IDF1 may be relabelled as slot
Macro F1. See
`../data/STAGE_M_OPEN_SOURCE_TRACKING_ROBUSTNESS_REPORT.md`.

## Important differences from the papers

### Improved MobileNetV3 [2]

The paper uses LeakyReLU6 in the shallow network, CBAM instead of SE, and
blueprint separable convolutions instead of depth-wise separable convolutions.
E1a intentionally uses the unmodified torchvision MobileNetV3-Small backbone
with ImageNet transfer learning. E1b implements verifiable component
ablations, but differs from the paper in three important ways: pretrained SE
is retained and supplemented rather than removed, LeakyReLU6 uses the local
explicit negative slope 0.1, and BSConv is not implemented. The E1b label is
therefore always "paper-inspired adaptation"; paper result tables cannot be
transferred to it.

### Detector evidence and E3b

The quantity historically named `P_det` is `detection confidence x slot
coverage`. Neither YOLO-World nor this geometric product makes it a native
occupancy probability. E3b first maps each branch through a fitted monotonic
calibrator, then uses only non-negative branch coefficients. This design and
its optimizer are local project contributions, not algorithms claimed by the
MobileNetV3 or YOLO-World papers.

### YOLO-World [6]

YOLO-World returns object detections for text categories. In this project, a
vacant slot is inferred only when classifier/detector occupancy evidence is
below the selected threshold. "Vacant" is not a YOLO-World detection class.

### TrackTrack [7]

The paper reports object-tracking metrics on standard MOT datasets. It does
not report slot occupancy. This project will report IDF1/HOTA only if
persistent human identity ground truth is acquired.

## References

[1] R. Grbic and B. Koch, "Automatic Vision-Based Parking Slot Detection and
Occupancy Classification," *Expert Systems with Applications*, vol. 225,
120147, 2023. DOI: 10.1016/j.eswa.2023.120147.

[2] Y. Yuldashev et al., "Parking Lot Occupancy Detection with Improved
MobileNetV3," *Sensors*, vol. 23, no. 17, 7642, 2023.
https://doi.org/10.3390/s23177642

[6] T. Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object
Detection," *CVPR*, 2024.
https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

[7] K. Shim et al., "Focusing on Tracks for Online Multi-Object Tracking,"
*CVPR*, 2025.
https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html

## Stage N LMOT and TrackEval provenance

The original Stage N acquisition was blocked before download and remains
preserved as historical evidence. Stage N-v2 subsequently completed the
actual LMOT validation after the user supplied the official archives. The
numeric `1..6` class map and active mark value `1` were frozen only after
released-box visual evidence and distribution checks; they were not inferred
from README order alone.

LMOT [8] is used only as a paired well-lit/low-light multi-object tracking
diagnostic. It has no parking-slot polygons or occupied/vacant interval truth
and cannot establish P3 or slot Macro F1. The evaluated truth class unifies
car, motorcycle, bus and truck. Person and bicycle truth suppress attributable
unified-D1 predictions rather than being counted as motor-vehicle false
positives.

Track metrics call the official TrackEval [9] source at commit
`12c8791b303e0a0b50f753af204249e622d0281a`. The local code prepares sequence
arrays, applies the frozen excluded-GT preprocessor, and calls TrackEval HOTA,
CLEAR, and Identity implementations. It does not reimplement HOTA or IDF1.
Two NumPy 2 scalar aliases preserve compatibility with the old official
commit without editing metric source. Detection AP is local single-class box
evaluation because TrackEval does not provide COCO-style detector AP.

Stage N-v3 is an additive offline correction to that local emitted-box
evaluation only. For every IoU threshold, predictions are confidence ordered
and matched to the highest-IoU unused GT in the same frame. The 16 saved v2
detection JSONL files and four released LMOT GT files are re-read without
loading a model or calling `model.track`. Its primary table is all-data
pooled/micro; unweighted per-sequence macro metrics remain explicitly
separate, and GT/prediction/TP/FP/FN counts are summed.

Official HOTA, DetA, AssA, IDF1, MOTA and ID-switch values are not recomputed
or rewritten in v3. They are unaffected because official TrackEval consumed
the saved tracks through its independent matching path. The corrected
detection values still describe boxes emitted by the complete tracking path
after excluded-class suppression; they are not raw detector-only metrics.

The L0--L3 comparison is described as a controlled end-to-end tracker backend
comparison. TrackTrack can recover raw pre-NMS candidates, so equal
post-NMS-input claims are prohibited. No LTrack training source or procedure
is used, and the project does not claim to reproduce LTrack.

[8] X. Wang et al., "Low-Light Multi-Object Tracking: A Benchmark," *CVPR*,
2024.
https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Low-Light_Multi-Object_Tracking_A_Benchmark_CVPR_2024_paper.html

[9] J. Luiten et al., "HOTA: A Higher Order Metric for Evaluating
Multi-Object Tracking," *International Journal of Computer Vision*, 2021.
Official evaluation code: https://github.com/JonathonLuiten/TrackEval

## Stage O low-light detector adaptation provenance

Stage O preserves the Stage N distinction between tracking and detection.
Stage N-v2/v3 consumes complete `model.track(...)` outputs; Stage O calls
`YOLO.predict(...)` without loading a tracker. Its pooled/micro detector table
sums GT, predictions, TP, FP and FN and uses a confidence ordering isolated by
sequence/frame key. The unweighted mean of sequence-level rates is reported
separately.

LMOT train supplies the only paired low-light adaptation data. Light/dark
versions of one frame share one group and cannot cross training/internal
development. LMOT validation and NDISPark night were already consumed and
are diagnostics, not untouched final tests. LMOT has no parking-slot polygon
or occupied/vacant truth, so neither raw detector AP nor Stage N tracking
metrics can establish a P3 occupancy improvement.

O1 is explicitly an OpenCV engineering baseline. O2 uses a pretrained
enhancer as preprocessing only. GLARE's official route was blocked before
build by the unavailable legacy/native CUDA toolchain; the protocol allowed
only the predeclared Retinexformer fallback. This does not reproduce GLARE's
ExDark/YOLOv3 experiment or Retinexformer's paper tables.

O3 is D1-LL: one ordinary supervised YOLOv8n fine-tune initialized from the
formal D1 checkpoint. It uses deterministic sequence-grouped, paired LMOT
train samples plus NDISPark daylight parking training to reduce
road-scene-only forgetting. There is no LTrack source, consistency loss,
hyperparameter search, LMOT-validation tuning or tracker tuning.

The Stage O candidate-selection rule was frozen before formal comparison.
O1 and Retinexformer O2 were negative results under that rule; O3 alone was
eligible and was selected as D1-LL. Only that detector-side candidate entered
the unchanged P3 interface; B1, E1b, F2 and E4 remain frozen. In the absence of new nighttime
fixed-camera slot truth, the P3 execution is an interface smoke and cannot
support an occupancy or event-accuracy claim.

[10] H. Zhou et al., "GLARE: Low Light Image Enhancement via Generative
Latent Feature based Codebook Retrieval," *ECCV*, 2024.
Official code: https://github.com/LowLevelAI/GLARE

[11] Y. Cai et al., "Retinexformer: One-stage Retinex-based Transformer for
Low-light Image Enhancement," *ICCV*, 2023.
Official code: https://github.com/caiyuanhao1998/Retinexformer

## Stage P and Stage Q-v2 parking-domain evidence

Stage P compares frozen D1 and D1-LL only on previously consumed NDISPark
partitions. Its confidence-truncated AP values are diagnostics at confidence
0.30, not standard COCO AP. The frozen retention decision is `FAIL`; it is
not altered by later data acquisition.

Stage Q-v1 remains `BLOCKED_BEFORE_DOWNLOAD`. Stage Q-v2 is additive: after
explicit user authorization it acquired the official ETSIT/UPM-GTI
`test.zip`, froze a model-independent low-light manifest, aligned the
21-value truth vectors with paper Figure 4(a), and required explicit human
polygon confirmation before loading a model.

QV2-0 and QV2-1 run the same P3 implementation and parameters. They differ
only in D1 versus D1-LL weights. Both use `YOLO.predict`, B1 one-to-one
polygon coverage, E1b/F2 detector-negative review and E4; no tracker is
loaded. The source has no reliable FPS or timestamp, so transition evidence
is reported only in selected-frame and source-frame-index units. The 2 FPS
annotated MP4 is a visualization reconstruction.

The external result is slot occupancy from one shared fixed-camera geometry,
not detector AP or tracking evidence. It cannot transfer LMOT detector
performance into an occupancy claim and cannot establish universal
generalization. D1 remains the default, D1-LL remains secondary, and Stage
P2 remains `FAIL`.

[12] L. Encío et al., "Visual Parking Occupancy Detection Using Extended
Contextual Image Information via a Multi-Branch Output ConvNeXt Network,"
*Sensors*, vol. 23, no. 6, 3329, 2023.
https://doi.org/10.3390/s23063329
