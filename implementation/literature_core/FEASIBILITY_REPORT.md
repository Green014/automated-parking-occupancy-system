# Phase 0 Feasibility Report

Audit date: 25 July 2026 (Asia/Shanghai)

## Decision

The minimum literature-core workflow is feasible on the available RTX 3060
Laptop GPU without modifying the existing baseline environment:

```text
slot polygon -> OpenCV perspective-normalized patch
             -> adapted pretrained MobileNetV3-Small -> P_cls
YOLO-World prompted object boxes -> polygon mapping -> P_det
P_cls + P_det -> weighted fusion -> optional temporal hysteresis
```

The implementation lives only in `implementation/literature_core/`. Existing
baseline code, model files, reports, CSV files, and output directories are not
modified.

## Coursework audit

The four rendered pages of the supplied coursework brief and
`literature_review/coursework_requirements_extracted.txt` were checked. Part II
requires Python and OpenCV, an improvement or combination of known algorithms,
task-appropriate evaluation, explicit differences from existing systems, and
analysis of negative results. The proposed workflow meets the "combination"
route. It must not be described as a new detector or tracker.

## Baseline audit

- Existing workflow: YOLOv8 -> optional ByteTrack -> manually defined convex
  slot polygons -> centre/overlap mapping -> confidence-aware EMA/hysteresis.
- Existing tests on 25 July 2026: **23 passed** with
  `.\.venv\Scripts\python.exe -m pytest -q`.
- Existing full reports and outputs were found for PKLot, NDISPark, Grand
  Bassin, and the licensed-data smoke test.
- Git has no commits and all project content is untracked. All pre-existing
  files are therefore treated as user-owned state.
- The baseline reports a negative static smoke result for its temporal
  pipeline and a positive-only Grand Bassin stability result. These results
  are retained, not rewritten.

## Hardware and software

| Component | Audited value | Consequence |
|---|---|---|
| OS | Windows 11 | PowerShell reproduction commands are supplied |
| Python | 3.12.13 | Supported by the current project |
| OpenCV | 4.13.0 | Patch warping, polygon geometry, video I/O |
| NumPy | 2.4.4 | Compatible with the existing environment |
| PyTorch | 2.13.0+cu130 | CUDA inference/training available |
| torchvision | 0.28.0+cu130 | MobileNetV3-Small implementation available |
| Ultralytics | 8.4.104 | `YOLOWorld` and `set_classes()` available |
| GPU | RTX 3060 Laptop, 6 GiB | batch 16-32, AMP, frozen backbone first |
| NVIDIA driver | 610.62 | CUDA 13.0 PyTorch build works |

Direct Ultralytics imports try to create a user-level configuration directory
in this managed environment. Both workflows avoid that by setting
`YOLO_CONFIG_DIR` to a project-local ignored directory.

## Literature verification

### APSD-OC [1]

The local 39-page paper was checked rather than inferred from its title. The
paper applies YOLOv5 to a sequence, transforms vehicle-box centres into a
bird's-eye view using a CNN-estimated homography, clusters the centres with
DBSCAN, filters cluster centres using dispersion, maps them back to the camera
view, and classifies detected slot crops with an ImageNet-pretrained ResNet34.
No official implementation was identified in the local records or the checked
paper indexes. Exact reproduction would additionally require the paper's
homography-regression model and enough observations of every slot. APSD-OC is
therefore provenance/design support and an optional future extension, not part
of the minimum implementation.

### Improved MobileNetV3 [2]

The local literature metadata states that the PDF download was blocked. The
publisher/PMC full text was therefore checked. The paper:

- processes manually located parking-space patches at 224 x 224;
- replaces shallow ReLU6 behaviour with LeakyReLU6;
- replaces MobileNetV3 SE attention with CBAM;
- replaces depth-wise separable convolutions with blueprint separable
  convolutions;
- reports Adam, initial learning rate 0.0001, weight decay 0.0005, momentum
  0.99, batch size 64, and 500 epochs;
- trains/evaluates on PKLot and CNRPark-EXT.

No confirmed official implementation or weights were found. The deliverable
therefore uses a standard torchvision MobileNetV3-Small initialized from
ImageNet and replaces its head for two-class transfer learning. This is an
**adapted implementation**, not an exact reproduction. It intentionally
retains the standard SE/depth-wise blocks and does not quote the paper's
reported numbers as this implementation's results.

### YOLO-World [6]

The local CVPR paper and official CVF page were checked. YOLO-World combines a
YOLO detector, text encoder, RepVL-PAN, and region-text contrastive learning.
It outputs prompted object boxes and scores. It does **not** directly output a
vacant parking space. The installed Ultralytics adapter supports a pretrained
YOLO-World checkpoint and `set_classes(["car", "truck", "bus",
"motorcycle"])`; polygon mapping remains a separate project stage.

### TrackTrack [7]

The local CVPR paper was checked. Its contributions are
Track-Perspective-Based Association and Track-Aware Initialization, evaluated
on MOT17, MOT20, and DanceTrack. The paper does not establish parking
occupancy improvement. Code availability is not sufficiently clear in the
local audit, and the current data have no persistent identity ground truth.
TrackTrack is therefore deferred; the existing hysteresis is retained as the
first temporal comparison.

### Low-light references [9], [10]

LMOT and AODRaw support robustness-test design, but they are not parking-slot
methods. NDISPark already provides an audited night-domain detector test. No
RAW or low-light tracking module is required for the minimum closure.

## Data and truth suitability

| Source | Local state | Truth status | Allowed claim |
|---|---|---|---|
| PKLot | 27 images, 1,505 known slot labels | Dataset slot labels; 7 unknown excluded | Slot classification/fusion pilot |
| NDISPark | 30 validation images, 725 boxes | Manual box labels | Detector metrics/night analysis |
| Grand Bassin | 793-frame selected sequence | Machine box preannotations; 7 manually checked occupied slots only | Positive-only stability/false-free analysis |

For the literature-core pilot, cameras are separated:

- train: PUCPR (900 slot instances);
- development: UFPR04 (252 raw / 247 known slot instances);
- test: UFPR05 (360 raw / 358 known slot instances).

Seven `unknown` labels are excluded. The split prevents camera/date-adjacent
leakage, but the 27 images were previously used in the baseline development
study, so the new test is called a **pilot camera holdout**, not a globally
untouched final test.

## Dependency and resource risks

1. No package installation is needed for the first implementation. Installing
   an older MMDetection/MMCV YOLO-World stack into the baseline environment
   would be risky and is not attempted.
2. At audit start, the pretrained MobileNetV3 and YOLO-World weight files
   were not cached. They were subsequently downloaded from the official
   torchvision/Ultralytics sources without changing package versions. The
   exact SHA-256 values are recorded in `METHOD_PROVENANCE.md`.
3. The paper's MobileNetV3 batch size 64 may exceed the practical margin of a
   6 GiB laptop GPU once augmentation is included. The project default is
   batch 32 with AMP and a frozen backbone.
4. PKLot images are roughly five minutes apart and must not be used for
   tracking, flicker, or transition-latency claims.
5. This Phase 0 audit originally blocked E4/E5 for lack of mixed
   occupied/vacant truth. A later VIRAT search resolved the minimum access
   gate with one verified departure in each of two distinct scenes. The
   resulting frozen E4/E5 case study was negative and remains too small for a
   general tracking claim; details are in `RESULTS.md`.

## Phase 0 conclusion

The decision was to proceed with the adapted MobileNetV3, YOLO-World adapter,
OpenCV slot warp, polygon mapping, interpretable fusion, corrected temporal
metrics, tests, and pilot ablation. That minimum closure has now been
implemented and executed; results are in `RESULTS.md`. Exact Improved
MobileNetV3, TrackTrack, and APSD-OC reproduction remain deferred. E5 is a
simpler local ByteTrack-based gate, not a TrackTrack reproduction, and Fusion
V2 remains closed because E5 failed its development reliability gate.
