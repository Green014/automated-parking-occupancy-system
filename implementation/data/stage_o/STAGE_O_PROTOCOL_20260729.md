# Stage O low-light detector adaptation protocol

Protocol: `STAGE-O-LOW-LIGHT-DETECTOR-ADAPTATION-20260729-01`

Freeze date: 29 July 2026 (Asia/Shanghai)

Status: frozen before formal validation execution. No Stage O formal
validation inference or O3 training had been run when the final O1 tuple was
recorded.

## Scientific boundary

Stage O changes only the detector side of P3. The retained system boundary is:

```text
fixed-camera frame
  -> D1 or D1-LL detector
  -> frozen B1 polygon mapping
  -> frozen E1b detector-negative review
  -> frozen F2 asymmetric gate
  -> optional frozen E4 temporal stabilization
  -> per-slot states and events
```

LMOT supplies paired well-lit/dark sRGB images, motor-vehicle boxes and track
IDs, but no parking-slot polygon or occupied/vacant truth. Stage O can
therefore support a low-light vehicle-detection conclusion only. It cannot
produce or imply a parking-occupancy Macro F1 gain.

Stage N-v3 evaluated boxes emitted by complete `model.track(...)` paths.
Stage O uses an independent `YOLO.predict(...)` detector-only adapter.
`model.track`, ByteTrack, TrackTrack, HOTA, DetA, AssA, IDF1, MOTA and ID
switches are outside this experiment.

## Data grouping and consumption

The official LMOT train sequences are `02, 04, 06, 08, 09, 12, 16, 18, 20,
21, 26`. A fixed seed of 20260729 orders sequence IDs by
`SHA256("20260729:" + sequence_id)`. The first two, `LMOT-06` and `LMOT-26`,
form internal development. The other nine form O3 training.

Every pair is grouped by `(sequence_id, frame_number)`: the light and dark
versions of one frame cannot cross a split. O3 samples frames
`1, 11, ..., 1201`, giving 121 paired frames per sequence. NDISPark's 112
official daylight training images are mixed only into O3 training to reduce
parking-domain forgetting.

LMOT validation (`05, 13, 14, 25`) and NDISPark night validation have already
been inspected and are `consumed-development diagnostics`. NDISPark's night
test has counts but no box truth. None is described as an untouched final
box or occupancy test.

## Fixed inference and class policy

All formal detector-only arms use the Stage N frozen D1 operating point:
`imgsz=640`, `conf=0.30`, NMS IoU `0.70`, class-agnostic NMS, `max_det=300`,
no test-time augmentation, batch one and the unified D1 vehicle class.

LMOT IDs 3/4/5/6 map to car/motorcycle/bus/truck and one output class
`vehicle`. IDs 1/2 (person/bicycle) are prediction-suppression regions under
the Stage N-v2 verified rule. Thresholds, sequence membership and failure
samples cannot be changed after a validation result is viewed.

The primary table is pooled/micro: GT, predictions, TP, FP and FN are summed
and a single confidence ordering is evaluated over isolated sequence/frame
keys. Unweighted per-sequence macro rates are secondary. AP uses IoU
thresholds 0.50 through 0.95 in 0.05 steps.

## Controlled configurations

- O0 is frozen D1 on original sRGB.
- O1 is a declared OpenCV engineering baseline. It makes one brightness
  decision per sequence/source from 32 calibration frames, then holds that
  decision and Gamma/CLAHE parameters fixed. It never adapts parameters per
  frame. The small predeclared grid and deterministic tie-break are evaluated
  only on the two LMOT-train internal-development sequences; the selected
  tuple is frozen in the YAML before validation.
- O2 is a preprocessing-only diagnostic. GLARE is the only primary option.
  Its official environment calls for Python 3.8, PyTorch 1.11/CUDA 11.3 and a
  compiled deformable-convolution CUDA extension. It must use a separate
  environment. If that exact route is evidenced infeasible, the sole allowed
  fallback is the already named Retinexformer. No download or installation is
  permitted without explicit user approval.
- O3 starts from the exact formal D1 `best.pt`. It is one ordinary supervised
  fine-tune, not an LTrack reproduction and not a consistency-learning
  experiment. A one-epoch smoke must pass before one formal run of at most 20
  epochs with patience five. There is no hyperparameter search.

The O1 internal-development run retained all eight candidates. Every candidate
had zero IoU-0.50 true positives over 1,151 motor-vehicle GT boxes in the
sampled dark frames. Under the predeclared tie-break, candidate 02 was frozen:
sequence brightness threshold 45.0, Gamma 0.50, CLAHE clip limit 3.0 and
8x8 tiles. This is an adverse selection diagnostic, not evidence that the
transform works; the grid was not expanded after seeing the failure.

## Selection and P3 gate

Before formal comparison, an adapted candidate is eligible only if its pooled
dark recall improves by at least 0.02 and AP50 by at least 0.01 over O0,
AP50-95 does not decrease, and precision falls by no more than 0.05. Eligible
methods are ordered by dark AP50, recall, AP50-95, precision and FPS. If none
passes, O0/D1 remains the default.

Only a selected detector candidate may replace D1 in a P3 interface smoke.
B1, E1b, F2 and E4 remain frozen. Without new night parking video plus
per-slot truth, Stage O may produce qualitative output but cannot report an
occupancy improvement. VIRAT 0502 cannot be reused for tuning.

## Source and method provenance

- LMOT official repository: <https://github.com/xinzwang/LMOT>; dataset
  CC BY-NC 4.0 for non-commercial research, repository code MIT.
- GLARE official repository: <https://github.com/LowLevelAI/GLARE>; official
  code/model terms Apache-2.0; ECCV 2024.
- Retinexformer official repository:
  <https://github.com/caiyuanhao1998/Retinexformer>; repository code MIT;
  ICCV 2023.
- Gamma/CLAHE are OpenCV operations and are not presented as a paper method.

The exact source parts, hashes, split IDs, current consumption state and D1
binding are in `STAGE_O_DATA_MANIFEST_20260729.yaml`.
