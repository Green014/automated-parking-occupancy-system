# Automated Parking Lot Occupancy and Tracking System

This Part II package now contains both the closed geometry baselines and the
completed final P3 integrated workflow. The baselines convert vehicle
detections into parking-slot states:

```text
OpenCV video -> YOLOv8 vehicles -> slot polygon mapping
             -> optional legacy ByteTrack/hysteresis variant
             -> video + occupancy/events/timing logs
```

YOLO detects vehicles. It does not directly detect vacant spaces. The
centre-point or polygon-overlap mapper is the explicit step that assigns a
vehicle detection to a predefined parking slot.

## Closed baseline names

- **B0**: YOLOv8 + bounding-box centre inside the slot polygon.
- **B1**: YOLOv8 + confidence-weighted slot polygon coverage.
- **E0**: historical static external YOLOv8 coverage baseline; frozen and not
  a new runnable tuning condition.
- **T0**: raw YOLOv8 temporal comparator without dwell or hysteresis.

Exact detector, `conf`, `imgsz`, mapping, and data-role settings are registered
in `configs/baseline_methods.yaml` and explained in
`BASELINE_CLOSURE.md`. Historical artifact names are unchanged.

The legacy `--experiment proposed` path remains available to reproduce the
engineering combination of YOLOv8, ByteTrack IDs, polygon coverage, and
hysteresis. It is not the converged Part II main workflow.

The completed P3 workflow has its own `parking-run-integrated` entry point. It
uses D1 YOLOv8 as primary evidence, B1 geometry assignment, E1b
MobileNetV3-CBAM review of detector-negative slots, asymmetric
uncertainty-gated fusion, optional E4 stabilization, and optional ByteTrack or
TrackTrack. It accepts arbitrary video/slot maps and does not require truth
for inference. The historical `--experiment proposed` path remains a legacy
engineering path and is not renamed P3. YOLO-World remains a
detector-replacement experiment rather than a parallel vote.

```powershell
.\.venv\Scripts\parking-run-integrated.exe `
  --input <video> `
  --slots <slot-map.json> `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --tracker none `
  --output-dir <new-output-directory>
```

Add `--truth <slot-state.csv>` only when evaluation truth exists. Use
`--tracker bytetrack|tracktrack`, `--no-temporal`, or `--config <yaml>` for
explicit runtime variants. Defaults are recorded in
`configs/p3_integrated_runtime_defaults_20260729.yaml` and come from the
frozen Stage L configuration.

The final result boundary is:

- Stage K is the frozen static P0/P1/P2 test.
- P3 on the Stage K images is a retrospective extension, not a new untouched
  test.
- E1b-only Macro F1 (0.992226) is higher than P3 (0.987061).
- P3's main operational advantage is its lower false-free rate (0.004632).
- The continuous VIRAT case is a geometry-association negative result: the
  neighbouring vehicle remained mapped to the departed vehicle's slot.
- LMOT supports only paired low-light tracking robustness analysis; it has no
  parking-slot occupancy truth.

Current development reports include:

- `metadata/PKLOT_DEVELOPMENT_REPORT.md`;
- `metadata/GRAND_BASSIN_TEMPORAL_REPORT.md`;
- `metadata/FINETUNING_GATE_ASSESSMENT.md`.

The Part I data-role audit is:

- `data/PART1_DATASET_ALIGNMENT.md`;
- `data/part1_dataset_alignment.yaml`.

It distinguishes vehicle boxes/counts from slot polygons/states and records
official sources, licenses, local holdings, leakage risks, task fit, and the
A-H data-role decisions. The user selected the local NDISPark-only backup
route on 2026-07-27; the Zenodo API confirms NDISPark as ODC-By/open. The
PKLot download was stopped with its incomplete resumable file preserved, and
CARPK is deferred. Stage C is now frozen as
`DPROTO-NDISPARK-ONLY-20260727-01`:

- `configs/ndispark_only_dataset_frozen_20260727.yaml`;
- `data/NDISPARK_ONLY_DATASET_CARD.md`;
- `data/manifests/ndispark_only_20260727/`.

The manifests contain 112 train, 30 consumed-development validation, and 117
count-only test images. Stage D preprocessing and the Stage E detector
comparison protocol have passed. The backup route has no new detector-mAP
test. A separate PKLot data gate later supplied the untouched Stage K
slot-occupancy test.

Stage D is recorded in:

- `data/NDISPARK_PREPROCESSING_REPORT.md`;
- `data/preprocessing/ndispark_only_20260727.yaml`.

It verified all 259 frozen image hashes and converted all 3,302 train/validation
boxes to one-class YOLO labels without clipping, exclusion, or duplicate
removal. Ultralytics loaded 112 train and 30 validation images with class 0
only. No model or prediction was run.

Stage E is recorded in:

- `configs/detector_comparison_frozen_20260727.yaml`;
- `data/DETECTOR_COMPARISON_PROTOCOL.md`;
- `data/comparisons/detector_comparison_preflight_20260727.yaml`.

It freezes D0/D1/D2, common 640-pixel inference and box-metric conditions,
class canonicalization, runtime fields, and non-overwriting outputs. The real
historical Stage E preflight verified D0/D2 but stopped before model loading
because D1 had not yet been trained. That expected Stage F gate artifact
remains unchanged; Stage I later supplied the frozen formal D1 checkpoint.

Reproduce into a new ignored output directory:

```powershell
$env:PARKING_DATA_ROOT = "D:\datasets\ndispark\extracted"
$env:PARKING_OUTPUT_ROOT = "D:\parking_generated\ndispark_only_20260727"

python scripts\prepare_ndispark.py `
  --protocol configs\ndispark_only_dataset_frozen_20260727.yaml `
  --source-root $env:PARKING_DATA_ROOT `
  --output-root $env:PARKING_OUTPUT_ROOT
```

The output root must not already exist.

Run the Stage E preflight without loading a model:

```powershell
$env:PARKING_PREPARED_DATA = "D:\parking_generated\ndispark_only_20260727_v1\dataset.yaml"
$env:PARKING_D0_WEIGHTS = "D:\models\yolov8n.pt"
$env:PARKING_D2_WEIGHTS = "D:\models\yolov8s-worldv2.pt"

python scripts\run_detector_comparison.py `
  --config configs\detector_comparison_frozen_20260727.yaml `
  --data $env:PARKING_PREPARED_DATA `
  --d0-weights $env:PARKING_D0_WEIGHTS `
  --d2-weights $env:PARKING_D2_WEIGHTS `
  --preflight-output outputs\detector_comparison_preflight_<new-id>.json
```

D1 is not supplied here, so the report must say
`blocked_before_output_creation` and `predictions_run=false`. Stage F's local
pretrained D1 smoke run passed, and Stage H has now produced the formal D1
checkpoint. A new Stage I comparison run must supply its recorded SHA-256;
the historical preflight artifact remains unchanged.

Stage F evidence is in:

- `data/D1_SMOKE_REPORT.md`;
- `data/training/d1_ndispark_smoke_20260727.yaml`;
- ignored `outputs/d1_ndispark_smoke_20260727_v3/`.

The successful run completed three epochs at batch 4 and image size 640 on the
local RTX 3060. It recorded finite changing losses, validation over all 30
development images, updated checkpoints, 767,557,632 bytes peak Torch reserved
VRAM, and no OOM/NaN or material dataloader wait. Two earlier zero-epoch
runner failures remain preserved in separate v1/v2 output directories. The
smoke checkpoint is not formal D1; Stage G must decide the local formal-run
configuration first.

Reproduce only into a new ignored directory:

```powershell
python scripts\run_d1_smoke.py `
  --dataset-protocol configs\ndispark_only_dataset_frozen_20260727.yaml `
  --comparison-protocol configs\detector_comparison_frozen_20260727.yaml `
  --data $env:PARKING_PREPARED_DATA `
  --initial-weights $env:PARKING_D0_WEIGHTS `
  --output-dir outputs\d1_ndispark_smoke_<new-id> `
  --device 0 --workers 2 --execute
```

Stage G is now complete:

- `data/GPU_DECISION_REPORT.md`;
- `data/training/d1_gpu_decision_20260727.yaml`;
- `configs/d1_ndispark_formal_frozen_20260727.yaml`.

At 640/batch 4, Stage F measured 0.715 GiB peak Torch reserved memory on the
6 GiB RTX 3060 Laptop GPU. The frozen 50-epoch formal run is estimated at
2.43 minutes centrally, 4.35 minutes conservatively, and 8.52 minutes under a
two-times-slowest-epoch stress bound. These are explicit extrapolations, not a
measured 50-epoch runtime. Batch 4 is retained because it is the largest
allowed batch directly executed; the nominal batch of 64 gives 16 post-warm-up
accumulation steps.

Local formal training is authorized as
`D1-NDISPARK-FT-20260727-01`, starting again from the frozen COCO-pretrained
`yolov8n.pt`, never from the smoke checkpoint. Paid/remote GPU use remains
prohibited, rental duration is zero, and an A100 is unnecessary. The Stage G
calculator itself performs no training or prediction:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\analyze_gpu_decision.py `
  --smoke-summary outputs\d1_ndispark_smoke_20260727_v3\smoke_summary.json `
  --output outputs\gpu_decision_<new-id>.json
```

Stage H is complete:

- `data/D1_FORMAL_TRAINING_REPORT.md`;
- `data/training/d1_formal_training_20260727.yaml`;
- ignored `outputs/d1_ndispark_formal_20260727_v1/`.

The single fixed-seed run started from the original COCO-pretrained YOLOv8n,
completed 47 of 50 epochs, and stopped with patience 10. Epoch 37 produced the
selected `best.pt`:

- SHA-256
  `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`;
- development Precision 0.93708, Recall 0.88339, mAP@0.5 0.94478, and
  mAP@0.5:0.95 0.67556.

These are consumed-development diagnostics, not a final detector test. The
initial runner retained a post-training
resource-callback audit failure; it was recovered offline without retraining,
and all 14 recorded artifacts passed size/SHA-256 verification.

Stage I is complete:

- `configs/stage_i_count_test_frozen_20260727.yaml`;
- `data/DETECTOR_EVALUATION_REPORT.md`;
- `data/comparisons/stage_i_detector_evaluation_20260727.yaml`;
- ignored `outputs/detector_comparison_stage_i_20260727_v1/`;
- ignored `outputs/detector_count_test_stage_i_20260727_v2/`;
- ignored `outputs/detector_qualitative_stage_i_20260727_v1/`.

On the 30-image/725-box consumed development validation, D1 led D0 and D2
with Precision 0.88153, Recall 0.84160, mAP@0.5 0.89910, and
mAP@0.5:0.95 0.64969 at 37.77 FPS. D1 was selected before test access.
A shared count threshold of 0.10 was then selected from development data and
frozen for all models.

The 117-image official count-only test retained a negative result: D2 had the
lowest MAE (2.58974), followed by D0 (2.99145) and D1 (3.46154). The detector
selection and threshold were not changed afterward. No box metrics are
reported on this split because it has count truth only. All 24 selected Stage
I artifacts passed size/SHA-256 verification. Stage J reconnects D0/D1/D2 to
identical B1 mapping; Stage K later evaluates those frozen pipelines on an
independent PKLot slot-occupancy subset.

Stage I v1 is now explicitly historical and immutable. The corrected Stage
I-v2 evaluation is documented in
`data/STAGE_I_V2_CORRECTED_EVALUATION_REPORT.md`. It applies class-agnostic NMS
before source classes are mapped to `vehicle`, reports common-threshold
diagnostics separately from per-model development calibration, and compares
`max_det=300` with 1000 on development only. The ranking remained
D1 > D2 > D0, so 300 was retained. The development-selected confidence
thresholds are D0 0.10, D1 0.30, and D2 0.10.

One settings-frozen run on the already consumed count split is reported only
as `consumed_test_posthoc_sensitivity`. Under the per-model development
thresholds, MAE was D0 2.96581, D1 1.52991, and D2 2.58974. This result did not
reselect the detector or change any setting; it is not a new untouched test
and count MAE is not slot-occupancy accuracy. The v2 registry contains 44
critical source/output artifacts, all of which passed SHA-256 and size
verification.

Verify Stage I-v2 without rerunning predictions:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\verify_stage_i_v2_artifacts.py `
  --record data\comparisons\stage_i_v2_corrected_evaluation_20260727.yaml `
  --source-root . `
  --external-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --output $env:PARKING_NEW_VERIFICATION_OUTPUT
```

Stage J is complete on consumed PKLot development data:

- `configs/stage_j_p0_p1_p2_pklot_development_frozen_20260727.yaml`;
- `data/STAGE_J_K_OCCUPANCY_REPORT.md`;
- `data/comparisons/stage_j_p0_p1_p2_development_20260727.yaml`;
- ignored `outputs/P0_P1_P2_stage_j_20260727_v1/`.

P0, P1 and P2 use D0, D1 and D2 respectively with identical B1 polygon
coverage (0.40), one-to-one assignment, `agnostic_nms=true`, `max_det=300`
and no temporal filtering. On 1,505 known slot labels, consumed-development
Macro F1 was 0.768040, 0.825723 and 0.735168 respectively. P1 led this
integration check but still had a 0.328930 false-free rate. No result was used
to reselect a detector or change a setting.

The 27 ordered images in each `annotated.mp4` are a non-contiguous
development montage, not a time series. `events.csv` is therefore
intentionally header-only. The read-only Stage J grouped analysis further
shows that P1's pooled gain is camera-dependent: P1 versus P0 is 6 wins, 2
ties and 19 losses by image, and its paired 95% interval includes zero. The
Stage J result must not be described as an untouched test.

Stage K is complete:

- `configs/stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml`;
- `data/comparisons/stage_k_slot_occupancy_data_gate_20260728_v2.yaml`;
- `data/comparisons/stage_k_p0_p1_p2_test_20260727.yaml`;
- `configs/stage_k_posthoc_stratified_analysis_frozen_20260728.yaml`;
- `data/comparisons/stage_k_posthoc_stratified_analysis_20260728.yaml`;
- `data/STAGE_K_FINAL_REPORT.md`;
- ignored `outputs/P0_P1_P2_stage_k_20260727_v1/`;
- ignored `outputs/stage_k_posthoc_strata_20260728_v1/`.

The additive data-gate v2 preserves the earlier blocked local-inventory audit
but supersedes its decision after 90 complete, previously unused PKLot
JPG/XML pairs were recovered. Their 90 image hashes have zero overlap with
Stage J. The frozen test contains 5,034 known slot labels and six excluded
unknown labels.

Pooled Macro F1 is 0.785612 for P0, 0.808398 for P1 and 0.796548 for P2.
P1 has the highest pooled value, but camera-macro F1 is 0.841099, 0.824835
and 0.849168. P1 versus P0 is 17 wins, 11 ties and 62 losses by image, with
a mean paired difference of -0.040322 and 95% interval
[-0.068804, -0.009457]. D1 remains the pre-test selected detector in the
provenance record; Stage K did not trigger detector reselection or tuning.

Date and weather outputs are descriptive only. Each camera contributes one
date and cloudy weather occurs only for UFPR04, so those factors are
confounded with camera.

Preflight or execute only with externally supplied data/weight paths and a
new output directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\run_stage_j_occupancy.py `
  --config configs\stage_j_p0_p1_p2_pklot_development_frozen_20260727.yaml `
  --source-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --p0-weights $env:PARKING_D0_WEIGHTS `
  --p1-weights $env:PARKING_D1_WEIGHTS `
  --p2-weights $env:PARKING_D2_WEIGHTS
```

Add `--execute --output-dir <new-directory>` only after preflight passes.
Verify the frozen run without model loading:

```powershell
python scripts\verify_stage_j_artifacts.py `
  --record data\comparisons\stage_j_p0_p1_p2_development_20260727.yaml `
  --source-root . `
  --external-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --output $env:PARKING_NEW_VERIFICATION_OUTPUT
```

Verify the completed Stage K records without model loading:

```powershell
python scripts\verify_stage_k_artifacts.py `
  --record data\comparisons\stage_k_p0_p1_p2_test_20260727.yaml `
  --source-root . `
  --external-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --output <new-stage-k-verification.json>

python scripts\verify_stage_k_strata_artifacts.py `
  --record data\comparisons\stage_k_posthoc_stratified_analysis_20260728.yaml `
  --source-root . `
  --external-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --output <new-strata-verification.json>

python scripts\verify_stage_k_data_gate_v2.py `
  --record data\comparisons\stage_k_slot_occupancy_data_gate_20260728_v2.yaml `
  --source-root . `
  --external-root $env:PARKING_IMPLEMENTATION_ARTIFACT_ROOT `
  --output <new-gate-verification.json>
```

Reproduce the frozen count workflow only with externally supplied paths and a
new output directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\run_stage_i_evaluation.py preflight-count `
  --config configs\stage_i_count_test_frozen_20260727.yaml `
  --selection-record $env:PARKING_STAGE_I_SELECTION `
  --comparison $env:PARKING_STAGE_I_COMPARISON `
  --comparison-config configs\detector_comparison_frozen_20260727.yaml `
  --truth-manifest data\manifests\ndispark_only_20260727\ndispark_test_frozen_20260727.csv `
  --test-images $env:PARKING_NDISPARK_TEST_IMAGES `
  --d0-weights $env:PARKING_D0_WEIGHTS `
  --d1-weights $env:PARKING_D1_WEIGHTS `
  --d2-weights $env:PARKING_D2_WEIGHTS
```

## Setup

PowerShell:

```powershell
cd .\implementation
.\.venv\Scripts\python.exe -m pip install `
  torch==2.13.0+cu130 torchvision==0.28.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The CUDA command above matches the checked RTX 3060/driver environment. On a
different machine, select the appropriate official PyTorch build instead.
For the integrated P3 entry, install the sibling package and integrated
extra; the broader research extra additionally expresses Shapely and the
frozen TrackEval source:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\literature_core
.\.venv\Scripts\python.exe -m pip install -e ".[integrated]"
# Use ".[research]" only for the official/research evaluation paths.
```

The first YOLO command downloads the selected pretrained weights if they are
not already cached. Start with `yolov8n.pt`; use `yolov8s.pt` as a controlled
baseline variant, not as a separately tuned experiment.

## 1. Annotate parking slots

```powershell
.\.venv\Scripts\parking-annotate.exe `
  --input data\raw\video\camera01.mp4 `
  --frame 0 `
  --output data\annotations\camera01_slots.json
```

Controls:

- left click: add a polygon point;
- right click: undo the current point;
- Enter: commit a convex polygon with at least three points;
- Backspace: delete the most recently committed slot;
- `S`: save;
- `Q`/Escape: quit without saving.

## 2. Run a canonical baseline

```powershell
.\.venv\Scripts\parking-run.exe `
  --input data\raw\video\camera01.mp4 `
  --slots data\annotations\camera01_slots.json `
  --method B0 `
  --device auto `
  --output-dir outputs\camera01_b0
```

The output directory contains:

- `annotated.mp4`;
- `occupancy.csv` with one row per frame/slot;
- `events.csv` with slot transitions;
- `summary.json` with the canonical method name/data role, registry path,
  detector and mapping configuration, checksums, environment, FPS, and
  latency.

Run B1 or T0 by changing `--method`. Canonical method runs reject overrides of
weights, `conf`, `imgsz`, and coverage threshold so that an altered
configuration cannot silently retain a canonical name.

The old `--experiment b0|b1|proposed|t0` interface is retained for custom and
historical engineering runs. The `proposed` command enables ByteTrack with
the selected tracker configuration. Tracking IDs are attached to matched
boxes while unmatched YOLO detections are retained, so ByteTrack does not
become a second detector gate.

The current low-confidence development command is:

```powershell
.\.venv\Scripts\parking-run.exe `
  --input data\raw\grand_bassin\grand_bassin_aerial_development.mp4 `
  --slots data\annotations\grand_bassin_bus_stability_slots.json `
  --output-dir outputs\grand_bassin_proposed_retain `
  --experiment proposed `
  --weights yolov8n.pt `
  --tracker-config configs\bytetrack_parking_lowconf.yaml `
  --conf 0.025 --imgsz 1280 --device 0 `
  --overlap-threshold 0.40 `
  --rise-alpha 0.60 --fall-alpha 0.15 `
  --occupied-threshold 0.020 --vacant-threshold 0.005
```

Those temporal thresholds are provisional because the current continuous
stability subset contains only occupied slots. Revalidate them on a mixed
occupied/vacant video before the final test.

## 3. Evaluate slot states

Ground truth and prediction CSVs use this key:

```text
video_id,frame_index,timestamp_s,slot_id,state
```

Predictions may also include `evidence`, `raw_state`, `filtered_score`, and
`track_id`.

```powershell
.\.venv\Scripts\parking-evaluate.exe `
  --ground-truth data\annotations\camera01_truth.csv `
  --predictions outputs\camera01_b0\occupancy.csv `
  --fps 25 `
  --warmup-frames 3 `
  --output-dir outputs\camera01_b0\evaluation
```

This writes `metrics.json`, `confusion_matrix.png`, `pr_curve.png`, and an
`errors.csv` failure-case index. `--warmup-frames` affects only the temporal
flicker metric; it does not remove frames from classification scores.
Detection mAP is a separate detector-level evaluation and must be computed only
on data with vehicle-box ground truth.

For a box-labelled dataset converted to Ultralytics format:

```powershell
.\.venv\Scripts\parking-evaluate-detection.exe `
  --data data\processed\ndispark\dataset.yaml `
  --weights yolov8n.pt `
  --imgsz 1280 `
  --device 0 `
  --split val `
  --classes 2 `
  --output-dir outputs\ndispark_val_yolov8n_final_1280
```

This produces detector precision/recall, mAP@0.5, mAP@0.5:0.95, per-class AP,
timings, confusion/PR plots, and a JSON report. It is deliberately separate
from the slot evaluator. NDISPark's test split has counting labels only, so
the reported box metrics use its 30-image, 725-box validation split.

## 4. Reproduce the PKLot development experiment

After the licensed images listed in
`data/splits/pklot_development.csv` are present:

```powershell
.\.venv\Scripts\parking-evaluate-pklot.exe `
  --manifest data\splits\pklot_development.csv `
  --annotations data\annotations\pklot_development_samples.jsonl `
  --weights yolov8n.pt `
  --conf 0.025 --imgsz 1280 --device 0 `
  --overlap-threshold 0.40 `
  --output-dir outputs\pklot_dev_yolov8n_c0025_overlap040
```

## 5. Tests

From the repository root, run both source-tree suites against the current
worktree:

```powershell
python -m pytest -q
```

The root `pytest.ini` prepends both `src` directories and `implementation/`,
so an older editable installation cannot shadow current modules.

The implementation suite can also be run alone:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The independent `literature_core` suite must also pass after cross-package
evaluation changes:

```powershell
cd .\literature_core
..\.venv\Scripts\python.exe -m pytest
```

## 6. Data and scope

See `data/PART1_DATASET_ALIGNMENT.md` for the current Part I decision and
`data/DATASET_MANIFEST.md` for the wider Part II source inventory. Raw data,
generated outputs, and model weights are ignored by Git. Source paths are
provided through `--source-root` or `PARKING_DATA_ROOT`; generated dataset
paths use `--output-root`. Video/camera/date/source groups remain intact, and
adjacent frames are never randomly divided between train, validation, and
test.

`data/finetune/grand_bassin_vehicle_preannotations.json` is not ground truth.
It must be manually corrected according to
`data/finetune/ANNOTATION_PLAN.md` before any YOLO fine-tuning run.

CNR-EXT has already been consumed and cannot be used for new parameter
selection. VIRAT 0503 has been viewed and cannot be presented as a new
method's untouched holdout. NDISPark validation is development validation,
and its test split has count truth only. PKLot/CNRPark slot polygons are not
vehicle boxes.

## 7. Stage L integrated Part I workflow

Stage L adds a separate P3 experiment without changing the frozen P0/P1/P2
results:

```text
D1 YOLOv8n -> B1 polygon mapping -> E1b detector-negative review
  -> asymmetric gate -> temporal hysteresis -> optional ByteTrack support
```

On the 90 previously consumed Stage K images, P3 reached 0.987061 Macro F1
and reduced false-free rate to 0.004632. A fixed-threshold ablation found that
E1b alone had higher balanced Macro F1 (0.992226), while P3 had higher
occupied recall and lower false-free rate. The complete continuous workflow
then missed the VIRAT 0502 departure because an oblique slot polygon remained
associated with an adjacent stationary vehicle. The run, failure diagnosis
and claim boundaries are in `data/STAGE_L_INTEGRATED_WORKFLOW_REPORT.md`.

## 8. Stage M open-source and TrackTrack paths

Stage M adds two non-overwriting runnable paths while preserving every Stage L
artifact:

- `OS0-Controlled` calls the official
  `ultralytics.solutions.ParkingManagement` object with D1, the shared
  polygons/settings and explicit TrackTrack. Official centre-point occupancy
  is retained; the local adapter adds per-slot logs and the common seven-file
  output contract.
- `T0` is the frozen P3 gate without time or tracking, `T1` adds E4, `T2`
  adds ByteTrack and `T3` adds TrackTrack through
  `model.track(..., persist=True, tracker=...)`.

Check the current task-specific data gates and run a new smoke directory:

```powershell
.\.venv\Scripts\python.exe scripts\check_stage_m_data_gates.py
.\.venv\Scripts\python.exe scripts\run_stage_m_smoke.py `
  --output-root outputs\stage_m_smoke_<new-id> --device 0
```

The registered `stage_m_smoke_20260728_v2` run completed OS0 continuous,
OS0 per-image reset and T0--T3, with 7/7 non-empty files per run. It repeats a
consumed development image and has no truth, so its metrics correctly remain
`not_computed_no_truth`; it is not accuracy, transition or real-time
evidence.

Formal execution is code-complete but blocked until a licensed fixed-camera
development/test pair has distinct, human-reviewed polygon and transition
truth bundles frozen before the one test run. AODRaw is detector-only and
currently licence-blocked; LMOT is at most a validation tracking diagnostic
and cannot support parking-slot occupancy claims. See
`data/STAGE_M_OPEN_SOURCE_TRACKING_ROBUSTNESS_REPORT.md` and
`data/stage_m/STAGE_M_DATA_GATES_20260728.yaml`.

## 9. Original Stage N LMOT diagnostic

Stage N freezes an independent LMOT validation diagnostic without changing
Stage L or Stage M. It adds a strict nine-column parser, paired
dark/light-sequence audit, evidence-gated numeric class mapper, unified
`motor_vehicle` truth (`car`, `motorcycle`, `bus`, `truck`), excluded-object
prediction suppression, and an official TrackEval adapter for HOTA, CLEAR,
and Identity.

The original Stage N attempt was blocked before download because the official
single Baidu share did not publish package size or validation/sRGB-only
selectivity, and the official README did not state the numeric class-ID map
or ignore-value semantics. That blocked record is preserved as historical
evidence. Stage N-v2 below subsequently completed the actual validation after
manual data acquisition and frozen visual class-map evidence.

Run only the synthetic adapter verification:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_n_synthetic_trackeval.py `
  --output-root outputs\stage_n_lmot_synthetic_adapter_<new-id>
.\.venv\Scripts\python.exe scripts\verify_stage_n_output.py `
  --output-root outputs\stage_n_lmot_synthetic_adapter_<new-id>
```

The registered v2 output is synthetic-only; its perfect, ID-switch, missed,
and false-positive fixtures verify official metric plumbing and make no LMOT
or parking claim. See `data/STAGE_N_LMOT_TRACKING_REPORT.md` and
`data/stage_n/STAGE_N_LMOT_DATA_GATE_20260728.yaml`.

## 10. Stage N-v2 actual LMOT validation

Stage N-v2 is an additive completion of the previously blocked diagnostic.
The official annotation tar and 13 RGB split-tar parts were supplied manually.
The extractor streamed the transport archives and wrote only LMOT-05,
LMOT-13, LMOT-14 and LMOT-25 validation RGB data. The resulting 9,688 files
(14,013,374,410 bytes) have a frozen per-file SHA-256 manifest.

The class map was frozen only after released-box visual inspection:
`1=person`, `2=bicycle`, `3=car`, `4=motorcycle`, `5=bus`, and `6=truck`.
All released validation rows carry mark value `1`, which is treated as active
MOT truth. The actual L0--L3 run processed 19,360 method-frames with no
LMOT-driven parameter changes.

| Method | Input / tracker | HOTA | IDF1 | MOTA | IDSW | FPS |
|---|---|---:|---:|---:|---:|---:|
| L0 | well-lit / ByteTrack | 26.613 | 34.147 | 22.039 | 481 | 75.20 |
| L1 | well-lit / TrackTrack | 22.940 | 29.840 | 20.635 | 147 | 27.17 |
| L2 | low-light / ByteTrack | 6.028 | 4.442 | 2.114 | 39 | 76.03 |
| L3 | low-light / TrackTrack | 3.454 | 1.552 | 0.814 | 7 | 24.22 |

The low-light drop is primarily a detection-coverage failure. TrackTrack
reduces ID switches but emits a much smaller, more conservative track set, so
it does not improve aggregate HOTA under the frozen settings. These are
LMOT motor-vehicle tracking diagnostics, not parking-slot occupancy results.
See `data/STAGE_N_V2_LMOT_TRACKING_REPORT.md` and
`data/stage_n_v2/STAGE_N_V2_ARTIFACT_REGISTRY_20260729.yaml`.

## 11. Stage N-v3 emitted-box correction

Stage N-v3 corrects only the local emitted-box IoU matcher and aggregate
definition. It reads the 16 saved Stage N-v2 detection JSONL files and four
released LMOT GT files offline; it does not load D1, call `model.track`, run
TrackEval, or alter any v2 artifact. At each IoU threshold, a prediction now
selects the highest-IoU unused GT rather than failing immediately when its
highest-overlap GT has already been used.

The primary v3 table is an all-data pooled/micro aggregate. Unweighted
per-sequence macro values are retained separately, and all box/count/TP/FP/FN
fields are sums. HOTA, DetA, AssA, IDF1, MOTA, and ID switches remain the
official v2 TrackEval results because they use independent TrackEval matching
and are unaffected by this local AP/precision/recall bug. These remain
complete-path emitted-box metrics, not raw detector-only metrics and not
parking-slot occupancy performance. See
`data/STAGE_N_V3_EMITTED_BOX_CORRECTION_REPORT.md`.

## 12. Stage O raw detector-only low-light adaptation

Stage O separates raw `YOLO.predict` detector behavior from the
tracker-emitted boxes measured in Stage N. On the already consumed LMOT
validation diagnostic, the selected O3 D1-LL checkpoint raises dark pooled
recall from 0.034694 to 0.259222 and AP50 from 0.034954 to 0.230516 at the
unchanged confidence/NMS settings. Gamma/CLAHE O1 and pretrained
Retinexformer O2 are retained as negative results; the GLARE route is
formally blocked by its unavailable legacy/native CUDA build requirements.

O3 replaces only the D1 detector candidate. B1, E1b, F2 and E4 remain frozen,
and the successful P3 reintegration is a truth-free interface smoke. LMOT has
no parking-slot polygons or occupied/vacant truth, so Stage O establishes no
nighttime occupancy or event-accuracy improvement. See
`data/STAGE_O_LOW_LIGHT_ADAPTATION_REPORT.md` and
`data/stage_o/STAGE_O_SELECTION_20260729.json`.
