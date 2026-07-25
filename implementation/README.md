# Automated Parking Lot Occupancy and Tracking System

This independent Part II implementation converts vehicle detections into
parking-slot states:

```text
OpenCV video -> YOLOv8 vehicles -> slot polygon mapping
             -> optional ByteTrack -> temporal hysteresis
             -> video + occupancy/events/timing logs
```

YOLO detects vehicles. It does not directly detect vacant spaces. The
centre-point or polygon-overlap mapper is the explicit step that assigns a
vehicle detection to a predefined parking slot.

## Current experiments

- `b0`: YOLO + bounding-box centre inside polygon.
- `b1`: YOLO + confidence-weighted slot polygon coverage.
- `proposed`: YOLO + ByteTrack + polygon coverage + temporal hysteresis.

The no-training baseline is complete. Current development reports are:

- `metadata/PKLOT_DEVELOPMENT_REPORT.md`;
- `metadata/GRAND_BASSIN_TEMPORAL_REPORT.md`;
- `metadata/FINETUNING_GATE_ASSESSMENT.md`.

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

## 2. Run B0

```powershell
.\.venv\Scripts\parking-run.exe `
  --input data\raw\video\camera01.mp4 `
  --slots data\annotations\camera01_slots.json `
  --experiment b0 `
  --weights yolov8n.pt `
  --device auto `
  --output-dir outputs\camera01_b0
```

The output directory contains:

- `annotated.mp4`;
- `occupancy.csv` with one row per frame/slot;
- `events.csv` with slot transitions;
- `summary.json` with checksums, environment, FPS, and latency.

Run `b1` or `proposed` by changing `--experiment`. The Proposed command enables
ByteTrack with the selected tracker configuration. Tracking IDs are attached
to matched boxes while unmatched YOLO detections are retained, so ByteTrack
does not become a second detector gate.

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

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## 6. Data and scope

See `data/DATASET_MANIFEST.md` for license and suitability decisions. Raw data,
generated outputs, and model weights are ignored by Git. The first experiments
use video/camera/date-level splits; adjacent frames are never randomly divided
between train, validation, and test.

`data/finetune/grand_bassin_vehicle_preannotations.json` is not ground truth.
It must be manually corrected according to
`data/finetune/ANNOTATION_PLAN.md` before any YOLO fine-tuning run.
