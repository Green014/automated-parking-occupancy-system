# D0/D1/D2 Detector Comparison Protocol

Date frozen: 27 July 2026  
Protocol: `D-COMP-NDISPARK-DEV-20260727-01`  
Data role: consumed development validation

## Decision

Stage E is complete at the protocol and implementation level. It does not
contain detector scores. The real preflight verified the prepared NDISPark
data and the existing D0/D2 weights, then stopped before model loading because
D1 does not yet exist. That stop is the intended gate before the Stage F
smoke run.

| ID | Frozen method | Vehicle evidence | Current state |
|---|---|---|---|
| D0 | COCO-pretrained YOLOv8n | COCO car, motorcycle, bus, truck | Weight hash verified |
| D1 | NDISPark-fine-tuned YOLOv8n | One-class vehicle | Not trained; execution blocker |
| D2 | YOLO-World zero-shot | Prompts car, motorcycle, bus, truck | Weight hash verified |
| D3 | Fine-tuned YOLO-World | Not defined for the required comparison | Deferred optional extension |

The D1 name is now frozen as **NDISPark-fine-tuned YOLOv8n**. It must start
from the D0 pretrained weight and use the frozen 112-image training and
30-image development-validation membership.

## Common comparison conditions

- identical 30-image, 725-box NDISPark consumed development validation;
- one project class, `0: vehicle`;
- image size 640, batch 1, no test-time augmentation;
- confidence floor 0.001, NMS IoU 0.7, maximum 300 detections;
- identical one-to-one box matching at IoU 0.50 through 0.95 in steps of 0.05;
- mAP@0.5:0.95 and recall are primary, with precision and mAP@0.5 secondary;
- mean Ultralytics predict preprocess, inference, and postprocess time is the
  framework-pipeline latency; initialization and downloads are excluded;
- the NDISPark count-only test and all slot-occupancy results remain hidden
  from detector selection.

The exact registry is
`../configs/detector_comparison_frozen_20260727.yaml`.

## Class-alignment correction

The Stage D dataset deliberately uses one target class, `vehicle=0`. D0 still
emits COCO classes 2, 3, 5, and 7, while prompted D2 emits its own four prompt
indices. Passing D0's COCO IDs through Ultralytics `model.val(classes=...)`
would also filter dataset labels and would therefore remove class-0 truth.

The unified adapter instead:

1. filters source vehicle classes during prediction;
2. maps each retained prediction to project class 0;
3. leaves every class-0 ground-truth box intact;
4. performs one-to-one descending-IoU matching;
5. computes precision, recall, mAP@0.5, and mAP@0.5:0.95 with the installed
   Ultralytics AP implementation.

A synthetic regression test proves that a COCO class-2 prediction can match
class-0 vehicle truth after canonicalization without filtering that truth.
Another test prevents two predictions from claiming the same ground-truth
box.

This correction applies to the new Stage E protocol. It does not rewrite the
historical NDISPark comparison, whose older converted labels used COCO class
2, and it does not modify any historical artifact.

## Output contract

Only a complete, preflight-passing D0/D1/D2 run may create a new comparison
root. It will contain:

- `preflight.json`, `comparison.json`, and `model_runtime_table.csv`;
- one directory per D0/D1/D2;
- `metrics.json`, `runtime_metadata.json`, and `detections.jsonl`;
- PR, F1, precision, recall, and raw/normalized confusion-matrix plots.

An interrupted or failed directory is retained as a partial run and cannot be
presented as a comparison. Existing output roots are never overwritten.
Negative D1 or D2 results remain in the comparison.

## Executed preflight

The executed preflight record is
`comparisons/detector_comparison_preflight_20260727.yaml`. It verified:

- train: 112 images and 2,577 boxes;
- validation: 30 images and 725 boxes;
- D0 SHA-256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`;
- D2 SHA-256
  `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`;
- D1: missing, not trained;
- `predictions_run=false`;
- execution gate: blocked before comparison-output creation.

The ignored JSON evidence uses a new `v2` filename. An earlier preflight-only
`v1` file was retained rather than overwritten, but it is not the frozen
Stage E evidence.

## Verification

- Stage E targeted tests: 9 passed;
- complete `implementation` suite: 55 passed;
- complete `literature_core` suite: 82 passed;
- read-only AST syntax check: 119 Python files passed;
- historical static freeze: 17/17 artifacts, 4,081 frames, and 144,965 slot
  records verified;
- historical temporal freeze: 11/11 artifacts verified;
- `git diff --check`: passed, with line-ending notices only.

## Reproduction

Paths remain runtime inputs:

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

After Stage F/H creates the recorded D1 weight:

```powershell
python scripts\run_detector_comparison.py `
  --config configs\detector_comparison_frozen_20260727.yaml `
  --data $env:PARKING_PREPARED_DATA `
  --d0-weights $env:PARKING_D0_WEIGHTS `
  --d1-weights $env:PARKING_D1_WEIGHTS `
  --d1-sha256 $env:PARKING_D1_SHA256 `
  --d2-weights $env:PARKING_D2_WEIGHTS `
  --device 0 `
  --execute `
  --output-dir outputs\detector_comparison_<new-id>
```

No D0, D1, or D2 prediction was run during Stage E.
