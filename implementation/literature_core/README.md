# Literature-Core Parking Occupancy Workflow

This independent Part II workflow combines a slot-patch classifier and an
open-vocabulary object detector. It does not replace the existing
`parking_occupancy` baseline.

```text
frame + slot polygons
  |-- OpenCV perspective warp -> adapted MobileNetV3-Small -> P_cls --|
  |-- YOLO-World prompted boxes -> polygon mapping -> P_det ----------|-> weighted fusion
                                                                        -> optional hysteresis
                                                                        -> slot states/events
```

YOLO-World detects prompted objects. It does not detect vacant spaces. A slot
is inferred vacant only when the fused occupancy evidence is below the
development-selected threshold.

## Executed status

- Existing baseline: 23/23 tests passed.
- Literature-core module: 37/37 tests passed.
- Adapted MobileNetV3 pilot training: completed on CUDA.
- YOLO-World single-image check: completed.
- E0-E3 camera-holdout ablation: completed.
- Frozen E0-E3 error attribution and patch montage: completed.
- Post-hoc three-camera rotation: completed with disjoint selection per fold.
- NDISPark night-domain detector comparison: completed on 725 manual boxes.
- Two-frame/100-slot end-to-end video check: completed.
- Restricted Grand Bassin positive-only temporal check: completed.
- Grand Bassin continuous-truth candidate audit: completed; no valid vacant
  slot or transition was found.
- Full E4/E5 claims: intentionally not made because suitable mixed-class
  continuous/identity ground truth is not locally available.

See `RESULTS.md` for the measured values and negative-result analysis.

## Environment

The audited baseline environment already contains compatible versions of
PyTorch, torchvision, OpenCV, and Ultralytics:

```powershell
cd C:\Users\panda\Documents\停车场识别系统项目\implementation
.\.venv\Scripts\python.exe -m pip install -e .\literature_core
```

Do not install an older MMDetection/MMCV stack into the baseline environment.
For a clean environment, first install a CUDA build of PyTorch/torchvision
appropriate for the machine, then install `requirements.txt`.

Before commands that import Ultralytics, keep its settings inside the project:

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path ..\.ultralytics).Path
$env:TORCH_HOME = (Resolve-Path .\models\torch-cache).Path
```

`models/` and `outputs/` are ignored by the independent module. The CLIP
adapter cache used by the installed Ultralytics version is the existing shared
project path `..\..\weights\clip\ViT-B-32.pt`; do not copy it into the module.

## Tests

```powershell
cd C:\Users\panda\Documents\停车场识别系统项目\implementation\literature_core
..\.venv\Scripts\python.exe -m pytest -q
```

## Extract an auditable patch manifest

The training loader extracts patches on demand; this command is an optional
visual/data audit and does not duplicate the full dataset:

```powershell
..\.venv\Scripts\python.exe scripts\extract_patches.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. `
  --split-config configs\pklot_camera_split.json `
  --output-dir outputs\patch_audit `
  --limit-per-split 24
```

## Train the adapted MobileNetV3-Small classifier

The default freezes the backbone, uses ImageNet initialization, mixed
precision on CUDA, batch 32, and a deterministic seed. This is an adaptation
of [2], not its exact architecture.

```powershell
..\.venv\Scripts\python.exe scripts\train_classifier.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. `
  --split-config configs\pklot_camera_split.json `
  --config configs\default.yaml `
  --output-dir outputs\mobilenet_pilot `
  --epochs 4 --batch-size 32 --device 0
```

The executed pilot used four epochs. The YAML default remains 12 for a longer
follow-up. For a low-memory run, use `--batch-size 16`. To test code without
downloading ImageNet weights, use `--no-pretrained`; the result is not a valid
transfer-learning experiment.

## Validate YOLO-World on one local image

Place/download the checkpoint under the ignored `models/` directory:

```powershell
..\.venv\Scripts\python.exe scripts\verify_yolo_world.py `
  --image ..\data\research_samples\pklot_2012-09-21_14-45-32.jpg `
  --weights models\yolov8s-worldv2.pt `
  --output-dir outputs\yolo_world_smoke `
  --conf 0.025 --imgsz 1280 --device 0
```

Prompts default to `car`, `truck`, `bus`, and `motorcycle`. The command saves
raw bounding boxes/confidences and an annotated image.

## Run the frozen E0-E3 pilot ablation

```powershell
..\.venv\Scripts\python.exe scripts\run_pklot_ablation.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. `
  --split-config configs\pklot_camera_split.json `
  --classifier-checkpoint outputs\mobilenet_pilot\best.pt `
  --world-weights models\yolov8s-worldv2.pt `
  --baseline-weights ..\yolov8n.pt `
  --config configs\default.yaml `
  --output-dir outputs\pklot_ablation
```

The script selects thresholds and fusion weights only on UFPR04, freezes them,
and evaluates UFPR05. It writes branch probabilities, raw detections,
development sensitivity, selected parameters, and test metrics.

The executed pilot selected E1 threshold 0.61, E2 threshold 0.08, and E3
weights 0.5/0.5 with threshold 0.37. Those PKLot-specific values are stored in
`outputs/pklot_ablation/selected_parameters.json`; the generic defaults are
not silently rewritten.

Explain the frozen predictions without changing them:

```powershell
..\.venv\Scripts\python.exe scripts\analyze_ablation_errors.py `
  --probabilities outputs\pklot_ablation\branch_probabilities.csv `
  --parameters outputs\pklot_ablation\selected_parameters.json `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. `
  --split-config configs\pklot_camera_split.json `
  --output-dir outputs\pklot_error_analysis
```

## Run the post-hoc camera rotation

Fold A is the original run. Fold B/C use the same seed, four epochs, frozen
backbone, and AMP:

```powershell
..\.venv\Scripts\python.exe scripts\train_classifier.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. --split-config configs\pklot_fold_b.json `
  --config configs\default.yaml `
  --output-dir outputs\cross_camera\fold_b_classifier `
  --epochs 4 --batch-size 32 --device 0

..\.venv\Scripts\python.exe scripts\train_classifier.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. --split-config configs\pklot_fold_c.json `
  --config configs\default.yaml `
  --output-dir outputs\cross_camera\fold_c_classifier `
  --epochs 4 --batch-size 32 --device 0
```

Run `run_pklot_ablation.py` for each split using its corresponding checkpoint:

```powershell
..\.venv\Scripts\python.exe scripts\run_pklot_ablation.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. --split-config configs\pklot_fold_b.json `
  --classifier-checkpoint outputs\cross_camera\fold_b_classifier\best.pt `
  --world-weights models\yolov8s-worldv2.pt `
  --baseline-weights ..\yolov8n.pt --config configs\default.yaml `
  --output-dir outputs\cross_camera\fold_b_ablation --device 0

..\.venv\Scripts\python.exe scripts\run_pklot_ablation.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. --split-config configs\pklot_fold_c.json `
  --classifier-checkpoint outputs\cross_camera\fold_c_classifier\best.pt `
  --world-weights models\yolov8s-worldv2.pt `
  --baseline-weights ..\yolov8n.pt --config configs\default.yaml `
  --output-dir outputs\cross_camera\fold_c_ablation --device 0
```

Aggregate the three frozen metrics:

```powershell
..\.venv\Scripts\python.exe scripts\summarize_cross_camera.py `
  --fold A=outputs\pklot_ablation\metrics.json `
  --fold B=outputs\cross_camera\fold_b_ablation\metrics.json `
  --fold C=outputs\cross_camera\fold_c_ablation\metrics.json `
  --output-dir outputs\cross_camera\summary
```

## Compare detector boxes on NDISPark night images

This is separate from slot occupancy evaluation. Both methods are evaluated
against the same 725 manual boxes as a single vehicle class:

```powershell
..\.venv\Scripts\python.exe scripts\run_ndispark_detection.py `
  --data ..\data\processed\ndispark\dataset.yaml `
  --world-weights models\yolov8s-worldv2.pt `
  --baseline-weights ..\yolov8n.pt `
  --output-dir outputs\ndispark_detection_comparison `
  --imgsz 1280 --device 0
```

## Run video inference

```powershell
..\.venv\Scripts\python.exe scripts\run_video.py `
  --input ..\data\raw\video\camera01.mp4 `
  --slots ..\data\annotations\camera01_slots.json `
  --classifier-checkpoint outputs\mobilenet_pilot\best.pt `
  --world-weights models\yolov8s-worldv2.pt `
  --config configs\default.yaml `
  --output-dir outputs\camera01_literature_core
```

Use temporal results only when the complete video has manually verified
occupied/vacant and transition truth.

Evaluate a verified sequence with the corrected temporal definition:

```powershell
..\.venv\Scripts\python.exe scripts\evaluate_predictions.py `
  --ground-truth data\camera01_truth.csv `
  --predictions outputs\camera01_literature_core\occupancy.csv `
  --fps 25 --stable-frames 3 --tolerance-frames 1 `
  --output outputs\camera01_literature_core\evaluation\metrics.json
```

The evaluator reports delayed stable transitions as transition latency. It
does not also count that same delayed transition as ordinary flicker.

Video `occupancy.csv` explicitly separates raw fused `p_occ` from
EMA-filtered `p_occ_filtered`, then records `raw_state` and the final
hysteresis state.

## Restricted Grand Bassin positive-only check

The following command uses the parameters frozen before looking at the
Grand Bassin result:

```powershell
..\.venv\Scripts\python.exe scripts\run_video.py `
  --input ..\data\raw\grand_bassin\grand_bassin_aerial_development.mp4 `
  --slots ..\data\annotations\grand_bassin_bus_stability_slots.json `
  --classifier-checkpoint outputs\mobilenet_pilot\best.pt `
  --world-weights models\yolov8s-worldv2.pt `
  --config configs\grand_bassin_frozen.yaml `
  --output-dir outputs\grand_bassin_frozen `
  --device 0 --no-video

..\.venv\Scripts\python.exe scripts\evaluate_positive_stability.py `
  --ground-truth ..\data\annotations\grand_bassin_bus_stability_ground_truth.csv `
  --predictions outputs\grand_bassin_frozen\occupancy.csv `
  --summary outputs\grand_bassin_frozen\summary.json `
  --parameters outputs\pklot_ablation\selected_parameters.json `
  --fps 2 --warmup-frames 6 `
  --output-dir outputs\grand_bassin_frozen\evaluation
```

This evaluator refuses negative labels and explicitly excludes vacant,
transition, and tracking claims.

## Audit continuous transition candidates

The local Grand Bassin sources were searched for a defensible mixed-class or
transition annotation before any full E4 metric was attempted. The search
reviewed 1,349 ordered samples across three sequences. Sixteen automated
hypotheses (34 proposed changes) and seven targeted manual hypotheses were
adjudicated with full-context and ROI contact sheets. None met the acceptance
rule for one complete, legal marked bay with a human-visible state change.

The review utilities are reproducible:

```powershell
..\.venv\Scripts\python.exe scripts\build_transition_review.py `
  --video ..\data\raw\grand_bassin\grand_bassin_aerial_development.mp4 `
  --candidates outputs\candidate_search\development.csv `
  --output-dir outputs\candidate_search\development_review `
  --radius 10 --uniform-samples 12 --padding 45
```

See `TRANSITION_AUDIT.md` and the adjudication CSVs under
`data/annotations/`. Review sheets are evidence, not ground truth. The
positive-only labels and results were retained unchanged.

## Documentation

- `FEASIBILITY_REPORT.md`: Phase 0 audit and go/no-go decisions.
- `METHOD_PROVENANCE.md`: paper/code/weights/license/difference record.
- `EXPERIMENT_PLAN.md`: frozen split, ablations, metrics, and negative-result
  rules.
- `RESULTS.md`: executed outcomes only.
- `TRANSITION_AUDIT.md`: continuous-sequence search, acceptance protocol, and
  per-candidate negative result.
- `REPORT_SNIPPETS.md`: conservative report-ready method, contribution, and
  structural-comparison text.
- `FILE_MANIFEST.md`: complete new-file and generated-artifact inventory.
