# Literature-Core Parking Occupancy Workflow

This independent Part II workflow combines a slot-patch classifier and an
open-vocabulary object detector. It does not replace the existing
`parking_occupancy` baseline.

This remains the literature/ablation module, not the final default workflow.
For the closed B0/B1/E0/T0 names and the unified runnable geometry-baseline
entry point, see `../BASELINE_CLOSURE.md`. The next convergence stage will use
YOLOv8 as primary evidence and treat YOLO-World as a detector-replacement
experiment.

```text
frame + slot polygons
  |-- OpenCV perspective warp -> adapted MobileNetV3-Small -> P_cls --|
  |-- YOLO-World prompted boxes -> polygon mapping -> evidence --------|-> E3a raw weighted
                                                                        |-> E3b calibrated
                                                                            logistic fusion
                                                                        -> optional hysteresis
                                                                        -> slot states/events
```

YOLO-World detects prompted objects. It does not detect vacant spaces. A slot
is inferred vacant only when the fused occupancy evidence is below the
development-selected threshold.

## Executed status

- Implementation package: 87/87 tests passed.
- Literature-core module: 82/82 tests passed.
- Adapted MobileNetV3 pilot training: completed on CUDA.
- YOLO-World single-image check: completed.
- Historical E0-E3 Fold A camera-transfer ablation: completed.
- Frozen E0-E3 error attribution and patch montage: completed.
- Post-hoc three-camera internal development rotation: completed with
  camera-grouped selection per fold.
- E3b camera-grouped calibration/fusion development: completed; calibration
  improved reliability but did not beat E3a Macro F1.
- Standard/LeakyReLU6/CBAM/combined MobileNetV3 ablation: completed on RTX
  3060; E1b is explicitly paper-inspired, not an exact reproduction.
- Official CNR-EXT once-only external holdout: completed on 4,081 images and
  144,965 slot labels with complete-image bootstrap confidence intervals.
- NDISPark night-domain detector comparison: completed on 725 manual boxes.
- Canonical Stage I D0/D1/D2 comparison: completed; D1 selected on consumed
  development, while D2 produced the lowest count-only test MAE under the
  shared frozen rule. The negative D1 count result is retained.
- Stage I-v2: v1 retained; class-agnostic NMS, per-model development
  calibration and max-det sensitivity completed. The consumed-test rerun is
  explicitly post-hoc sensitivity.
- Stage J: P0/P1/P2 connected to identical B1 geometry on consumed PKLot
  development; P1 led pooled Macro F1 but its image-grouped interval included
  zero and substantial false-free errors remained.
- Stage K: complete on 90 previously unpredicted PKLot images and 5,034 known
  slot labels after an additive zero-overlap data gate. P1 led pooled Macro F1
  but was lower than P0 by paired-image and camera-macro summaries.
- Stage K artifacts: 43/43 main, 9/9 date/weather and 11/11 data-gate v2
  bindings verified without rerunning predictions.
- Two-frame/100-slot end-to-end video check: completed.
- Restricted Grand Bassin positive-only temporal check: completed.
- Grand Bassin continuous-truth candidate audit: completed; no valid vacant
  slot or transition was found.
- Phase A immutable-artifact verification: 17/17 hashes and both frozen CNR
  integrity counts passed.
- Phase B continuous-video source audit: VIRAT agreement acceptance is
  recorded; 26 official clips have been screened. Distinct `0502` development
  and `0503` holdout sequences have machine-verified local polygon/frame truth.
- Frozen E4/E5 departure case studies: executed. Neither beat T0 on holdout;
  E5 failed its development reliability gate, so Fusion V2 remains closed.
- General tracking, arrival, IDF1, and HOTA claims: intentionally not made
  because the temporal truth contains only one slot and one departure per
  video and no identity annotations.

All 27 selected PKLot images are method-development data; none is presented as
an external final test. See `RESULTS.md` for the measured values and
negative-result analysis.

## Environment

The audited baseline environment already contains compatible versions of
PyTorch, torchvision, OpenCV, and Ultralytics:

```powershell
cd <repository-root>\implementation
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
cd <repository-root>\implementation\literature_core
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

## Fit and audit E3b calibrated fusion

This command consumes only the three historical camera-held development
prediction files. It does not read CNR-EXT:

```powershell
..\.venv\Scripts\python.exe scripts\run_calibrated_fusion.py `
  --fold A=outputs\pklot_ablation\branch_probabilities.csv `
  --fold B=outputs\cross_camera\fold_b_ablation\branch_probabilities.csv `
  --fold C=outputs\cross_camera\fold_c_ablation\branch_probabilities.csv `
  --output-dir outputs\calibrated_fusion_development `
  --frozen-config configs\proposed_fusion.yaml
```

The tracked YAML is the single frozen E3b configuration. `P_det` is documented
as confidence times slot coverage evidence, not as a detector-native
probability.

## Run the MobileNetV3 structure ablation

Use a new output directory for every variant:

```powershell
..\.venv\Scripts\python.exe scripts\train_classifier.py `
  --annotations ..\data\annotations\pklot_development_samples.jsonl `
  --project-root .. --split-config configs\pklot_camera_split.json `
  --config configs\default.yaml `
  --output-dir outputs\mobilenet_variant_ablation\cbam_supplement `
  --variant cbam --epochs 4 --batch-size 32 --device 0
```

Available variants are `standard`, `leakyrelu6`, `cbam`, and
`cbam_leakyrelu6`. CBAM supplements the pretrained SE path. BSConv is not
implemented or claimed.

## Run the frozen CNR-EXT external evaluation

The official archive/metadata URLs, license, byte sizes, SHA-256 hashes,
geometry format, and validation counts are stored in
`data/manifests/cnrpark_ext_external_holdout.yaml`. After extracting the
official archive, the once-only command was:

```powershell
..\.venv\Scripts\python.exe scripts\run_cnr_ext_frozen_evaluation.py `
  --config configs\external_holdout_frozen.yaml `
  --dataset-root datasets\cnrpark_ext `
  --output-dir outputs\cnrpark_ext_frozen_evaluation_20260725 `
  --device 0 --warmup-frames 8
```

The evaluator refuses to run if the output directory already exists. It does
not calculate an external threshold sweep.

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

The evaluator matches an actual stable prediction change on either side of
each truth transition. It reports signed prediction-minus-truth error and
classifies events as `early`, `on_time`, `delayed`, or `missed`. Early changes
are therefore not rewritten as zero latency. Entry/exit timing, the configured
stable-frame requirement, unsupported flicker, and transition-instability
counts remain explicit. A matched delayed transition is not also counted as
ordinary flicker.

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

## Validate the frozen temporal-data protocol

The historical `pending` filename is retained for command compatibility, but
its status is now frozen:

```powershell
..\.venv\Scripts\python.exe scripts\validate_temporal_protocol.py `
  --protocol configs\temporal_protocol_pending.yaml `
  --output outputs\phase_b_protocol_audit_20260726_rerun\validation.json
```

The latest executed audit is stored under
`outputs/phase_b_protocol_audit_20260726_v7/`. Its result is
`schema_valid: true`, both partition artifact checks are true, and
`ready_for_experiment: true`. Each partition has at least 100 occupied and 100
vacant slot-frames plus one verified transition.

Run the frozen case-study implementation only into a new output directory:

```powershell
..\.venv\Scripts\python.exe scripts\run_frozen_temporal_case_study.py `
  --partition development `
  --output-dir outputs\virat_temporal_case_study_dev_rerun `
  --device 0
```

The once-only `0503` holdout has already been consumed and must not be rerun
for model selection or described as untouched for a new method. The executed
raw YOLOv8 comparator is now reported as **T0** while its historical artifact
key `e0_raw` remains unchanged. The executed E4/E5 outputs are frozen by
`data/manifests/temporal_case_study_frozen_20260726.yaml`.

## Documentation

- `FEASIBILITY_REPORT.md`: Phase 0 audit and go/no-go decisions.
- `METHOD_PROVENANCE.md`: paper/code/weights/license/difference record.
- `EXPERIMENT_PLAN.md`: frozen split, ablations, metrics, and negative-result
  rules.
- `RESULTS.md`: executed outcomes only.
- `TRANSITION_AUDIT.md`: continuous-sequence search, acceptance protocol, and
  per-candidate negative result.
- `CONFIG_AUDIT.md`: raw-result reconciliation, corrected PKLot data role, and
  generic-versus-executed configuration differences.
- `DATASET_AUDIT.md`: licensed continuous-video candidate comparison and
  conditional VIRAT acquisition/split protocol.
- `DATASET_ACCESS_BLOCKER.md`: resolved access history and the remaining
  methodological blocker for Fusion V2.
- `data/manifests/cnrpark_ext_external_holdout.yaml`: official external-data
  license, source, hashes, geometry, and integrity record.
- `data/manifests/temporal_dataset_audit_20260726.yaml`: machine-readable
  dataset facts and frozen split.
- `REPORT_SNIPPETS.md`: conservative report-ready method, contribution, and
  structural-comparison text.
- `FILE_MANIFEST.md`: complete new-file and generated-artifact inventory.
