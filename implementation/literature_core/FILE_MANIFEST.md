# Delivery File Manifest

All implementation files below are new. No pre-existing baseline source,
report, CSV, model, or output file was edited or overwritten.

## Documentation and packaging

- `.gitignore`
- `README.md`
- `FEASIBILITY_REPORT.md`
- `METHOD_PROVENANCE.md`
- `EXPERIMENT_PLAN.md`
- `RESULTS.md`
- `TRANSITION_AUDIT.md`
- `REPORT_SNIPPETS.md`
- `FILE_MANIFEST.md`
- `requirements.txt`
- `pyproject.toml`

## Configuration

- `configs/default.yaml`
- `configs/grand_bassin_frozen.yaml`
- `configs/pklot_fold_b.json`
- `configs/pklot_fold_c.json`
- `configs/pklot_camera_split.json`

## Python package

- `src/literature_core/__init__.py`
- `src/literature_core/models.py`
- `src/literature_core/config.py`
- `src/literature_core/data.py`
- `src/literature_core/patches.py`
- `src/literature_core/classifier.py`
- `src/literature_core/detector.py`
- `src/literature_core/mapping.py`
- `src/literature_core/fusion.py`
- `src/literature_core/temporal.py`
- `src/literature_core/metrics.py`
- `src/literature_core/error_analysis.py`
- `src/literature_core/stability.py`
- `src/literature_core/cross_validation.py`
- `src/literature_core/annotation_review.py`
- `src/literature_core/pipeline.py`

## Reproduction scripts

- `scripts/extract_patches.py`
- `scripts/train_classifier.py`
- `scripts/verify_yolo_world.py`
- `scripts/run_pklot_ablation.py`
- `scripts/run_ndispark_detection.py`
- `scripts/run_video.py`
- `scripts/evaluate_predictions.py`
- `scripts/analyze_ablation_errors.py`
- `scripts/evaluate_positive_stability.py`
- `scripts/summarize_cross_camera.py`
- `scripts/build_transition_review.py`
- `scripts/draw_slot_map.py`

## Tests

- `tests/test_patches.py`
- `tests/test_data.py`
- `tests/test_classifier.py`
- `tests/test_mapping.py`
- `tests/test_fusion.py`
- `tests/test_temporal.py`
- `tests/test_metrics.py`
- `tests/test_pipeline.py`
- `tests/test_error_analysis.py`
- `tests/test_stability.py`
- `tests/test_cross_validation.py`
- `tests/test_annotation_review.py`

## Continuous-sequence audit data

- `data/annotations/grand_bassin_transition_candidate_adjudication.csv`
- `data/annotations/grand_bassin_rejected_manual_hypotheses.csv`
- `data/annotations/grand_bassin_development_rejected_review_candidates.csv`
- `data/annotations/grand_bassin_holdout_rejected_review_candidates.csv`
- `data/annotations/grand_bassin_bus_area_review.csv`

## Generated, intentionally ignored artifacts

- `models/yolov8s-worldv2.pt`
- `models/torch-cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`
- `outputs/mobilenet_pilot/`
- `outputs/yolo_world_smoke/`
- `outputs/pklot_ablation/`
- `outputs/ndispark_detection_comparison/`
- `outputs/pipeline_smoke/`
- `outputs/pklot_error_analysis/`
- `outputs/grand_bassin_frozen/`
- `outputs/cross_camera/`
- `outputs/phase0_baseline_smoke/`
- `outputs/patch_audit/`
- `outputs/candidate_search/`
- `outputs/manual_transition_review/`

The shared CLIP adapter cache is stored once at
`../../weights/clip/ViT-B-32.pt`. The new `../../weights/.gitignore` prevents
that 354 MB runtime cache from being committed.
