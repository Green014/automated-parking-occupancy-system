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
- `CONFIG_AUDIT.md`
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
- `configs/proposed_fusion.yaml`
- `configs/external_holdout_frozen.yaml`
- `configs/temporal_protocol_pending.yaml`

## Python package

- `src/literature_core/__init__.py`
- `src/literature_core/models.py`
- `src/literature_core/config.py`
- `src/literature_core/data.py`
- `src/literature_core/patches.py`
- `src/literature_core/classifier.py`
- `src/literature_core/calibration.py`
- `src/literature_core/cnrpark.py`
- `src/literature_core/detector.py`
- `src/literature_core/mapping.py`
- `src/literature_core/fusion.py`
- `src/literature_core/temporal.py`
- `src/literature_core/metrics.py`
- `src/literature_core/error_analysis.py`
- `src/literature_core/stability.py`
- `src/literature_core/cross_validation.py`
- `src/literature_core/annotation_review.py`
- `src/literature_core/temporal_protocol.py`
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
- `scripts/run_calibrated_fusion.py`
- `scripts/evaluate_classifier_variants.py`
- `scripts/run_cnr_ext_frozen_evaluation.py`
- `scripts/verify_frozen_artifacts.py`
- `scripts/validate_temporal_protocol.py`
- `scripts/acquire_virat_screening.py`
- `scripts/build_virat_screening_contact_sheet.py`
- `scripts/verify_virat_screening.py`

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
- `tests/test_calibration.py`
- `tests/test_cnrpark.py`
- `tests/test_frozen_artifacts.py`
- `tests/test_temporal_protocol.py`
- `tests/test_virat_access.py`

## External-data manifest

- `data/manifests/cnrpark_ext_external_holdout.yaml`
- `data/manifests/frozen_artifacts_20260725.yaml`
- `data/manifests/temporal_dataset_audit_20260726.yaml`
- `data/manifests/virat_screening_20260726.yaml`

## Stage A/B audit reports

- `DATASET_AUDIT.md`
- `DATASET_ACCESS_BLOCKER.md`

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
- `outputs/calibrated_fusion_development/`
- `outputs/mobilenet_variant_ablation/`
- `outputs/mobilenet_variant_evaluation/`
- `outputs/cnrpark_ext_frozen_evaluation_20260725/`
- `outputs/phase_a_freeze_audit_20260726/`
- `outputs/phase_a_freeze_audit_20260726_v2/`
- `outputs/phase_a_freeze_audit_20260726_v3/`
- `outputs/phase_b_protocol_audit_20260726/`
- `outputs/phase_b_protocol_audit_20260726_v2/`
- `outputs/phase_b_protocol_audit_20260726_v3/`
- `outputs/phase_b_protocol_audit_20260726_v4/`
- `outputs/phase_b_protocol_audit_20260726_v5/`
- `outputs/phase_b_protocol_audit_20260726_v6/`
- `outputs/virat_0503_targeted_verification_20260726/`
- `outputs/virat_screening*/`
- `outputs/virat_review_*/`
- `outputs/virat_grid_*/`
- `outputs/virat_slot_review_*/`
- `datasets/cnrpark_ext/`
- `datasets/virat/screening/`

The shared CLIP adapter cache is stored once at
`../../weights/clip/ViT-B-32.pt`. The new `../../weights/.gitignore` prevents
that 354 MB runtime cache from being committed.
