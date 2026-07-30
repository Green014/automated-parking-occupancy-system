# Delivery File Manifest

The original literature-core delivery files are listed below. Its initial
creation did not overwrite the pre-existing baseline. The later baseline
closure intentionally edits baseline source/documentation and is recorded in
the addendum; no historical CSV, model, or frozen output was edited.

## 26 July evaluation/baseline-closure addendum

New cross-package files:

- `../BASELINE_CLOSURE.md`
- `../configs/baseline_methods.yaml`
- `../src/parking_occupancy/method_registry.py`
- `../tests/test_method_registry.py`
- `../tests/test_pipeline_outputs.py`

Updated implementation/evaluation files:

- `../PLAN.md`
- `../README.md`
- `../src/parking_occupancy/cli.py`
- `../src/parking_occupancy/pipeline.py`
- `src/literature_core/metrics.py`
- `scripts/evaluate_predictions.py`
- `tests/test_metrics.py`
- `README.md`
- `RESULTS.md`
- `METHOD_PROVENANCE.md`
- `EXPERIMENT_PLAN.md`
- `REPORT_SNIPPETS.md`
- `DATASET_ACCESS_BLOCKER.md`
- `FILE_MANIFEST.md`

The addendum contains code, tests, configuration, and documentation only. It
does not contain a recomputed experiment result.

## 27 July Stage I-v2 and Stage J/K addendum

- `../configs/detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml`
- `../configs/detector_comparison_stage_i_v2_maxdet1000_frozen_20260727.yaml`
- `../configs/stage_i_v2_posthoc_count_frozen_20260727.yaml`
- `../data/STAGE_I_V2_CORRECTED_EVALUATION_REPORT.md`
- `../data/comparisons/stage_i_v2_corrected_evaluation_20260727.yaml`
- `../scripts/verify_stage_i_v2_artifacts.py`
- `../configs/stage_j_p0_p1_p2_pklot_development_frozen_20260727.yaml`
- `../data/manifests/stage_j_pklot_development_20260727.csv`
- `../data/STAGE_J_K_OCCUPANCY_REPORT.md`
- `../data/comparisons/stage_j_p0_p1_p2_development_20260727.yaml`
- `../data/comparisons/stage_k_slot_occupancy_data_gate_20260727.yaml`
- `../src/parking_occupancy/stage_j_occupancy.py`
- `../scripts/run_stage_j_occupancy.py`
- `../scripts/verify_stage_j_artifacts.py`
- `../tests/test_stage_j_occupancy.py`

Generated Stage I-v2 and Stage J outputs remain under ignored
`../outputs/` directories. No v1 output, weight or raw dataset is included.

## 28 July Stage K closure addendum

- `../configs/stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml`
- `../configs/stage_k_posthoc_stratified_analysis_frozen_20260728.yaml`
- `../data/annotations/pklot_stage_k_candidate_20260727_v2.jsonl`
- `../data/manifests/pklot_stage_k_candidate_20260727_v2.csv`
- `../data/preprocessing/pklot_stage_k_candidate_audit_20260727_v2.json`
- `../data/preprocessing/pklot_stage_k_manual_visual_review_20260727_v2.json`
- `../data/comparisons/stage_k_slot_occupancy_data_gate_20260728_v2.yaml`
- `../data/comparisons/stage_k_p0_p1_p2_test_20260727.yaml`
- `../data/comparisons/stage_k_posthoc_stratified_analysis_20260728.yaml`
- `../data/STAGE_K_FINAL_REPORT.md`
- `../src/parking_occupancy/stage_k_data_gate.py`
- `../src/parking_occupancy/stage_k_occupancy.py`
- `../src/parking_occupancy/stage_k_stratified_analysis.py`
- `../scripts/freeze_stage_k_data_gate_v2.py`
- `../scripts/verify_stage_k_data_gate_v2.py`
- `../scripts/verify_stage_k_artifacts.py`
- `../scripts/verify_stage_k_strata_artifacts.py`
- `../tests/test_stage_k_data_gate.py`
- `../tests/test_stage_k_occupancy.py`
- `../tests/test_stage_k_stratified_analysis.py`

Generated Stage K predictions remain under ignored
`../outputs/P0_P1_P2_stage_k_20260727_v1/`. The read-only date/weather
outputs remain under ignored `../outputs/stage_k_posthoc_strata_20260728_v1/`.
The registries bind those outputs by byte count and SHA-256; no model output
is copied into the tracked source tree.

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
- `configs/temporal_e4_e5_frozen.yaml`

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
- `src/literature_core/temporal_tracking.py`
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
- `scripts/run_frozen_temporal_case_study.py`

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
- `tests/test_temporal_tracking.py`
- `tests/test_detector_tracking.py`
- `tests/test_virat_access.py`
- `../src/parking_occupancy/gpu_decision.py`
- `../scripts/analyze_gpu_decision.py`
- `../tests/test_gpu_decision.py`
- `../tests/test_gpu_decision_freeze.py`
- `../configs/d1_ndispark_formal_frozen_20260727.yaml`
- `../data/GPU_DECISION_REPORT.md`
- `../data/training/d1_gpu_decision_20260727.yaml`
- `../data/training/D1_GPU_DECISION_FROZEN_CHECKSUMS.yaml`
- `../src/parking_occupancy/formal_training.py`
- `../scripts/run_d1_formal.py`
- `../scripts/finalize_d1_formal.py`
- `../scripts/verify_d1_formal_artifacts.py`
- `../tests/test_formal_training.py`
- `../tests/test_formal_training_freeze.py`
- `../data/D1_FORMAL_TRAINING_REPORT.md`
- `../data/training/d1_formal_training_20260727.yaml`
- `../data/training/D1_FORMAL_TRAINING_FROZEN_CHECKSUMS.yaml`
- `../configs/stage_i_count_test_frozen_20260727.yaml`
- `../src/parking_occupancy/stage_i_evaluation.py`
- `../scripts/run_stage_i_evaluation.py`
- `../scripts/verify_stage_i_artifacts.py`
- `../tests/test_stage_i_evaluation.py`
- `../data/DETECTOR_EVALUATION_REPORT.md`
- `../data/comparisons/stage_i_detector_evaluation_20260727.yaml`
- `../data/comparisons/stage_i_timestamp_correction_20260727.yaml`
- `../data/comparisons/STAGE_I_FROZEN_CHECKSUMS.yaml`

## External-data manifest

- `data/manifests/cnrpark_ext_external_holdout.yaml`
- `data/manifests/frozen_artifacts_20260725.yaml`
- `data/manifests/temporal_dataset_audit_20260726.yaml`
- `data/manifests/virat_screening_20260726.yaml`
- `data/manifests/virat_0503_targeted_screening_20260726.yaml`
- `data/manifests/temporal_case_study_frozen_20260726.yaml`

## Stage A/B audit reports

- `DATASET_AUDIT.md`
- `DATASET_ACCESS_BLOCKER.md`

## Continuous-sequence audit data

- `data/annotations/grand_bassin_transition_candidate_adjudication.csv`
- `data/annotations/grand_bassin_rejected_manual_hypotheses.csv`
- `data/annotations/grand_bassin_development_rejected_review_candidates.csv`
- `data/annotations/grand_bassin_holdout_rejected_review_candidates.csv`
- `data/annotations/grand_bassin_bus_area_review.csv`
- `data/annotations/virat_0502_departure_truth.yaml`
- `data/annotations/virat_0503_departure_truth.yaml`

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
- `outputs/phase_a_freeze_audit_20260726_v4/`
- `outputs/phase_a_freeze_audit_20260726_v5/`
- `outputs/phase_b_protocol_audit_20260726/`
- `outputs/phase_b_protocol_audit_20260726_v2/`
- `outputs/phase_b_protocol_audit_20260726_v3/`
- `outputs/phase_b_protocol_audit_20260726_v4/`
- `outputs/phase_b_protocol_audit_20260726_v5/`
- `outputs/phase_b_protocol_audit_20260726_v6/`
- `outputs/phase_b_protocol_audit_20260726_v7/`
- `outputs/phase_b_protocol_audit_20260726_v8/`
- `outputs/virat_0503_targeted_verification_20260726/`
- `outputs/virat_0503_targeted_verification_20260726_v2/`
- `outputs/virat_temporal_case_study_dev_20260726_v1/`
- `outputs/virat_temporal_case_study_holdout_20260726_v1/`
- `outputs/temporal_case_study_freeze_audit_20260726_v1/`
- `outputs/temporal_case_study_freeze_audit_20260726_v2/`
- `../outputs/d1_ndispark_smoke_20260727_v1/`
- `../outputs/d1_ndispark_smoke_20260727_v2/`
- `../outputs/d1_ndispark_smoke_20260727_v3/`
- `../outputs/gpu_decision_20260727_v1.json`
- `../outputs/d1_ndispark_formal_20260727_v1/`
- `../outputs/d1_formal_launcher_stdout_20260727_v1.log`
- `../outputs/d1_formal_launcher_stderr_20260727_v1.log`
- `../outputs/d1_formal_training_verification_20260727_v1.json`
- `../outputs/detector_comparison_stage_i_20260727_v1/`
- `../outputs/stage_i_detector_selection_20260727_v1.json`
- `../outputs/detector_count_test_stage_i_20260727_v1/`
- `../outputs/detector_count_test_stage_i_20260727_v2/`
- `../outputs/detector_qualitative_stage_i_20260727_v1/`
- `../outputs/stage_i_artifact_verification_20260727_v1.json`
- `../outputs/detector_comparison_stage_i_v2_maxdet300_20260727_v2/`
- `../outputs/detector_comparison_stage_i_v2_maxdet1000_20260727_v1/`
- `../outputs/detector_count_test_stage_i_v2_posthoc_20260727_v1/`
- `../outputs/P0_P1_P2_stage_j_20260727_v1/`
- `../outputs/stage_j_artifact_verification_20260727_v1.json`
- `outputs/virat_screening*/`
- `outputs/virat_review_*/`
- `outputs/virat_grid_*/`
- `outputs/virat_slot_review_*/`
- `datasets/cnrpark_ext/`
- `datasets/virat/screening/`

The shared CLIP adapter cache is stored once at
`../../weights/clip/ViT-B-32.pt`. The new `../../weights/.gitignore` prevents
that 354 MB runtime cache from being committed.
