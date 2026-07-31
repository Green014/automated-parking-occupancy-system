from __future__ import annotations

import importlib.util
import os

import pytest


OFFICIAL_TRACKEVAL_TESTS = {
    "tests/test_stage_n_lmot.py::test_official_trackeval_perfect_tracking",
    "tests/test_stage_n_lmot.py::test_official_trackeval_detects_id_switch",
    "tests/test_stage_n_lmot.py::test_official_trackeval_counts_miss_and_false_positive",
}

LOCAL_ONLY_ARTIFACT_TESTS = {
    "tests/test_stage_s_release.py::test_historical_registry_entry_state_is_unchanged",
    "tests/test_stage_n_lmot.py::test_stage_n_protocol_preserves_frozen_inputs",
    "tests/test_stage_n_lmot_v2.py::test_stage_n_v2_protocol_preserves_original_stage_n",
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_real_preflight_verifies_confirmation_and_all_manifest_images"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_preflight_refuses_existing_output_root"
    ),
    (
        "tests/test_stage_r_component_attribution.py::"
        "test_repository_stage_r_matches_independent_sanity_values"
    ),
    (
        "tests/test_stage_s_demo.py::"
        "test_demo_plan_uses_required_frozen_consecutive_segments"
    ),
    (
        "tests/test_stage_s_demo.py::"
        "test_demo_recoveries_and_failures_follow_frozen_fields"
    ),
}

PUBLIC_SOURCE_OMITTED_ARTIFACT_TESTS = {
    (
        "tests/test_p1_temporal_case.py::"
        "test_frozen_p1_temporal_protocol_binds_truth_before_prediction"
    ),
    (
        "tests/test_p1_temporal_case.py::"
        "test_temporal_truth_uses_half_open_transition_boundary"
    ),
    (
        "tests/test_stage_j_occupancy.py::"
        "test_frozen_stage_j_protocol_keeps_detector_and_mapping_isolation"
    ),
    (
        "tests/test_stage_k_stratified_analysis.py::"
        "test_stage_k_strata_protocol_is_read_only"
    ),
    (
        "tests/test_stage_m_tracking.py::"
        "test_stage_m_protocol_and_tracker_configs_are_frozen"
    ),
    (
        "tests/test_stage_o_low_light.py::"
        "test_detector_only_runner_writes_complete_contract_without_tracker"
    ),
    (
        "tests/test_stage_o_low_light.py::"
        "test_o2_preprocessor_is_applied_to_dark_stream_only"
    ),
    (
        "tests/test_stage_q_external.py::"
        "test_repository_gate_preserves_stage_p_fail_and_d1_default"
    ),
    (
        "tests/test_stage_q_v2_artifacts.py::"
        "test_repository_premodel_artifacts_are_bound_and_blocked"
    ),
    (
        "tests/test_stage_q_v2_artifacts.py::"
        "test_repository_completed_registry_verifies_when_present"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_config_rejects_changed_d1_default_role"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_frozen_config_keeps_roles_and_identical_inference"
    ),
    (
        "tests/test_stage_r_component_attribution.py::"
        "test_repository_stage_r_registry_verifies_when_present"
    ),
    (
        "tests/test_stage_s_release.py::"
        "test_stage_s_registry_when_frozen"
    ),
    (
        "tests/test_stage_t_tracktrack.py::"
        "test_stage_s_registry_remains_valid_after_stage_t"
    ),
    (
        "tests/test_stage_t_tracktrack.py::"
        "test_stage_t_registry_when_frozen"
    ),
    (
        "tests/test_stage_u_p3_tt_runtime.py::"
        "test_stage_s_default_and_registry_remain_unchanged"
    ),
    (
        "tests/test_stage_u_portable_release.py::"
        "test_portable_registry_has_relative_non_output_paths_when_generated"
    ),
    (
        "tests/test_stage_u_portable_release.py::"
        "test_saved_audit_csv_and_registry_form_stable_one_way_chain"
    ),
    (
        "tests/test_stage_w_3_privacy_and_model_release.py::"
        "test_frozen_model_release_assets_are_exact_and_git_ignored"
    ),
    (
        "tests/test_stage_w_3_privacy_and_model_release.py::"
        "test_w3_registry_covers_release_boundary_and_verifies"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    trackeval_available = importlib.util.find_spec("trackeval") is not None
    portable_package = os.environ.get("STAGE_U_PORTABLE_PACKAGE") == "1"
    public_source_package = (
        os.environ.get("PARKING_PUBLIC_SOURCE_PACKAGE") == "1"
    )
    for item in items:
        if item.nodeid in OFFICIAL_TRACKEVAL_TESTS:
            item.add_marker(pytest.mark.trackeval)
            if not trackeval_available:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "optional TrackEval dependency is not installed; "
                            "install .[trackeval] to run this MOT diagnostic"
                        )
                    )
                )
        if portable_package and item.nodeid in LOCAL_ONLY_ARTIFACT_TESTS:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "requires a local-only historical registry, frozen "
                        "runtime output, model weight, or dataset omitted from "
                        "the Stage U portable package"
                    )
                )
            )
        if (
            public_source_package
            and item.nodeid in PUBLIC_SOURCE_OMITTED_ARTIFACT_TESTS
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "requires a frozen local artifact, model weight, or "
                        "intermediate evidence intentionally omitted from "
                        "the W.3 public source package"
                    )
                )
            )
