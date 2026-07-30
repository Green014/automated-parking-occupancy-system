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

PUBLIC_SOURCE_ARTIFACT_TESTS = {
    (
        "tests/test_stage_q_v2_artifacts.py::"
        "test_repository_premodel_artifacts_are_bound_and_blocked"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_frozen_config_keeps_roles_and_identical_inference"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_config_rejects_changed_d1_default_role"
    ),
    "tests/test_stage_s_release.py::test_stage_s_registry_when_frozen",
    (
        "tests/test_stage_t_tracktrack.py::"
        "test_stage_s_registry_remains_valid_after_stage_t"
    ),
    (
        "tests/test_stage_u_1_presentation.py::"
        "test_frozen_stage_t_demo_hash_remains_unchanged"
    ),
    (
        "tests/test_stage_u_1_presentation.py::"
        "test_generated_presentation_metadata_has_required_claim_boundaries"
    ),
    (
        "tests/test_stage_u_p3_tt_runtime.py::"
        "test_stage_s_default_and_registry_remain_unchanged"
    ),
    (
        "tests/test_stage_u_portable_release.py::"
        "test_submission_audit_has_no_forbidden_bulk_artifacts"
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
        "tests/test_stage_u_portable_release.py::"
        "test_frozen_demo_hashes_are_unchanged"
    ),
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    trackeval_available = importlib.util.find_spec("trackeval") is not None
    portable_package = os.environ.get("STAGE_U_PORTABLE_PACKAGE") == "1"
    public_source = os.environ.get("PARKING_PUBLIC_SOURCE_PACKAGE") == "1"
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if nodeid.startswith("implementation/"):
            nodeid = nodeid.removeprefix("implementation/")
        if nodeid in OFFICIAL_TRACKEVAL_TESTS:
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
        if portable_package and nodeid in LOCAL_ONLY_ARTIFACT_TESTS:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "requires a local-only historical registry, frozen "
                        "runtime output, model weight, or dataset omitted from "
                        "the Stage U portable package"
                    )
                )
            )
        if public_source and nodeid in PUBLIC_SOURCE_ARTIFACT_TESTS:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "requires a frozen dataset binding, presentation "
                        "medium, or historical release hash chain omitted "
                        "from the public source repository"
                    )
                )
            )
