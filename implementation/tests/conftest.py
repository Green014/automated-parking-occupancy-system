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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    trackeval_available = importlib.util.find_spec("trackeval") is not None
    portable_package = os.environ.get("STAGE_U_PORTABLE_PACKAGE") == "1"
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
