from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from parking_occupancy.stage_i_evaluation import (
    StageIProtocolError,
    analyze_stage_i_v2_max_det_sensitivity,
    classify_detection_errors,
    load_count_test_protocol,
    load_stage_i_v2_posthoc_count_protocol,
    select_detector_and_count_rule,
    select_stage_i_v2_operating_points,
    verify_stage_i_record,
    verify_stage_i_v2_record,
)


def _write_comparison(root: Path, *, stage_i_v2: bool = False) -> None:
    models = {}
    confidences = {
        "D0": [[0.9], [0.2]],
        "D1": [[0.9, 0.8], [0.9]],
        "D2": [[0.9], [0.9, 0.1]],
    }
    metrics = {
        "D0": (0.3, 0.5, 30.0),
        "D1": (0.7, 0.8, 29.0),
        "D2": (0.5, 0.7, 10.0),
    }
    for method_id in ("D0", "D1", "D2"):
        method_root = root / method_id
        method_root.mkdir(parents=True)
        map_value, recall, fps = metrics[method_id]
        models[method_id] = {
            "name": method_id,
            "dataset_role": "consumed_development_validation",
            "images": 2,
            "map_50_95": map_value,
            "recall": recall,
            "framework_pipeline_fps": fps,
            "common_inference": (
                {"agnostic_nms": True}
                if stage_i_v2
                else {}
            ),
        }
        rows = []
        for index, scores in enumerate(confidences[method_id]):
            rows.append(
                json.dumps(
                    {
                        "image_name": f"{index}.jpg",
                        "ground_truth_box_count": index + 1,
                        "detections": [
                            {"confidence": score} for score in scores
                        ],
                    }
                )
            )
        (method_root / "detections.jsonl").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "comparison_protocol_id": (
                    "D-COMP-NDISPARK-DEV-V2-MAXDET300-20260727-01"
                    if stage_i_v2
                    else "D-COMP-NDISPARK-DEV-20260727-01"
                ),
                "dataset_role": "consumed_development_validation",
                "models": models,
            }
        ),
        encoding="utf-8",
    )


def test_selection_uses_primary_metrics_and_one_shared_count_rule(
    tmp_path: Path,
) -> None:
    comparison_root = tmp_path / "comparison"
    _write_comparison(comparison_root)
    output = tmp_path / "selection.json"

    report = select_detector_and_count_rule(
        comparison_root=comparison_root,
        output_path=output,
        candidate_thresholds=[0.15, 0.5],
    )

    assert report["detector_selection"]["selected_method_id"] == "D1"
    assert report["detector_selection"]["ranking"] == ["D1", "D2", "D0"]
    assert report["shared_count_rule"]["selected_confidence"] == 0.15
    assert report["shared_count_rule"]["applies_identically_to"] == [
        "D0",
        "D1",
        "D2",
    ]
    assert report["test_counts_read"] is False
    assert output.is_file()


def test_selection_rejects_different_development_membership(
    tmp_path: Path,
) -> None:
    comparison_root = tmp_path / "comparison"
    _write_comparison(comparison_root)
    d2_path = comparison_root / "D2" / "detections.jsonl"
    rows = d2_path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[1])
    changed["image_name"] = "different.jpg"
    rows[1] = json.dumps(changed)
    d2_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(StageIProtocolError, match="memberships"):
        select_detector_and_count_rule(
            comparison_root=comparison_root,
            output_path=tmp_path / "selection.json",
        )


def test_selection_refuses_overwrite(tmp_path: Path) -> None:
    comparison_root = tmp_path / "comparison"
    _write_comparison(comparison_root)
    output = tmp_path / "selection.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        select_detector_and_count_rule(
            comparison_root=comparison_root,
            output_path=output,
        )


def test_stage_i_v2_reports_both_threshold_regimes_and_input_hashes(
    tmp_path: Path,
) -> None:
    comparison_root = tmp_path / "comparison"
    _write_comparison(comparison_root, stage_i_v2=True)
    project_root = Path(__file__).resolve().parents[1]
    config = (
        project_root
        / "configs"
        / "detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml"
    )
    output = tmp_path / "selection_v2.json"

    report = select_stage_i_v2_operating_points(
        comparison_root=comparison_root,
        comparison_config=config,
        output_path=output,
    )

    assert report["data_role"] == "consumed_development_validation"
    assert report["test_labels_read"] is False
    assert report["detector_ranking"][
        "selected_before_posthoc_test"
    ] == "D1"
    common = report["common_threshold_diagnostic"]
    assert common["role"] == "controlled_sensitivity_only"
    assert common["deployment_selection_allowed"] is False
    assert common["selected_confidence"] == 0.10
    calibrated = report["per_model_development_calibration"]
    assert calibrated["role"] == "primary_deployment_operating_point"
    assert {
        method_id: calibrated["selected"][method_id]["confidence"]
        for method_id in ("D0", "D1", "D2")
    } == {"D0": 0.20, "D1": 0.90, "D2": 0.10}
    assert len(report["all_candidate_results"]) == 19
    assert all(
        set(row["per_model"]) == {"D0", "D1", "D2"}
        and all(
            {"mae", "rmse"}.issubset(row["per_model"][method_id])
            for method_id in ("D0", "D1", "D2")
        )
        for row in report["all_candidate_results"]
    )
    assert report["selected_at"]
    for artifact in (
        report["source_artifacts"]["comparison_config"],
        report["source_artifacts"]["comparison"],
        *report["source_artifacts"]["detections"].values(),
    ):
        assert artifact["bytes"] > 0
        assert len(artifact["sha256"]) == 64
    assert output.is_file()


def test_stage_i_v2_calibration_rejects_unrecorded_agnostic_nms(
    tmp_path: Path,
) -> None:
    comparison_root = tmp_path / "comparison"
    _write_comparison(comparison_root, stage_i_v2=True)
    comparison_path = comparison_root / "comparison.json"
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    del payload["models"]["D2"]["common_inference"]["agnostic_nms"]
    comparison_path.write_text(json.dumps(payload), encoding="utf-8")
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml"
    )

    with pytest.raises(StageIProtocolError, match="agnostic_nms"):
        select_stage_i_v2_operating_points(
            comparison_root=comparison_root,
            comparison_config=config,
            output_path=tmp_path / "selection_v2.json",
        )


def _write_max_det_comparison(
    root: Path,
    *,
    max_det: int,
    ranking: tuple[str, str, str] = ("D1", "D2", "D0"),
) -> None:
    metric_rank = {
        method_id: float(len(ranking) - ranking.index(method_id))
        for method_id in ranking
    }
    models = {}
    for method_id in ("D0", "D1", "D2"):
        score = metric_rank[method_id] / 10.0
        models[method_id] = {
            "common_inference": {
                "agnostic_nms": True,
                "max_detections": max_det,
            },
            "precision": score,
            "recall": score,
            "map_50": score,
            "map_50_95": score,
            "images_reaching_max_det": (
                30 if method_id == "D1" and max_det == 300 else 0
            ),
            "framework_pipeline_latency_ms_per_image": 10.0 + score,
            "framework_pipeline_fps": 90.0 - score,
            "peak_cuda_memory_allocated_bytes": 1000 + max_det,
            "peak_cuda_memory_reserved_bytes": 2000 + max_det,
        }
    root.mkdir(parents=True)
    (root / "comparison.json").write_text(
        json.dumps(
            {
                "comparison_protocol_id": (
                    "D-COMP-NDISPARK-DEV-V2-"
                    f"MAXDET{max_det}-20260727-01"
                ),
                "dataset_role": "consumed_development_validation",
                "models": models,
            }
        ),
        encoding="utf-8",
    )


def test_max_det_sensitivity_retains_300_when_ranking_is_unchanged(
    tmp_path: Path,
) -> None:
    root_300 = tmp_path / "max300"
    root_1000 = tmp_path / "max1000"
    _write_max_det_comparison(root_300, max_det=300)
    _write_max_det_comparison(root_1000, max_det=1000)

    report = analyze_stage_i_v2_max_det_sensitivity(
        max_det_300_root=root_300,
        max_det_1000_root=root_1000,
        output_path=tmp_path / "decision.json",
    )

    assert report["ranking_unchanged"] is True
    assert report["final_max_detections"] == 300
    assert report["test_labels_read"] is False
    assert report["per_model"]["D1"]["300"][
        "images_reaching_max_det"
    ] == 30


def test_max_det_sensitivity_blocks_silent_change_when_ranking_changes(
    tmp_path: Path,
) -> None:
    root_300 = tmp_path / "max300"
    root_1000 = tmp_path / "max1000"
    _write_max_det_comparison(root_300, max_det=300)
    _write_max_det_comparison(
        root_1000,
        max_det=1000,
        ranking=("D2", "D1", "D0"),
    )

    report = analyze_stage_i_v2_max_det_sensitivity(
        max_det_300_root=root_300,
        max_det_1000_root=root_1000,
        output_path=tmp_path / "decision.json",
    )

    assert report["ranking_unchanged"] is False
    assert report["final_max_detections"] is None
    assert report["status"] == (
        "blocked_pending_explicit_methodology_review"
    )


def test_committed_stage_i_v2_posthoc_protocol_is_consumed_and_frozen() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage_i_v2_posthoc_count_frozen_20260727.yaml"
    )

    payload = load_stage_i_v2_posthoc_count_protocol(config)

    assert payload["result_boundary"]["role"] == (
        "consumed_test_posthoc_sensitivity"
    )
    assert payload["common_inference"]["agnostic_nms"] is True
    assert payload["common_inference"]["max_detections"] == 300
    assert payload["operating_points"][
        "common_threshold_diagnostic"
    ]["deployment_selection_allowed"] is False
    assert payload["operating_points"][
        "per_model_development_calibration"
    ]["thresholds"] == {"D0": 0.10, "D1": 0.30, "D2": 0.10}
    assert payload["operating_points"][
        "test_labels_used_for_selection"
    ] is False


def test_stage_i_v2_posthoc_protocol_rejects_untouched_test_claim(
    tmp_path: Path,
) -> None:
    import yaml

    committed = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage_i_v2_posthoc_count_frozen_20260727.yaml"
    )
    payload = yaml.safe_load(committed.read_text(encoding="utf-8"))
    payload["result_boundary"]["role"] = "untouched_test"
    changed = tmp_path / "changed.yaml"
    changed.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(StageIProtocolError, match="post-hoc"):
        load_stage_i_v2_posthoc_count_protocol(changed)


def test_committed_count_protocol_is_frozen_before_test_predictions() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "stage_i_count_test_frozen_20260727.yaml"
    )

    payload = load_count_test_protocol(config)

    assert payload["selected_detector"]["method_id"] == "D1"
    assert payload["shared_count_rule"]["confidence_threshold"] == 0.10
    assert payload["shared_count_rule"]["per_model_thresholds"] == (
        "prohibited"
    )
    assert payload["count_test"]["vehicle_box_ground_truth"] == "unavailable"


def test_qualitative_error_matching_is_one_to_one() -> None:
    matched_predictions, matched_truth = classify_detection_errors(
        ground_truth_boxes=np.asarray([[0, 0, 10, 10]], dtype=np.float32),
        predicted_boxes=np.asarray(
            [[0, 0, 10, 10], [0, 0, 9, 9]],
            dtype=np.float32,
        ),
    )

    assert matched_predictions == {0}
    assert matched_truth == {0}


def test_stage_i_verifier_detects_changed_generated_artifact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    import hashlib
    import yaml

    record = tmp_path / "record.yaml"
    record.write_text(
        yaml.safe_dump(
            {
                "record_id": "D-STAGE-I-RECORD-NDISPARK-20260727-01",
                "artifacts": [
                    {
                        "role": "test",
                        "path": "artifact.json",
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(
                            artifact.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert verify_stage_i_record(
        record_path=record,
        implementation_root=tmp_path,
    )["passed"]
    artifact.write_text("changed\n", encoding="utf-8")
    assert not verify_stage_i_record(
        record_path=record,
        implementation_root=tmp_path,
    )["passed"]


def test_stage_i_record_matches_frozen_registry() -> None:
    import hashlib
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (
            project_root
            / "data"
            / "comparisons"
            / "STAGE_I_FROZEN_CHECKSUMS.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["registry_id"] == (
        "D-STAGE-I-FREEZE-NDISPARK-20260727-01"
    )
    for artifact in registry["artifacts"]:
        path = project_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            artifact["sha256"]
        )


def test_stage_i_v2_verifier_supports_source_and_external_roots(
    tmp_path: Path,
) -> None:
    import hashlib
    import yaml

    source_root = tmp_path / "source"
    external_root = tmp_path / "external"
    source_root.mkdir()
    external_root.mkdir()
    source_artifact = source_root / "config.yaml"
    external_artifact = external_root / "comparison.json"
    source_artifact.write_text("frozen: true\n", encoding="utf-8")
    external_artifact.write_text("{}\n", encoding="utf-8")
    record = tmp_path / "record.yaml"
    artifacts = []
    for role, root_name, path in (
        ("config", "source", source_artifact),
        ("comparison", "external", external_artifact),
    ):
        artifacts.append(
            {
                "role": role,
                "root": root_name,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    record.write_text(
        yaml.safe_dump(
            {
                "record_id": (
                    "D-STAGE-I-V2-RECORD-NDISPARK-20260727-01"
                ),
                "artifacts": artifacts,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = verify_stage_i_v2_record(
        record_path=record,
        source_root=source_root,
        external_root=external_root,
    )

    assert report["passed"] is True
    assert report["passed_count"] == 2
    external_artifact.write_text("changed\n", encoding="utf-8")
    assert verify_stage_i_v2_record(
        record_path=record,
        source_root=source_root,
        external_root=external_root,
    )["passed"] is False
