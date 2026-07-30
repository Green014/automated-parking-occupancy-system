from literature_core.error_analysis import (
    classify_ablation_record,
    summarize_ablation_records,
)


PARAMETERS = {
    "e1_threshold": 0.6,
    "e2_threshold": 0.1,
    "e3_classifier_weight": 0.5,
    "e3_detector_weight": 0.5,
    "e3_threshold": 0.4,
}


def test_frozen_error_analysis_identifies_fusion_rescue() -> None:
    row = classify_ablation_record(
        {
            "truth": 1,
            "p_cls": 0.8,
            "p_world": 0.05,
            "p_baseline": 0.0,
        },
        PARAMETERS,
    )
    assert row["predictions"] == {"E0": 0, "E1": 1, "E2": 0, "E3": 1}
    assert row["branch_pattern"] == "classifier_only_correct"
    assert row["fusion_rescued_branch_error"]
    assert row["error_signature"] == "E0,E2"


def test_summary_counts_fusion_harm_without_parameter_selection() -> None:
    rows = [
        classify_ablation_record(
            {
                "truth": 1,
                "p_cls": 0.65,
                "p_world": 0.11,
                "p_baseline": 0.2,
            },
            PARAMETERS,
        ),
        classify_ablation_record(
            {
                "truth": 0,
                "p_cls": 0.2,
                "p_world": 0.0,
                "p_baseline": 0.0,
            },
            PARAMETERS,
        ),
    ]
    summary = summarize_ablation_records(rows)
    assert summary["samples"] == 2
    assert summary["errors"]["E3"] == 1
    assert summary["fusion_harmed_both_correct"] == 1
    assert summary["error_signatures"]["none"] == 1
