# Configuration and Result Audit

Audit date: 25 July 2026 (Asia/Shanghai)

## Immutable artifact verification

On 26 July the freeze manifest was expanded to cover the actual method
templates/protocols, all three PKLot camera splits, E0/E1a/E1b/E2 weights,
E1a/E1b training summaries, Fold A selected parameters, the two CNR-EXT input
artifacts, and the once-only metrics file. All 17 SHA-256 checks passed.

Traceability is explicit:

| Item | Frozen source |
|---|---|
| Training seed | E1a/E1b `training_summary.json`: 20260725 |
| Bootstrap seed | `external_holdout_frozen.yaml`: 20260725 |
| YOLO-World prompts | `car`, `truck`, `bus`, `motorcycle` |
| E0 mapping threshold | minimum slot coverage 0.40 |
| E1a/E1b thresholds | 0.61 / 0.76 |
| E2 evidence threshold | 0.08 |
| E3a weights/threshold | 0.50/0.50, threshold 0.37 |
| E3b parameters/threshold | hashed `proposed_fusion.yaml`, threshold 0.67 |

The verification report is a new ignored artifact at
`outputs/phase_a_freeze_audit_20260726_v3/verification.json`; no prior output
was overwritten.

The temporal configuration is separate. The compatibility-named
`configs/temporal_protocol_pending.yaml` now has `status: frozen` and is
experiment-ready. Artifact-level verification passed for video/truth paths,
SHA-256, decoded frame bounds, polygon bounds, full interval coverage, both
occupancy classes, transitions, and distinct `0502`/`0503` scenes.

`configs/temporal_e4_e5_frozen.yaml` was written before any VIRAT model output.
It fixes model/checkpoint hashes, E3b calibration, generic E4 EMA/hysteresis,
ByteTrack settings, E5 mapping/motion/dwell rules, 30 warm-up frames, and the
once-only holdout identity. No value was altered after development or holdout
results. The 11 configuration, truth, runner, and result artifacts in
`data/manifests/temporal_case_study_frozen_20260726.yaml` all passed hash and
size verification.

## Data-role correction

All 27 locally selected PKLot images have now contributed to method
development, model selection, error analysis, or the post-hoc camera
rotation. The historical `train`, `development`, and `test` field names are
retained inside existing output JSON/CSV files for schema compatibility, but
none of the three cameras is described as an untouched final test set.

The three-camera rotation is an internal, camera-grouped development study.
An external dataset that has not participated in model or threshold selection
is required for final evaluation.

## Configuration layers

`configs/default.yaml` is a generic runnable template, not a record of the
executed pilot:

| Parameter | Generic default | Executed Fold A / Grand Bassin |
|---|---:|---:|
| Classifier epochs | 12 | 4 |
| Classifier weight | 0.65 | 0.50 |
| Detector weight | 0.35 | 0.50 |
| Fusion threshold | 0.50 | 0.37 |
| Temporal ON threshold | 0.58 | 0.58 |
| Temporal OFF threshold | 0.42 | 0.42 |

The Fold A values came from
`outputs/pklot_ablation/selected_parameters.json`. The Grand Bassin command
used `configs/grand_bassin_frozen.yaml`, which copies those Fold A fusion
values and the pre-registered generic temporal values. Grand Bassin was not
used to select or repair any parameter.

Fold B and C used their own development-camera selections. Those per-fold
values are evidence of transfer instability and are not a single deployable
configuration.

## Raw-result reconciliation

The following report values were checked against their saved JSON/CSV:

- Fold A E0/E3 macro F1: 0.980446 / 0.980446;
- three-camera E0/E3 camera-equal mean macro F1:
  0.973121 / 0.989539;
- Grand Bassin temporal occupied recall: 0.336336;
- Grand Bassin raw-to-temporal state-change reduction: approximately 97.5%.

The Fold A fusion failure is attributed to probability-scale calibration and
threshold-transfer failure. The saved error overlap also shows complementary
cases in which each branch corrects examples missed by the other.
