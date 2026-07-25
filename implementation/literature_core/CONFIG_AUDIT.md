# Configuration and Result Audit

Audit date: 25 July 2026 (Asia/Shanghai)

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
