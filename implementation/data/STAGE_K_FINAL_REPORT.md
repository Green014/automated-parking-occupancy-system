# Stage K final evaluation and project closure report

Updated: 28 July 2026

## Purpose

Stage K evaluates the frozen detector-to-slot pipelines on a previously
unpredicted PKLot subset. It closes the static slot-occupancy comparison; it
does not turn sampled images into tracking evidence and it does not authorize
test-driven model reselection.

## Data gate

The first local-inventory gate was blocked and is retained as historical
evidence. The v2 gate later admitted 90 complete JPG/XML pairs recovered from
the partial official PKLot archive:

- 30 PUCPR sunny images from 2012-09-15;
- 30 UFPR04 cloudy images from 2013-01-16;
- 30 UFPR05 sunny images from 2013-04-14;
- 90 unique image SHA-256 values;
- zero overlap with the 27 Stage J development images;
- 5,034 known slot labels: 1,943 occupied and 3,091 vacant;
- six unknown labels retained and excluded from metrics.

Membership within each group was selected by timestamp-sorted evenly spaced
sampling, not by model output. The raw and truth-overlay contact sheets were
reviewed before protocol freeze. The v2 gate is
`comparisons/stage_k_slot_occupancy_data_gate_20260728_v2.yaml`.

## Frozen methods

| Pipeline | Detector | Confidence | Common mapping |
|---|---|---:|---|
| P0 | COCO-pretrained YOLOv8n | 0.10 | B1 polygon coverage 0.40, one-to-one |
| P1 | NDISPark-fine-tuned YOLOv8n | 0.30 | B1 polygon coverage 0.40, one-to-one |
| P2 | zero-shot YOLO-World | 0.10 | B1 polygon coverage 0.40, one-to-one |

All methods used image size 640, class-agnostic NMS, IoU 0.7,
`max_det=300`, no augmentation and no temporal stabilization.

## Overall result

| Pipeline | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied | Slot AP |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.785612 | 0.544519 | 0.992883 | 0.455481 | 0.007117 | 0.718263 |
| P1 | **0.808398** | **0.598044** | 0.984148 | **0.401956** | 0.015852 | **0.739705** |
| P2 | 0.796548 | 0.559444 | **0.997735** | 0.440556 | **0.002265** | 0.729244 |

P1 improves the pooled Macro F1 and occupied recall, but still marks 781 of
1,943 occupied slots as free. This false-free rate is operationally
significant.

## Grouped result

| Pipeline | Camera-macro F1 | PUCPr | UFPR04 | UFPR05 |
|---|---:|---:|---:|---:|
| P0 | 0.841099 | 0.576628 | **0.998367** | 0.948304 |
| P1 | 0.824835 | **0.735789** | 0.871630 | 0.867086 |
| P2 | **0.849168** | 0.601529 | 0.991748 | **0.954226** |

The 90-image paired analysis reaches a different conclusion from the pooled
slot total:

| Comparison | Win/tie/loss | Mean paired delta | 95% paired bootstrap CI |
|---|---:|---:|---:|
| P1 - P0 | 17 / 11 / 62 | -0.040322 | [-0.068804, -0.009457] |
| P2 - P0 | 33 / 34 / 23 | 0.035781 | [0.009939, 0.062788] |

PUCPR contributes 2,996 of the 5,034 labels and therefore dominates the
pooled result. P1's pooled lead is real for that weighting, but it is not an
across-camera improvement. D1 remains the pre-test selected detector in the
provenance record; the test was not used to replace it with P0 or P2.

## Date and weather layers

The read-only post-hoc analysis reports all methods by date and weather.
Because each camera contributes one date, date and camera are confounded.
Cloudy weather occurs only in UFPR04, so the cloudy/sunny comparison is also
camera-confounded. These tables support error description, not a causal
weather-robustness claim.

The analysis registry is
`comparisons/stage_k_posthoc_stratified_analysis_20260728.yaml`.

## Runtime

Run-inclusive means contain lazy initialization and are retained without
rewriting. Median frame latency is 74.334 ms for P0, 73.014 ms for P1 and
76.220 ms for P2. Stage I-v2 remains the controlled detector-runtime
comparison.

## Continuous-video boundary

The separate P1+B1 VIRAT 0502 case is real continuous video but consumed
development evidence. Raw and hysteresis variants both missed the single
departure and produced Macro F1 0.456072. This negative case demonstrates the
video/event interface and shows that temporal smoothing cannot repair a
persistent detector-to-polygon error. It does not establish successful
tracking generalization.

## Reproducibility and conclusion

- Stage K static result artifacts: 43/43 verified.
- Stage K date/weather analysis artifacts: 9/9 verified.
- Stage K data-gate v2 evidence: 11/11 verified.
- Predictions were not rerun during closure.
- No threshold, checkpoint, mapping rule or detector was changed from test
  output.

The project demonstrates a complete, modular computer-vision workflow:
pretrained detector comparison, parking-domain fine-tuning, polygon-based
slot assignment, controlled ablation, slot-level evaluation, qualitative
outputs and a continuous-video failure case. Its main limitation is
cross-camera instability, especially false-free errors, rather than missing
software functionality.
