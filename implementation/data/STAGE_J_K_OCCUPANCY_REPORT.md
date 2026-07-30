# Stage J/K detector-to-slot occupancy report

Updated: 28 July 2026  
Stage J protocol: `P-COMP-PKLOT-DEV-STAGEJ-20260727-01`  
Stage K protocol: `P-COMP-PKLOT-TEST-STAGEK-20260727-01`  
Stage K gate v2: `STAGE-K-PKLOT-DATA-GATE-20260728-02`

## Scope and claim boundary

P0, P1 and P2 were connected to the same B1 polygon-coverage mapping:

- P0 = D0 COCO-pretrained YOLOv8n + B1;
- P1 = D1 NDISPark-fine-tuned YOLOv8n + B1;
- P2 = D2 zero-shot YOLO-World + B1.

The frozen comparison used `imgsz=640`, class-agnostic NMS, `max_det=300`,
one-to-one assignment and minimum slot coverage 0.40. Detector thresholds
were frozen before slot inference from Stage I-v2 development calibration:
0.10 for P0, 0.30 for P1 and 0.10 for P2. No temporal stabilization,
classifier fusion, test-based tuning or D1 retraining was performed.

All 27 PKLot images were already consumed by earlier development work.
Therefore these results are **consumed-development diagnostics**, not a new
untouched test. The montage videos contain non-contiguous images and are not
temporal evidence; their `events.csv` files intentionally contain only the
header.

## Stage J result

The preflight verified 27 image hashes, 1,505 known slot labels (757 occupied,
748 vacant), seven excluded unknown labels, and all three weight hashes.

| Pipeline | Detector | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied | Slot AP |
|---|---|---:|---:|---:|---:|---:|---:|
| P0 | D0 | 0.768040 | 0.566711 | 0.991979 | 0.433289 | 0.008021 | 0.784329 |
| P1 | D1 | **0.825723** | **0.671070** | 0.990642 | **0.328930** | 0.009358 | **0.835115** |
| P2 | D2 | 0.735168 | 0.504624 | **1.000000** | 0.495376 | **0.000000** | 0.753793 |

P1 led this fixed development comparison, but still missed 249 of 757
occupied slots. P2 produced no false-occupied slots but missed 375 occupied
slots. P0's largest weakness was PUCPR occupied recall (0.286031), showing
that detector-to-polygon transfer remains strongly camera dependent.

The model selection remains D1 because it was selected on detector
development evidence before this slot comparison. Stage J did not trigger
model reselection or parameter changes.

The additive image-grouped post-hoc analysis shows why the pooled result must
be qualified. P1 versus P0 was 6 wins, 2 ties and 19 losses across the 27
images; its mean paired image Macro-F1 difference was -0.020935 and its 95%
paired bootstrap interval was [-0.080493, 0.044616]. P1's pooled improvement
was driven mainly by PUCPR and was not confirmed as a camera-stable gain.

## Runtime interpretation

The run-inclusive mean includes lazy model loading, first CUDA context setup
and video rendering. It is therefore unsuitable for ranking the models in
this sequential run. The median per-frame end-to-end values better describe
steady processing after initialization:

| Pipeline | Run-inclusive FPS | Median frame latency (ms) | Derived median-frame FPS |
|---|---:|---:|---:|
| P0 | 0.869 | 34.105 | 29.321 |
| P1 | 27.127 | 31.974 | 31.275 |
| P2 | 4.768 | 32.668 | 30.611 |

This runtime caveat is retained rather than rerunning or rewriting the
completed output. Stage I-v2 remains the controlled detector-runtime source.

## Output and verification

Each method produced `annotated.mp4`, `occupancy.csv`, `events.csv`,
`detections.jsonl`, `summary.json`, `metrics.json` and
`runtime_metadata.json`. Each video opens and contains 27 frames; each
occupancy CSV contains 1,512 rows, including the seven rows whose truth is
blank and excluded from metrics; each detections log contains 27 records.

The source protocol, annotations, manifest, preflight, comparison and 21
required per-method artifacts are frozen in
`comparisons/stage_j_p0_p1_p2_development_20260727.yaml`.

## Stage K data gate

The original gate `STAGE-K-PKLOT-DATA-GATE-20260727-01` correctly recorded
that the first local inventory contained no eligible unseen data. It is
preserved as a historical blocker. Additional complete JPG/XML pairs were
later recovered from the partial official PKLot archive, so the additive v2
gate supersedes that inventory decision without rewriting it.

Before prediction, 90 images were selected by timestamp-sorted, evenly spaced
sampling: 30 each from one previously unused date for PUCPR, UFPR04 and
UFPR05. The manifest contains 90 unique image hashes with zero overlap against
the 27 Stage J development hashes. Official XML supplies 5,034 known slot
labels (1,943 occupied and 3,091 vacant); six unknown labels are retained but
excluded from metrics. Raw and truth-overlay contact sheets were manually
reviewed before the test protocol was frozen.

The v2 gate is recorded in
`comparisons/stage_k_slot_occupancy_data_gate_20260728_v2.yaml`. Its 11 source
and external evidence bindings are independently verifiable. The local
official archive is truncated, so only complete, individually hashed JPG/XML
pairs before the stream boundary were eligible.

## Stage K untouched-test result

The three methods used the same frozen inference and B1 mapping settings as
Stage J. No Stage K result changed a threshold, detector, mapping rule or
checkpoint.

| Pipeline | Macro F1 | Occupied recall | Vacant recall | False-free | False-occupied | Slot AP |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.785612 | 0.544519 | 0.992883 | 0.455481 | 0.007117 | 0.718263 |
| P1 | **0.808398** | **0.598044** | 0.984148 | **0.401956** | 0.015852 | **0.739705** |
| P2 | 0.796548 | 0.559444 | **0.997735** | 0.440556 | **0.002265** | 0.729244 |

P1 has the highest pooled Macro F1, but this is not a general superiority
result. Camera-macro F1 is 0.841099 for P0, 0.824835 for P1 and 0.849168 for
P2. P1 improves PUCPR but is lower than P0 on UFPR04 and UFPR05. Across 90
images, P1 versus P0 is 17 wins, 11 ties and 62 losses; the paired mean
difference is -0.040322 with 95% CI [-0.068804, -0.009457]. P2 versus P0 is
33 wins, 34 ties and 23 losses, with paired mean 0.035781 and 95% CI
[0.009939, 0.062788].

D1/P1 remains the pre-test selected candidate for provenance purposes, but
Stage K does not justify presenting it as an across-camera improvement. Test
results were not used for post-hoc detector reselection.

## Stratified interpretation

Date and weather tables were computed from the frozen occupancy CSV files
without prediction or parameter selection. Each camera contributes exactly
one date, so date and camera strata are numerically identical. Cloudy weather
occurs only for UFPR04; weather and camera are therefore confounded. The
stratified tables are descriptive condition slices, not independent evidence
of date or weather generalization.

The read-only analysis and its input bindings are frozen in
`comparisons/stage_k_posthoc_stratified_analysis_20260728.yaml`.

## Final verification boundary

The Stage K result registry binds 43/43 protocol, data, visual-review,
prediction, metric, error-analysis and visualization artifacts. The
stratified registry binds 9/9 inputs and outputs, and the data-gate v2 registry
binds 11/11 items. The 90-frame annotated files remain sampled-image montages,
not temporal evidence, and their event files are intentionally header-only.
