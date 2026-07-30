# Stage N-v3 emitted-box evaluation correction

Protocol: `STAGE-N-V3-EMITTED-BOX-CORRECTION-20260729-01`

Status: complete offline correction. Stage N-v2 remains unchanged.

## Scope

Stage N-v3 corrects a local emitted-box matching bug. In v2, a
confidence-ordered prediction was marked false positive when its highest-IoU
GT had already been assigned, even if another unused GT in the same frame
still exceeded the IoU threshold. The corrected matcher chooses the
highest-IoU unused GT independently at every threshold from 0.50 through
0.95.

This is an offline metric-only run over the 16 saved Stage N-v2 detections
JSONL files and four original LMOT validation GT files. Runtime metadata
records `inference_performed=false`, `model_loaded=false`,
`model_track_called=false`, `training_performed=false`, and
`trackeval_called=false`.

## Aggregate definition

The v2 main detection table was an unweighted mean of four per-sequence
metrics and also averaged count fields, yielding values such as 17,221.75 GT
boxes. Stage N-v3 makes the all-data pooled/micro result the primary table:
sequences receive isolated frame keys, all predictions share one confidence
ordering, and GT/prediction/TP/FP/FN counts are summed. Unweighted
per-sequence macro rates remain a secondary diagnostic.

These are emitted-box metrics from the saved complete `model.track(...)`
paths after excluded-class suppression. They are not raw detector-only
metrics.

## Corrected primary all-data aggregate

| Method | Precision | Recall | AP50 | AP50-95 | GT | Pred | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.770460 | 0.321294 | 0.292551 | 0.123845 | 68,887 | 28,727 | 22,133 | 6,594 | 46,754 |
| L1 | 0.847934 | 0.240974 | 0.236840 | 0.112470 | 68,887 | 19,577 | 16,600 | 2,977 | 52,287 |
| L2 | 0.751206 | 0.031646 | 0.034695 | 0.015054 | 68,887 | 2,902 | 2,180 | 722 | 66,707 |
| L3 | 0.732517 | 0.032996 | 0.034842 | 0.017454 | 68,887 | 3,103 | 2,273 | 830 | 66,614 |

The pooled/micro table must not be compared directly with the old displayed
macro table as though the difference were entirely caused by matching. It
also changes the aggregation population and confidence ordering.

## Like-for-like matcher correction

The following comparison holds aggregation constant as an unweighted
per-sequence macro, isolating the matching-code correction:

| Method | Metric | v2 recorded | v3 corrected | Delta |
|---|---|---:|---:|---:|
| L0 | Precision | 0.624199 | 0.624476 | +0.000277 |
| L0 | Recall | 0.228430 | 0.228530 | +0.000100 |
| L0 | AP50 | 0.192074 | 0.192106 | +0.000033 |
| L0 | AP50-95 | 0.082249 | 0.082253 | +0.000004 |
| L1 | Precision | 0.703726 | 0.703855 | +0.000129 |
| L1 | Recall | 0.185413 | 0.185463 | +0.000050 |
| L1 | AP50 | 0.169338 | 0.169352 | +0.000014 |
| L1 | AP50-95 | 0.080229 | 0.080231 | +0.000001 |
| L2 | Precision | 0.717658 | 0.717658 | 0.000000 |
| L2 | Recall | 0.042119 | 0.042119 | 0.000000 |
| L2 | AP50 | 0.038138 | 0.038138 | 0.000000 |
| L2 | AP50-95 | 0.017766 | 0.017766 | 0.000000 |
| L3 | Precision | 0.701533 | 0.701891 | +0.000358 |
| L3 | Recall | 0.043922 | 0.043952 | +0.000030 |
| L3 | AP50 | 0.040178 | 0.040178 | 0.000000 |
| L3 | AP50-95 | 0.020961 | 0.020961 | 0.000000 |

## TrackEval boundary

Official HOTA, DetA, AssA, IDF1, MOTA, and ID switches are neither
recomputed nor rewritten. They were produced by official TrackEval from the
saved track files using independent matching code, so the local emitted-box
AP/precision/recall error does not affect them.

LMOT still contains no parking-slot polygons or occupied/vacant truth. This
correction supports only the existing low-light tracking robustness
diagnostic and cannot be described as parking-slot occupancy performance.
