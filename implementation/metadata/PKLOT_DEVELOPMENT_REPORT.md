# PKLot Development Experiment Report

Run date: 24 July 2026

## Protocol

- 27 full-resolution PKLot images;
- balanced across PUCPR, UFPR04, UFPR05 and sunny/cloudy/rainy conditions;
- low/mid/high occupancy targets with a distinct acquisition date in each
  source/weather/occupancy bucket;
- 1,505 known slot labels; seven `unknown` slots excluded;
- pretrained COCO YOLOv8 weights, no training;
- one shared detection cache per detector configuration, so B0 and B1 use
  exactly the same boxes;
- development data only: these values are for selection, not final test
  claims.

## Main results

| Detector/configuration | Mapping | Precision | Recall | Slot F1 | False-free |
|---|---|---:|---:|---:|---:|
| YOLOv8n, conf 0.20, 1280 | B0 centre | 0.997 | 0.819 | 0.899 | 0.181 |
| YOLOv8n, conf 0.20, 1280 | B1 overlap 0.30 | 0.987 | 0.822 | 0.897 | 0.178 |
| YOLOv8n, conf 0.10, 1280 | B0 centre | 0.993 | 0.889 | 0.938 | 0.111 |
| YOLOv8n, conf 0.05, 1280 | B0 centre | 0.992 | 0.927 | 0.958 | 0.073 |
| YOLOv8n, conf 0.025, 1280 | B0 centre | 0.992 | 0.935 | 0.963 | 0.065 |
| YOLOv8n, conf 0.025, 1280 | B1 overlap 0.40 | 0.981 | 0.947 | **0.964** | **0.053** |
| YOLOv8s, conf 0.10, 1280 | B0 centre | 1.000 | 0.807 | 0.893 | 0.193 |

The provisional development configuration is YOLOv8n at confidence 0.025,
image size 1280, with a 0.40 slot-coverage threshold for B1/Proposed. The
threshold is frozen here; it must not be re-tuned on the final test split.

## Overlap threshold sweep

At YOLOv8n confidence 0.025, B1 F1 increased from 0.929 at overlap 0.10 to
0.964 at 0.40, then declined to 0.957 at 0.50. A permissive overlap threshold
increased false occupied assignments; a strict threshold increased
false-free errors. The 0.40 setting was the development-set knee.

## Error source and model-size gate

At the initial confidence 0.20 setting, 120 of 137 B0 false-free errors
(87.6%) were manually classified as detector miss or severe localization.
Only 15 were centre-mapping errors and two were assignment conflicts. PUCPR
was the hardest source: B0 F1 was 0.830 with 131 false-free errors, versus
0.992 for UFPR04 and 0.983 for UFPR05.

YOLOv8s did not solve the missed-small-vehicle problem. In a warm GPU
benchmark at image size 1280:

| Model | Mean inference | p95 | Throughput | Detections on benchmark image |
|---|---:|---:|---:|---:|
| YOLOv8n | 12.85 ms | 14.11 ms | 77.82 FPS | 143 |
| YOLOv8s | 19.35 ms | 20.92 ms | 51.69 FPS | 124 |

The confidence sweep materially improved recall, but 49 B0 false-free errors
remained at confidence 0.025. The fine-tuning gate is therefore satisfied,
subject to manual correction of a video-level-split annotation set.

## Interpretation

YOLO supplies vehicle boxes only. Slot F1 changes when the same boxes are
converted through centre or polygon-overlap mapping, proving that detector
metrics and occupancy metrics are not interchangeable. The small B1 gain at
the frozen setting is real but development-only; the stronger result is that
tiny overhead vehicle recall dominates the remaining errors.
