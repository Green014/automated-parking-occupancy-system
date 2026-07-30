# Licensed-data Smoke Test

Run date: 24 July 2026

## Purpose and limitations

This run proves that the code executes end to end on a real, openly licensed
parking image. It is not a final scientific comparison. One PKLot image was
encoded as ten repeated frames so that video decoding, YOLO inference, slot
mapping, ByteTrack, temporal filtering, logging, plotting, and output writing
could all be tested. Because the frames are repeated, this clip contains no
real arrivals or departures and cannot measure transition latency or tracking
identity quality.

The source image contains 100 annotated parking polygons: 68 occupied and
32 vacant. It is a fixed, high-angle camera view from PKLot, licensed CC BY
4.0. Its local SHA-256 is
`df6763f20814b490cdc87ae40d9d1892ddd9143252fe98878c987b9368e2215b`.

## Frozen smoke configuration

- YOLOv8n pretrained COCO weights, vehicle classes car/motorcycle/bus/truck;
- weight SHA-256
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`;
- confidence threshold 0.20, image size 1280, CUDA device 0;
- overlap threshold 0.30;
- Proposed ByteTrack settings in `configs/bytetrack_parking.yaml`;
- Proposed EMA parameters: rise 0.60, fall 0.15, occupied 0.18, vacant 0.06;
- one model-initialization frame excluded only from steady-state timing;
- three state-establishment frames excluded only from temporal flicker counts.

## Actual results

| Experiment | Precision | Recall | Slot F1 | Slot AP | False-free rate | Flickers after warm-up | Steady-state FPS | Detector p50 | Mapping/filter p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1.000 | 0.584 | 0.737 | 0.867 | 0.416 | 0 | 37.51 | 13.70 ms | 1.35 ms |
| B1 | 0.957 | 0.585 | 0.726 | 0.864 | 0.415 | 0 | 37.78 | 13.30 ms | 1.94 ms |
| Proposed | 0.976 | 0.488 | 0.651 | 0.847 | 0.512 | 0 | 30.55 | 19.20 ms | 1.62 ms |

Classification scores include all ten frames; warm-up exclusion applies only
to the flicker calculation. The Proposed system does not outperform B0 on this
static smoke sample. ByteTrack filtering and temporal start-up delay reduce
occupied recall. This negative result must be re-tested on genuine continuous
clips where false one-frame changes and transition latency can be measured.

An earlier 640-pixel B0 diagnostic produced only 0.100 occupied recall and
0.182 F1. Increasing inference size to 1280 raised recall to 0.584 without
training, confirming that tiny overhead vehicles are the current detector
bottleneck and that input-scale/model-size checks should precede fine-tuning.

## Produced artifacts

Each `outputs/final_smoke_*` directory contains `annotated.mp4`,
`occupancy.csv`, `events.csv`, `summary.json`, and an `evaluation/` directory
containing `metrics.json`, `confusion_matrix.png`, `pr_curve.png`, and
`errors.csv`. The steady-state FPS above includes OpenCV visualization and MP4
encoding. These generated artifacts are ignored by Git but remain in the local
workspace for inspection.
