# NDISPark Detector Evaluation

Run date: 24 July 2026

## Protocol

- official Zenodo archive `ndis_park.zip`;
- Open Data Commons Attribution License 1.0;
- downloaded bytes: 118,187,828;
- MD5 verified against Zenodo:
  `2825a2403794d233c278e2532d061359`;
- local SHA-256:
  `87ca20dfe5e5a5659a9a41e03724fdc38eed050de6ed6742995955fc0bd785c0`;
- 30 validation images with 725 manually annotated vehicle boxes;
- validation images are night-domain images according to the dataset authors;
- pretrained YOLOv8n, image size 1280, CUDA device 0;
- model class filter: COCO `car` (class 2);
- no fine-tuning.

The dataset test split provides counting truth only, not box truth. These are
therefore **validation** detection metrics, not test metrics.

## Results

| Metric | Result |
|---|---:|
| Box precision | 0.890 |
| Box recall | 0.677 |
| mAP@0.5 | 0.806 |
| mAP@0.5:0.95 | 0.415 |
| Preprocess | 6.98 ms/image |
| Inference | 14.17 ms/image |
| Postprocess | 2.15 ms/image |

The PR curve, confusion matrices, labelled batches, and predicted batches are
saved in `outputs/ndispark_val_yolov8n_final_1280/`. The visual batches show
that the night/domain-shift cases include missed distant or strongly shadowed
vehicles and occasional wrong COCO class predictions.

## Interpretation

These values describe vehicle bounding-box detection only. They must not be
quoted as slot occupancy F1, false-free rate, or transition performance.
Conversely, the PKLot/Grand Bassin slot metrics must not be called detector
mAP. The 0.677 detector recall independently supports the fine-tuning gate,
especially for dark and occluded vehicles.
