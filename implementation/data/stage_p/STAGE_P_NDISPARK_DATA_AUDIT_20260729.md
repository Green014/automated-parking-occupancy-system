# Stage P NDISPark Data Audit

Date: 29 July 2026

Protocol: `STAGE-P-PARKING-DOMAIN-RETENTION-20260729-01`

## Local holdings and truth

The local frozen NDISPark 1.0 preparation is complete and matches the
existing manifests:

| Official split | Lighting | Images | Available truth | Historical role |
|---|---|---:|---|---|
| train | daylight | 112 | 2,577 vehicle boxes | D1 training and D1-LL parking-domain mix |
| validation | night | 30 | 725 vehicle boxes | consumed development validation |
| test | night | 117 | image counts totalling 1,402 vehicles | consumed post-hoc count diagnostic |

The prepared tree has 112/30/117 images in train/validation/test, 112/30/0
YOLO label files, and exactly 2,577/725/0 non-empty label rows. All box truth
uses source COCO category 3 `car`, mapped to project class 0 `vehicle`.
NDISPark publishes no ignore-region field for these annotations. Stage P
therefore applies no ground-truth-derived ignored-class suppression to either
model.

NDISPark contains neither parking-slot polygons nor per-slot
occupied/vacant truth. It cannot support B1 mapping, slot Macro F1, event
accuracy, or transition latency evaluation.

## Consumption and leakage boundary

- The 112 daylight images trained D1 and were also mixed into D1-LL training.
  Daylight box metrics are training-resubstitution diagnostics only.
- The 30 night validation images were used by D1 training/model evidence and
  Stage I detector/threshold analysis. They are consumed-development data.
- The 117 night test counts were viewed in Stage I and Stage I-v2. They are
  consumed post-hoc count diagnostics and have no box truth.
- Six cameras span train, validation and test; this is a day-to-night
  condition split, not camera-independent generalization.
- Frozen image manifests report zero exact image duplicates across splits.

## Metric conflict and Stage P resolution

Historical NDISPark detector comparison constructed AP curves from a
confidence floor of 0.001. Stage P is required to compare D1 and D1-LL at the
deployed Stage O/P3 threshold 0.30. It therefore reports:

> confidence-truncated AP at the frozen confidence threshold of 0.30

This is not standard COCO AP and is not directly comparable with the older
0.001-floor table or paper-reported standard AP.

## Source and license

Dataset: NDISPark version 1.0, DOI `10.5281/zenodo.6560823`, licensed under
ODC-By 1.0. The frozen 118,187,828-byte source archive has SHA-256
`87ca20dfe5e5a5659a9a41e03724fdc38eed050de6ed6742995955fc0bd785c0`.
Original source and prepared files remain unmodified.
