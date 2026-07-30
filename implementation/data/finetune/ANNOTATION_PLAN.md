# YOLOv8 Fine-tuning Annotation Gate

Status: **prepared, not approved for training**

The pretrained baseline now satisfies the coursework plan's fine-tuning gate:

1. manual failure review found that detector miss/severe localization caused
   120 of 137 false-free errors in the initial PKLot development run; and
2. YOLOv8s and confidence-threshold sweeps did not remove the problem without
   a recall/runtime trade-off.

The 72-frame preparation manifest uses three whole, disjoint Grand Bassin
video sequences:

| Split | Sequence | Frames selected | Purpose |
|---|---|---:|---|
| train | `13-30-33.flv` | 24 | dense cars, buses and moving vehicles |
| validation | `16-34-07.264` | 24 | frozen threshold/model selection |
| holdout | `16-25-12.264` | 24 | untouched final detector check |

Adjacent frames are not randomly distributed across splits. The supplied COCO
file contains **machine preannotations only**. Before any training:

- remove duplicate sliced detections;
- correct missed and badly localized cars, buses, trucks and motorcycles;
- exclude people and infrastructure;
- mark every row `manually_verified`;
- run an independent visual audit on at least 10% of the boxes;
- preserve the holdout sequence without parameter tuning.

Do not train from `grand_bassin_vehicle_preannotations.json` in its current
state. It is an annotation accelerator, not ground truth.
