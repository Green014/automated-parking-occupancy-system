# Fine-tuning Gate Assessment

Decision date: 24 July 2026

Decision: **gate passed; annotation preparation started; training blocked
until manual correction is complete.**

## Evidence

1. Detector miss/severe localization caused 120 of 137 initial PKLot
   false-free errors (87.6%), far above the planned 10% gate.
2. Lowering YOLOv8n confidence from 0.20 to 0.025 improved recall from 0.819
   to 0.935 but left 49 B0 false-free errors.
3. YOLOv8s at confidence 0.10 produced lower recall (0.807) than YOLOv8n at
   the same threshold (0.889) and ran more slowly.
4. In the Grand Bassin bus region, one detection often covered only one of
   several adjacent buses; temporal filtering cannot create a vehicle box
   that the detector never produced.

## Prepared annotation package

- 72 frames, split by three complete `video_source` groups;
- 24 train, 24 validation, 24 untouched holdout;
- SHA-256 manifest for every image;
- COCO machine preannotations exported only as a labelling accelerator.

Training must not start until duplicate boxes, misses, localization errors, and
class errors are manually corrected and the annotation status is changed from
`needs_manual_review`. The holdout video must remain unavailable for
threshold selection.

## Planned fine-tuning comparison

- initialize from `yolov8n.pt`; do not train from scratch;
- compare pretrained versus fine-tuned YOLOv8n using the same image size,
  confidence-search protocol, slot polygons and mapping;
- report detector mAP only after manual box truth exists;
- report slot F1/false-free separately from detector mAP;
- retain the pretrained B0/B1/Proposed results even if fine-tuning fails.
