# Model Card: D1 NDISPark Detector

## Summary

D1 is the project's frozen YOLOv8n vehicle detector. In the default parking
system it supplies detections to B1 one-to-one polygon mapping; E1b/F2 can
review detector-negative slots. It is not a complete occupancy system by
itself.

| Field | Value |
|---|---|
| Release asset | `D1_NDISPark_best.pt` |
| Architecture | Ultralytics YOLOv8n |
| Frozen experiment ID | `D1-NDISPARK-FT-20260727-01` |
| Bytes | 6,255,409 |
| SHA-256 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| Intended asset license | AGPL-3.0-only, subject to the Ultralytics boundary in `THIRD_PARTY_NOTICES.md` |

## Training data and provenance

D1 was fine-tuned from the YOLOv8n workflow on the frozen NDISPark training
split: 112 images and 2,577 annotated boxes. Development validation used 30
images and 725 boxes. The frozen dataset audit records NDISPark Zenodo record
6560823 as ODC-By-1.0. The released checkpoint does not contain or redistribute
the training images.

The formal training used the fixed project experiment above. W.3 copies the
already frozen checkpoint byte-for-byte; it does not train, fine-tune, load,
or execute the model.

## Frozen evidence

On the frozen NDISPark development validation, the recorded detector metrics
were precision 0.88153, recall 0.84160, mAP@0.5 0.89910, and
mAP@0.5:0.95 0.64969. These are detector-development measurements, not a claim
of parking-slot occupancy accuracy or broad cross-camera generalization.

In the final compatible Stage S integrated external low-light attribution,
the D1-based system recorded Macro F1 0.706681 and occupied recall 0.370927.
Those numbers measure the complete fixed pipeline on that frozen evaluation,
not standalone D1 performance. The low occupied recall is a material
limitation.

## Intended use

- research and teaching on fixed-camera parking occupancy;
- reproduction of the frozen D1/B1/E1b/F2 workflows when the exact asset hash
  and formal configuration are verified;
- offline comparison under documented dataset and privacy permissions.

It is not validated for safety-critical control, enforcement, billing,
surveillance, autonomous driving, or unattended deployment.

## Limitations

- low light, glare, occlusion, small vehicles, and camera/domain shift can
  reduce detection coverage;
- the frozen evidence is limited to the documented datasets and splits;
- cross-scene performance is not guaranteed;
- the final system's occupied recall remains weak on the reported external
  low-light evaluation;
- threshold, polygon, and downstream fusion choices materially affect
  occupancy output;
- the checkpoint contains learned parameters only and does not include
  training data, annotations, or a claim of universal vehicle recognition.

Users must preserve NDISPark attribution and comply with the project,
Ultralytics, and dataset license boundaries.
