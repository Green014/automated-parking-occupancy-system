# Model Card: E1b MobileNetV3-CBAM Classifier

## Summary

E1b is the project's frozen parking-space classifier. It is a paper-inspired
local adaptation of torchvision MobileNetV3-Small: pretrained squeeze-and-
excitation blocks are retained and an identity-initialized CBAM supplement is
added. In the default system, F2 calls E1b only for detector-negative parking
slots; E1b does not override clear D1/B1 occupied evidence.

| Field | Value |
|---|---|
| Release asset | `E1b_CBAM_best.pt` |
| Architecture | torchvision MobileNetV3-Small with CBAM supplement |
| Frozen experiment ID | Not recorded in the historical training summary |
| Frozen artifact ID | `mobilenet_variant_ablation/cbam_supplement`, seed 20260725 |
| Parameters | 1,976,245 |
| Bytes | 8,045,704 |
| SHA-256 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |
| Frozen occupied threshold | 0.76 |
| Intended asset license | AGPL-3.0-only, subject to torchvision and dataset notices |

## Training data and provenance

The frozen run used ImageNet-pretrained MobileNetV3-Small initialization,
seed 20260725, a frozen backbone, and 900 PKLot parking-space patches
(449 vacant and 451 occupied). Development used 247 patches; 358 held-out
test patches were not loaded by the training loop. PKLot is published under
CC-BY-4.0. The checkpoint does not contain or redistribute PKLot images.

The selected checkpoint is
`mobilenet_variant_ablation/cbam_supplement/best.pt`. It is explicitly
paper-inspired and is not represented as an exact reproduction of another
paper. W.3 copies this frozen file byte-for-byte without loading or executing
it.

## Frozen evidence

The recorded internal static evaluation reports Macro F1 0.986033, occupied
recall 0.977901, and vacant recall 0.994350. The frozen CNRPark-EXT external
evaluation reports Macro F1 0.910801, occupied recall 0.843191, and vacant
recall 0.992433. These values apply only to the documented frozen protocols
and do not establish general performance.

In the integrated Stage S system, occupied recall is only 0.370927. That
complete-pipeline result reflects detector coverage, mapping, the asymmetric
gate, and domain/lighting effects; high internal classifier scores must not be
used to imply high end-to-end occupancy accuracy.

## Intended use

- research and teaching on fixed-camera parking-space patches;
- reproduction of the frozen E1b and D1/B1/E1b/F2 evaluations;
- offline comparison using independently authorized input data.

It is not validated for safety-critical control, enforcement, billing,
surveillance, or unattended deployment.

## Limitations

- the model expects rectified fixed-camera parking-space crops at the frozen
  preprocessing size and threshold;
- low light, glare, occlusion, camera shift, and different stall geometry can
  degrade occupied recall;
- PKLot and CNRPark-EXT evidence does not establish universal cross-scene
  generalization;
- class balance and fixed thresholds can hide site-specific failure modes;
- the end-to-end system remains limited by D1 detector-negative coverage;
- the checkpoint contains learned parameters only and no training data.

Users must preserve PKLot attribution and comply with the project,
torchvision/pretrained-weight, and dataset license boundaries.
