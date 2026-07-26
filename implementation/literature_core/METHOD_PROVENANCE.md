# Method Provenance

All provenance labels describe this repository's implementation, not merely
the paper category.

| Module | Part I source | Code/weights source | License | Project change | Status |
|---|---|---|---|---|---|
| E1a slot patch classification | Yuldashev et al. [2] | torchvision `mobilenet_v3_small`; ImageNet weights from PyTorch | BSD-style PyTorch/torchvision terms; ImageNet weights subject to their dataset terms | OpenCV perspective-normalized 224x224 slot patch, standard MobileNetV3-Small, two-class head, ImageNet transfer learning | **Adaptation**, not exact reproduction |
| E1b paper-inspired classification | Yuldashev et al. [2] for LeakyReLU6/CBAM/BSConv direction; Woo et al. for CBAM | Local PyTorch modules layered on torchvision | Project code plus the runtime terms above | Identity-initialized channel/spatial CBAM supplements pretrained SE; shallow capped LeakyReLU6 is separately switchable; BSConv is not implemented | **Paper-inspired local adaptation**, not exact reproduction |
| Open-vocabulary detection | Cheng et al., YOLO-World [6] | Original: `AILab-CVC/YOLO-World`; runtime adapter: Ultralytics `YOLOWorld`; `yolov8s-worldv2.pt` | Original repository GPL-3.0; Ultralytics package/weights AGPL-3.0 for this academic workflow | Prompts restricted to `car`, `truck`, `bus`, `motorcycle`; raw box and confidence retained | **Adaptation** |
| Detection-to-slot mapping | APSD-OC [1] supports linking detections and slot geometry; baseline provides an existing OpenCV implementation | Implemented locally with `cv2.intersectConvexConvex` | Project code | Confidence-weighted fraction of each slot polygon covered by a prompted object box, with one-to-one greedy assignment | **Inspired/local integration** |
| E3a weighted evidence fusion | Combination motivated by [2] and [6] | Implemented locally | Project code | Development-selected normalized weighted sum of raw `P_cls`, detector evidence, and optional `P_track`; every component is logged | **Local system integration / baseline** |
| E3b calibrated logistic fusion | Probability calibration literature; branch choices motivated by [2] and [6] | Implemented locally with deterministic NumPy IRLS | Project code | Separate monotonic Platt-style branch calibration followed by non-negative logistic fusion of calibrated log-odds; camera-grouped fitting only | **Local innovation / proposed** |
| Temporal hysteresis | Existing baseline; tracking motivation from TrackTrack [7] | Reimplemented in the independent package | Project code | Asymmetric EMA with separate occupied/vacant thresholds; delayed true transitions are separated from ordinary flicker | **Adaptation/reuse** |
| E5 track-aware gate | Tracking motivation from TrackTrack [7], but no code copied | Local project code plus Ultralytics ByteTrack runtime | Project code; Ultralytics runtime terms apply | YOLOv8 detections receive ByteTrack IDs, then deterministic one-to-one slot assignment, stationary-motion suppression, and occupied/vacant dwell; synthetic tests precede a frozen negative case study | **Local paper-inspired adaptation**, not TrackTrack reproduction |
| Automatic slot discovery | APSD-OC [1] | No official implementation confirmed | Not applicable | No homography/DBSCAN reproduction is claimed | **Not implemented / future work** |
| External holdout data | Amato et al., CNRPark+EXT | Official `cnrpark.it` page and `fabiocarrara/deep-parking` GitHub release | ODbL-1.0 stated by the official page | CNR-EXT occupancy metadata joined to official camera boxes and released full frames; boxes are scaled, not converted into claimed precise polygons | **Third-party data, once-only external evaluation** |
| Conditional temporal data | Oh et al., VIRAT | Official `viratdata.org` and Kitware collection | VIRAT individual Usage Agreement; restricted redistribution and PII duties | User acceptance is recorded; 26 official clips were checksum-verified; two distinct-scene single-slot departure truths are project-created and support a limited frozen case study | **Third-party data with local truth; not a native VIRAT occupancy benchmark** |

## Code and runtime addresses

- torchvision MobileNetV3-Small API:
  https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html
- YOLO-World authors' repository:
  https://github.com/AILab-CVC/YOLO-World
- Ultralytics YOLO/YOLO-World runtime reference:
  https://docs.ultralytics.com/reference/models/yolo/model/
- APSD-OC and the Improved MobileNetV3 paper: no official implementation was
  confirmed during this audit, so no unofficial repository is presented as
  authoritative.
- TrackTrack: no implementation was copied or reproduced. It only motivates
  the E5 tracking direction; the implemented gate is a simpler local rule
  system and is reported under that narrower label.
- VIRAT Ground Release 2.0:
  https://viratdata.org/
- VIRAT Usage Agreement:
  https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf
- Dragon Lake Parking candidate:
  https://sites.google.com/berkeley.edu/dlp-dataset

## Temporal data provenance boundary

The user personally accepted the VIRAT agreement on 26 July 2026. The project
then downloaded 26 unmodified official video items (1,605,720,653 video bytes)
for bounded screening. The targeted `0503` extension also downloaded the
official event/mapping/object files for five clips. Item IDs, SHA-256 values,
video properties, event-selection evidence, and negative decisions are
recorded in `data/manifests/virat_screening_20260726.yaml` and
`data/manifests/virat_0503_targeted_screening_20260726.yaml`.

The official Release 2.0 document defines filename digits `XXYY` as the
physical scene and `ZZ` as the sequence. The local acquisition helper enforces
that grouping, explicit byte budgets, recorded agreement acceptance, atomic
non-overwriting downloads, bounded retry of transient 429/5xx/network errors,
and checksum verification. Those safeguards and the parking-suitability
decisions are project adaptations, not VIRAT methods or labels.

DLP is deferred because the raw video requires a request and drone motion may
break fixed ROIs. EPFL's current official page provides only non-consecutive
ground-truth archives, ISLab-PVD lacks an explicit dataset license, and LMOT
lacks parking-slot truth and a released licensed dataset.

The two eligible VIRAT clips have project-created truth. `0502` development
uses frame 1660 as first vacant; `0503` holdout uses frame 1550. Both contain a
fixed polygon and half-open occupied/vacant intervals. The holdout was locked
before any model output. Polygon coordinates, interval labels, transition
adjudication, scene-level split, track-to-slot assignment, motion threshold,
and dwell rules are local project adaptations; none is a native VIRAT label or
a Part I paper implementation.

The frozen E4/E5 run is explicitly a two-video, one-slot-per-video departure
case study. E5 did not beat E0 on holdout and failed to recognize vacancy on
development, so it cannot support a general tracking-improvement claim.

## Frozen runtime artifacts

| Artifact | Local role | SHA-256 |
|---|---|---|
| `mobilenet_v3_small-047dcff4.pth` | torchvision ImageNet initialization | `047DCFF4ADDEF86EA5BC2EFF13C9614DC11F47AB1160D0A71A25E7DB994F4E1F` |
| `yolov8s-worldv2.pt` | Ultralytics YOLO-World detector | `9B2C17AB6124A913E9B3A5C170617920D91B0F01111A8479DA69F00E2CF27792` |
| `ViT-B-32.pt` | CLIP text encoder cache used by the adapter | `40D365715913C9DA98579312B702A82C18BE219CC2A73407C4526F58EBA950AF` |

The generated two-class `best.pt` checkpoint is an experiment output rather
than a third-party artifact. Its SHA-256 is
`B67A8A5217902601FDF7BCEDF64F739AB03E688AFE86A386E56106F9F968651E`;
its training settings and data split are stored inside the checkpoint and
repeated in `outputs/mobilenet_pilot/training_summary.json`.

Post-hoc camera-rotation classifier checkpoints:

- Fold B SHA-256:
  `5E8903987E2225EAE301ECC19DE6260891CCD33CFF70D532C38B708747A5932C`;
- Fold C SHA-256:
  `92110D18DB5587AF0C6766219BDD5A8B4C15920E37A7D186E5885CB8CD5697E3`.

Their split membership, best epoch, test-not-loaded count, and training
settings are stored in the corresponding
`outputs/cross_camera/*_classifier/training_summary.json` files.

The selected architecture-ablation checkpoints are also local outputs:

- E1a standard:
  `outputs/mobilenet_variant_ablation/standard/best.pt`,
  SHA-256 `600C90CCB271FB1DB3E39F87E500866E81C11781A8BB5A03A4285BE1CAF276B4`;
- E1b paper-inspired CBAM supplement:
  `outputs/mobilenet_variant_ablation/cbam_supplement/best.pt`,
  SHA-256 `F6966DABE0801F221CC6E67B9EE117AF1B06C93A7E34C96D25771572616DDBE3`.

Their component flags, parameter counts, training time, and selected internal
development threshold are embedded in each checkpoint and its adjacent
`training_summary.json`.

## Important differences from the papers

### Improved MobileNetV3 [2]

The paper uses LeakyReLU6 in the shallow network, CBAM instead of SE, and
blueprint separable convolutions instead of depth-wise separable convolutions.
E1a intentionally uses the unmodified torchvision MobileNetV3-Small backbone
with ImageNet transfer learning. E1b implements verifiable component
ablations, but differs from the paper in three important ways: pretrained SE
is retained and supplemented rather than removed, LeakyReLU6 uses the local
explicit negative slope 0.1, and BSConv is not implemented. The E1b label is
therefore always "paper-inspired adaptation"; paper result tables cannot be
transferred to it.

### Detector evidence and E3b

The quantity historically named `P_det` is `detection confidence x slot
coverage`. Neither YOLO-World nor this geometric product makes it a native
occupancy probability. E3b first maps each branch through a fitted monotonic
calibrator, then uses only non-negative branch coefficients. This design and
its optimizer are local project contributions, not algorithms claimed by the
MobileNetV3 or YOLO-World papers.

### YOLO-World [6]

YOLO-World returns object detections for text categories. In this project, a
vacant slot is inferred only when classifier/detector occupancy evidence is
below the selected threshold. "Vacant" is not a YOLO-World detection class.

### TrackTrack [7]

The paper reports object-tracking metrics on standard MOT datasets. It does
not report slot occupancy. This project will report IDF1/HOTA only if
persistent human identity ground truth is acquired.

## References

[1] R. Grbic and B. Koch, "Automatic Vision-Based Parking Slot Detection and
Occupancy Classification," *Expert Systems with Applications*, vol. 225,
120147, 2023. DOI: 10.1016/j.eswa.2023.120147.

[2] Y. Yuldashev et al., "Parking Lot Occupancy Detection with Improved
MobileNetV3," *Sensors*, vol. 23, no. 17, 7642, 2023.
https://doi.org/10.3390/s23177642

[6] T. Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object
Detection," *CVPR*, 2024.
https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

[7] K. Shim et al., "Focusing on Tracks for Online Multi-Object Tracking,"
*CVPR*, 2025.
https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html
