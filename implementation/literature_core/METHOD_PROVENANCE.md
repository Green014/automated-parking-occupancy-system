# Method Provenance

All provenance labels describe this repository's implementation, not merely
the paper category.

| Module | Part I source | Code/weights source | License | Project change | Status |
|---|---|---|---|---|---|
| Slot patch classification | Yuldashev et al. [2] | torchvision `mobilenet_v3_small`; ImageNet weights from PyTorch | BSD-style PyTorch/torchvision terms; ImageNet weights subject to their dataset terms | OpenCV perspective-normalized 224x224 slot patch, standard MobileNetV3-Small, two-class head, ImageNet transfer learning | **Adaptation**, not exact reproduction |
| Open-vocabulary detection | Cheng et al., YOLO-World [6] | Original: `AILab-CVC/YOLO-World`; runtime adapter: Ultralytics `YOLOWorld`; `yolov8s-worldv2.pt` | Original repository GPL-3.0; Ultralytics package/weights AGPL-3.0 for this academic workflow | Prompts restricted to `car`, `truck`, `bus`, `motorcycle`; raw box and confidence retained | **Adaptation** |
| Detection-to-slot mapping | APSD-OC [1] supports linking detections and slot geometry; baseline provides an existing OpenCV implementation | Implemented locally with `cv2.intersectConvexConvex` | Project code | Confidence-weighted fraction of each slot polygon covered by a prompted object box, with one-to-one greedy assignment | **Inspired/local integration** |
| Weighted evidence fusion | Combination motivated by [2] and [6] | Implemented locally | Project code | Development-selected normalized weighted sum of `P_cls`, `P_det`, and optional `P_track`; every component is logged | **Local system integration** |
| Temporal hysteresis | Existing baseline; tracking motivation from TrackTrack [7] | Reimplemented in the independent package | Project code | Asymmetric EMA with separate occupied/vacant thresholds; delayed true transitions are separated from ordinary flicker | **Adaptation/reuse** |
| Track-aware evidence | TrackTrack [7] | No dependable implementation frozen | Not applicable | Interface reserved, but no E5 result is claimed | **Not implemented / optional** |
| Automatic slot discovery | APSD-OC [1] | No official implementation confirmed | Not applicable | No homography/DBSCAN reproduction is claimed | **Not implemented / future work** |

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
- TrackTrack: no implementation was frozen into this project; the paper is
  used only to scope the optional tracking extension.

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

## Important differences from the papers

### Improved MobileNetV3 [2]

The paper uses LeakyReLU6 in the shallow network, CBAM instead of SE, and
blueprint separable convolutions instead of depth-wise separable convolutions.
This project intentionally uses the unmodified torchvision
MobileNetV3-Small backbone with ImageNet transfer learning. It is lighter to
audit and feasible on the available GPU, but paper result tables cannot be
transferred to this implementation.

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
