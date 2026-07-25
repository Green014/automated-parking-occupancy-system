# Code Availability for Candidate Papers

Checked on 2026-06-07.

## Core papers

| ID | Paper | Code status | Repository / project link | Notes |
|---|---|---|---|---|
| P1 | YOLO-World | Available | https://github.com/AILab-CVC/YOLO-World | Public GitHub repository; GPL-3.0 license. |
| P2 | Multi-Object Tracking in the Dark | Available | https://github.com/ying-fu/LMOT | Paper/arXiv points to dataset and code repository. |
| P3 | DiffMOT | Available | https://github.com/Kroery/DiffMOT | Public GitHub repository for the CVPR 2024 paper. |
| P4 | Towards Generalizable Multi-Object Tracking | Available | https://github.com/qinzheng2000/GeneralTrack | Public GitHub repository for GeneralTrack. |
| P5 | Focusing on Tracks for Online Multi-Object Tracking | Unclear / not confirmed as open source | https://github.com/kamkyu94/TrackTrack or https://github.com/kamkyu94/TrackTrackpytorch | Paper indexes list a GitHub link, but MOTChallenge marks the submission as "Open source: No"; verify before relying on it. |
| P6 | Multiple Object Tracking as ID Prediction | Available | https://github.com/MCG-NJU/MOTIP | Public GitHub repository; Apache-2.0 license. |
| P7 | Towards RAW Object Detection in Diverse Conditions | Available | https://github.com/lzyhha/AODRaw | Public GitHub repository; non-commercial CC BY-NC-SA style license noted by the repository. |
| P9 | OW-OVD | Available | https://github.com/xxyzll/OW_OVD | Public GitHub repository; code is based on YOLO-World. |

## Auxiliary candidates

| ID | Paper | Code status | Repository / project link | Notes |
|---|---|---|---|---|
| P8 | Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion | Available | https://github.com/konyul/MoME | Public GitHub repository; MIT license. |
| P10 | Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions | Not found during quick check | N/A | CVF page did not expose an obvious code link in the checked result. |
| P11 | DeCLIP | Available | https://github.com/xiaomoguhz/DeCLIP | CVF page states code is available. |
| P12 | Search and Detect | Not found during quick check | N/A | No official GitHub link found in the checked search results. |
| P13 | Omnidirectional Multi-Object Tracking | Available | https://github.com/xifen523/OmniTrack | Public official implementation. |
| P14 | MITracker | Available | https://github.com/XuM007/MITracker | Public official implementation. |
| P15 | Object Detection using Event Camera | Link announced / likely available | https://github.com/Event-AHU/OpenEvDET | Paper states source code will be released at this repository. |
| P16 | Can't Slow Me Down | Not found during quick check | N/A | No official GitHub link found in the checked search results. |

## Practical priority for this project

For implementation, the most useful open-source starting points are:

1. YOLO-World for open-vocabulary vehicle/object detection.
2. MOTIP, DiffMOT, GeneralTrack, or LMOT for tracking experiments.
3. AODRaw for robustness-oriented dataset/evaluation ideas.
4. OW-OVD if abnormal or unknown occupied-object detection becomes important.

