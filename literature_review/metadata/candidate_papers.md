# Candidate Paper Pool

The following papers were collected for the literature review. All PDFs are saved in `literature_review/papers/`.

| ID | Paper | Venue | Year | Local PDF | Role in Review |
|---|---|---:|---:|---|---|
| P1 | YOLO-World: Real-Time Open-Vocabulary Object Detection | CVPR | 2024 | `2024_CVPR_YOLO-World.pdf` | Core |
| P2 | Multi-Object Tracking in the Dark | CVPR | 2024 | `2024_CVPR_Multi-Object_Tracking_in_the_Dark.pdf` | Core |
| P3 | DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction | CVPR | 2024 | `2024_CVPR_DiffMOT.pdf` | Core |
| P4 | Towards Generalizable Multi-Object Tracking | CVPR | 2024 | `2024_CVPR_Generalizable_MOT.pdf` | Core |
| P5 | Focusing on Tracks for Online Multi-Object Tracking | CVPR | 2025 | `2025_CVPR_TrackTrack.pdf` | Core |
| P6 | Multiple Object Tracking as ID Prediction | CVPR | 2025 | `2025_CVPR_MOTIP.pdf` | Core |
| P7 | Towards RAW Object Detection in Diverse Conditions | CVPR | 2025 | `2025_CVPR_AODRaw.pdf` | Core |
| P8 | Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion | CVPR | 2025 | `2025_CVPR_MoME.pdf` | Auxiliary |
| P9 | OW-OVD: Unified Open World and Open Vocabulary Object Detection | CVPR | 2025 | `2025_CVPR_OW-OVD.pdf` | Core |
| P10 | Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions | ICCV | 2025 | `2025_ICCV_WeatherAware_Segmentation.pdf` | Auxiliary |
| P11 | DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception | CVPR | 2025 | `2025_CVPR_DeCLIP.pdf` | Candidate |
| P12 | Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval | CVPR | 2025 | `2025_CVPR_SearchDet.pdf` | Candidate |
| P13 | Omnidirectional Multi-Object Tracking | CVPR | 2025 | `2025_CVPR_OmniTrack.pdf` | Candidate |
| P14 | MITracker: Multi-View Integration for Visual Object Tracking | CVPR | 2025 | `2025_CVPR_MITracker.pdf` | Candidate |
| P15 | Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset | CVPR | 2025 | `2025_CVPR_EventCamera_Detection.pdf` | Candidate |
| P16 | Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices | CVPR | 2025 | `2025_CVPR_Cant_Slow_Me_Down.pdf` | Candidate |
| P17 | Automatic Vision-Based Parking Slot Detection and Occupancy Classification | Expert Systems with Applications | 2023 | `2023_ESWA_APSD-OC.pdf` | Parking-specific core |
| P18 | Parking Lot Occupancy Detection with Improved MobileNetV3 | Sensors | 2023 | MDPI PDF download blocked; page saved as source link | Parking-specific core |
| P19 | CMCA-YOLO: A Study on a Real-Time Object Detection Model for Parking Lot Surveillance Imagery | Electronics | 2024 | MDPI PDF download blocked; page saved as source link | Parking-specific core |
| P20 | Car Parking Space Detection Using YOLOv8 | ISCP / SCITEPRESS | 2024 | `2024_ISCP_Car_Parking_Space_Detection_YOLOv8.pdf` | Parking-specific candidate |
| P21 | Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11 | arXiv | 2024 | `2024_arXiv_Smart_Parking_Pixel-Wise_ROI_YOLO.pdf` | Parking-specific candidate |
| P22 | Optimizing YOLOv8 for Parking Space Detection: Comparative Analysis of Custom YOLOv8 Architecture | arXiv | 2025 | `2025_arXiv_Optimizing_YOLOv8_Parking_Space_Detection.pdf` | Parking-specific candidate |
| P23 | PKLot: A Robust Dataset for Parking Lot Classification | Expert Systems with Applications | 2015 | No local PDF; source link recorded | Foundational dataset reference |
| P24 | Real-Time Parking Space Detection Based on Deep Learning and Panoramic Images | Sensors | 2025 | No local PDF; PMC/MDPI source link recorded | Parking-specific core candidate |

## Selection rationale

No recent top-tier paper was found that directly solves parking lot occupancy from overhead surveillance footage. Therefore, the selected core papers cover the modules needed for such a system:

- open-vocabulary or open-world detection for flexible vehicle/space/object categories
- real-time detection for live surveillance
- tracking under occlusion, nonlinear motion, and changing views
- low-light and adverse-weather robustness
- domain generalization for new lots and camera positions

## Additional candidate rationale

The six added candidates broaden the pool in directions that may become useful during implementation:

- DeCLIP adds open-vocabulary dense perception, relevant if parking-space segmentation or surface/line parsing is needed.
- SearchDet adds training-free long-tail detection, relevant when the system must recognize rare or user-defined object types.
- OmniTrack addresses panoramic and distorted views, relevant to wide-angle parking-lot cameras.
- MITracker addresses multi-view tracking, relevant if one lot uses multiple overlapping cameras.
- Event-camera detection is relevant to low-light, motion blur, and high dynamic range scenarios if alternative sensors are considered.
- Can't Slow Me Down addresses edge-device detector robustness and latency, relevant to real-time deployment.

## Coursework alignment update

P17-P22 were added after reviewing the coursework brief. They are less "top conference" oriented than P1-P16, but they are directly relevant to the required literature review criteria:

- comparison of existing parking occupancy systems and algorithms
- commonly used datasets such as PKLot and CNRPark-EXT
- evaluation metrics such as accuracy, precision, recall, F1, AUC, mAP, inference time, and balanced accuracy
- design choices such as slot-patch classification, automatic slot detection, YOLO-based vehicle/space detection, and ROI-based post-processing

P23 was added as a foundational dataset reference. It is older than the coursework's preferred recent window, but it is peer-reviewed and widely used for parking occupancy benchmarking, so it is appropriate as a dataset/methodology citation rather than a recent algorithm citation.

P24 was added as the preferred replacement for the condensed 10-reference version because it is recent, peer-reviewed, parking-specific, and reports both detection accuracy and real-time performance.
