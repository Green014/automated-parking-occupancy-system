# Dataset Availability for Candidate Papers

Checked on 2026-06-07.

This file distinguishes between:

- **New dataset introduced by the paper**: the paper creates or releases a new dataset/benchmark.
- **Uses existing datasets only**: the paper evaluates on public benchmarks but does not contribute a new dataset.
- **Derived benchmark/split**: the paper defines a benchmark protocol or task split based on existing datasets.

## Core Papers

| ID | Paper | Dataset status | Dataset / benchmark names | Availability / link | Notes for parking project |
|---|---|---|---|---|---|
| P1 | YOLO-World | Uses existing large-scale datasets only | Objects365, GoldG, CC-LiteV2; evaluated on LVIS, LVIS-mini, COCO | https://github.com/AILab-CVC/YOLO-World | Useful for pretrained detection, but no parking-specific dataset. |
| P2 | Multi-Object Tracking in the Dark | Introduces new dataset | LMOT: LMOT-dual and LMOT-real | https://github.com/xinzwang/LMOT | Very useful for night/low-light tracking experiments; includes car, person, bicycle, motorcycle, bus, truck. Non-commercial research license. |
| P3 | DiffMOT | Uses existing datasets only | DanceTrack, SportsMOT, MOT17, MOT20 | https://github.com/Kroery/DiffMOT | Useful tracking benchmarks, but no new parking dataset. |
| P4 | Towards Generalizable Multi-Object Tracking | Uses existing datasets only | BDD100K, SportsMOT, DanceTrack, MOT17, MOT20 | https://github.com/qinzheng2000/GeneralTrack | Useful for cross-scene generalization thinking; no new dataset. |
| P5 | Focusing on Tracks for Online Multi-Object Tracking | Uses existing datasets only | MOT17, MOT20, DanceTrack | https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html | Good online tracking benchmark coverage; no new dataset found. |
| P6 | Multiple Object Tracking as ID Prediction | Uses existing datasets only | DanceTrack, SportsMOT, BFT, CrowdHuman | https://github.com/MCG-NJU/MOTIP/blob/main/docs/DATASET.md | Dataset preparation is documented; no new dataset. |
| P7 | Towards RAW Object Detection in Diverse Conditions | Introduces new dataset | AODRaw | https://github.com/lzyhha/AODRaw | Highly useful for robustness design: RAW and sRGB images, diverse lighting/weather tags, COCO-format annotations. Large storage requirement. |
| P9 | OW-OVD | Uses/defines OWOD benchmarks; no separate dataset release found | M-OWODB, S-OWODB | https://github.com/xxyzll/OW_OVD | Useful for open-world/unknown-object evaluation; not parking-specific and does not appear to release a new raw dataset. |

## Auxiliary Candidates

| ID | Paper | Dataset status | Dataset / benchmark names | Availability / link | Notes |
|---|---|---|---|---|---|
| P8 | Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion | Uses existing datasets/variants | nuScenes, nuScenes-C, nuScenes-R | https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html | Useful if project expands to multi-modal sensors. |
| P10 | Weather-aware Aggregation and Adaptation for Semantic Segmentation | Uses existing/synthetic adverse-condition segmentation data; no new released dataset found in quick check | Adverse-condition segmentation benchmarks | https://openaccess.thecvf.com/content/ICCV2025/html/Pan_Exploring_Weather-aware_Aggregation_and_Adaptation_for_Semantic_Segmentation_under_Adverse_ICCV_2025_paper.html | Useful for weather-aware segmentation ideas. |
| P11 | DeCLIP | Uses existing dense perception datasets | Dense perception/open-vocabulary benchmarks | https://openaccess.thecvf.com/content/CVPR2025/html/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.html | Method paper; no new dataset identified. |
| P12 | Search and Detect | Uses existing long-tail/open-vocabulary detection benchmarks plus web-image retrieval | Long-tail detection benchmarks; retrieved web images | https://openaccess.thecvf.com/content/CVPR2025/html/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.html | No new curated dataset found; relevant for rare-object detection. |
| P13 | Omnidirectional Multi-Object Tracking | Introduces/establishes dataset | OmniTrack dataset / panoramic MOT dataset | https://github.com/xifen523/OmniTrack | Useful only if using panoramic/fisheye/wide-FOV cameras. |
| P14 | MITracker | Introduces new dataset | MVTrack | https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_MITracker_Multi-View_Integration_for_Visual_Object_Tracking_CVPR_2025_paper.pdf | Useful if project expands to multiple overlapping cameras. |
| P15 | Object Detection using Event Camera | Introduces benchmark dataset | OpenEvDET | https://github.com/Event-AHU/OpenEvDET | Useful only if event-camera hardware is considered. |
| P16 | Can't Slow Me Down | Uses existing object detection datasets; no new dataset found | Object detection benchmarks | https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Cant_Slow_Me_Down_Learning_Robust_and_Hardware-Adaptive_Object_Detectors_CVPR_2025_paper.pdf | Deployment/latency paper; not a dataset source. |

## Parking-Specific Added Papers

| ID | Paper | Dataset status | Dataset / benchmark names | Availability / link | Notes |
|---|---|---|---|---|---|
| P17 | APSD-OC | Uses existing parking occupancy datasets | PKLot, CNRPark-EXT | https://arxiv.org/abs/2308.08192 | Very relevant because it evaluates automatic slot detection and occupancy classification. |
| P18 | Improved MobileNetV3 Parking Occupancy | Uses existing parking occupancy datasets | PKLot, CNRPark-EXT | https://www.mdpi.com/1424-8220/23/17/7642 | Good reference for slot-patch classification and AUC/accuracy reporting. |
| P19 | CMCA-YOLO Parking Surveillance | Introduces custom dataset | Parking-lot scene dataset with 4502 images | https://www.mdpi.com/2079-9292/13/8/1557 | Relevant for parking-lot surveillance object detection, small/overlapping targets, and custom data design. |
| P20 | Car Parking Space Detection Using YOLOv8 | Introduces custom dataset | UTA'45 Jakarta parking dataset, about 5000 images | https://www.scitepress.org/Papers/2023/125826/125826.pdf | Direct YOLOv8 parking detection example. |
| P21 | Smart Parking with Pixel-Wise ROI + YOLO | Introduces custom dataset | Custom dataset with 3484 images | https://arxiv.org/abs/2412.01983 | Useful for ROI-based occupancy assignment and runtime comparison. |
| P22 | Optimizing YOLOv8 for Parking Space Detection | Uses existing parking dataset | PKLot | https://arxiv.org/abs/2505.17364 | Useful for model/backbone comparison on parking occupancy. |

## Most Relevant Datasets for This Parking Project

1. **PKLot and CNRPark-EXT**: best direct public datasets for slot-level parking occupancy classification.
2. **Custom parking video/image dataset**: needed for this specific system if tracking, ROI mapping, or local camera geometry must be evaluated.
3. **LMOT**: useful for night/low-light vehicle tracking.
4. **AODRaw**: useful as a reference for diverse lighting and weather object detection.
5. **BDD100K / MOT17 / MOT20 / DanceTrack / SportsMOT**: useful public tracking/detection benchmarks, but not parking-specific.
6. **MVTrack / OmniTrack / OpenEvDET**: useful only for expanded hardware/camera setups.

## Practical Recommendation

For a parking occupancy project, none of these datasets directly provides overhead parking-space occupancy labels. Therefore, the project will likely need a small custom parking-lot dataset with:

- fixed parking-space polygons
- frame-level or time-interval occupancy labels
- vehicle bounding boxes for a subset of frames
- night/rain/fog/shadow condition tags
- optional track IDs for vehicles entering or leaving spaces
