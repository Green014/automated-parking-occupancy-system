# Literature Review: Automated Parking Lot Occupancy and Tracking System

## 1. Project Context

An automated parking lot occupancy and tracking system must infer whether each parking space is free or occupied from overhead or surveillance video. Compared with generic object detection, this task has several practical complications: vehicles can be partially occluded, parking spaces are fixed spatial regions, cars may stop or reverse slowly, lighting changes over the day, and weather can degrade image quality. A reliable system therefore needs more than a detector. It needs a pipeline that combines object detection, multi-object tracking, slot-level spatial reasoning, and robustness to changing visual conditions.

Because recent top-tier papers directly focused on parking occupancy are limited, this review selects recent papers from CVPR 2024, CVPR 2025, and ICCV 2025 that cover the core technical building blocks needed for the project. The final core set contains eight papers, exceeding the required six-paper minimum.

## 2. Screening Criteria

The papers were selected using the following criteria:

- Publication year: primarily 2024-2025.
- Venue: CVPR or ICCV, with preference for main-conference papers.
- Technical relevance to at least one project module: detection, tracking, adverse lighting/weather robustness, real-time inference, or generalization.
- Practical transferability to surveillance-based parking-lot video.

The selected papers do not all solve parking occupancy directly. Instead, they provide recent methods and design principles that can be combined into a parking occupancy system.

## 3. Detection for Parking-Lot Perception

Vehicle detection is the first stage of most parking occupancy systems. YOLO-World (Cheng et al., CVPR 2024) is especially relevant because it combines real-time detection with open-vocabulary recognition. The paper reports efficient zero-shot detection through a YOLO-style architecture enhanced by vision-language modeling. For parking lots, this is useful because a fixed detector trained only on "car" may miss vans, buses, motorcycles, cones, temporary barriers, or unusual objects that affect parking-space availability. Its limitation is that overhead surveillance views differ from common detection benchmarks, so fine-tuning or prompt engineering would likely still be required.

OW-OVD (Xi et al., CVPR 2025) extends this idea toward open-world and open-vocabulary detection. This matters for long-term deployments, where the system may encounter unknown object types or scene changes. A parking-lot system could use such a detector to flag abnormal space occupation, such as construction equipment or temporary signs. However, open-world methods can also increase false positives, so a production system should combine objectness scores with spatial rules tied to known parking-slot polygons.

AODRaw (Li et al., CVPR 2025) addresses detection under diverse lighting and weather conditions. The paper introduces AODRaw, which includes daylight, low-light, rain, and fog conditions, and studies RAW-domain pretraining with distillation. Although many surveillance systems use compressed video rather than RAW images, the paper is still important because it shows that detection robustness should be evaluated under multiple environmental conditions rather than only clear daytime imagery. For this project, the lesson is to build or augment a validation set that includes night, rain, fog, glare, shadows, and wet pavement.

## 4. Tracking for Occupancy Stability

Occupancy estimation should not simply classify each frame independently. Frame-level detection can flicker due to occlusion, shadows, or missed detections. Multi-object tracking helps maintain vehicle identity and smooth the occupancy state over time.

Multi-Object Tracking in the Dark (Wang et al., CVPR 2024) is directly useful for parking surveillance because parking lots often operate at night. The paper introduces a low-light MOT dataset and a tracker designed to suppress degradation from noise and poor illumination. This work supports the idea that nighttime performance should be treated as a first-class requirement rather than an afterthought.

DiffMOT (Lv et al., CVPR 2024) focuses on nonlinear motion prediction using a diffusion-based tracker. This is relevant because vehicles in parking lots do not move like pedestrians in a straight path. They stop, reverse, turn sharply, and move slowly around parked cars. A tracker that handles nonlinear motion may reduce ID switches when vehicles maneuver into or out of spaces. The tradeoff is that diffusion-based prediction should be benchmarked carefully if the system must run in real time on modest hardware.

Towards Generalizable Multi-Object Tracking (Qin et al., CVPR 2024) studies why trackers fail to generalize across scenarios and proposes GeneralTrack. This matters because a parking system may be installed in many different lots with different camera angles, layouts, traffic densities, and visual backgrounds. The paper supports choosing a tracker that is not overfitted to a single benchmark or camera geometry.

TrackTrack (Shim et al., CVPR 2025) improves online MOT through track-perspective association and track-aware initialization. For this project, online tracking is important because occupancy status should update continuously. Track-aware initialization is also useful for avoiding spurious vehicle tracks caused by reflections, shadows, or duplicated detections around parked cars.

MOTIP (Gao et al., CVPR 2025) reformulates tracking association as in-context ID prediction. This is conceptually useful because many parking-lot edge cases are hard to encode with manual association rules. A learned association approach could adapt better if trained or fine-tuned on parking-lot video. The likely challenge is data: domain-specific labeled trajectories would be needed to realize the full benefit.

## 5. Robustness to Weather, Lighting, and Deployment Conditions

Weather and illumination are central to this project. AODRaw provides the strongest direct reference for visual degradation in detection. Multi-Object Tracking in the Dark addresses low-light video tracking. As auxiliary support, Pan et al. (ICCV 2025) study weather-aware adaptation for semantic segmentation under adverse conditions. Although segmentation is not the same as parking occupancy, weather-aware feature adaptation could help if the system uses slot segmentation, drivable-area segmentation, or parking-line detection.

Park et al. (CVPR 2025) propose a robust multi-modal detector under sensor failures. This is less directly applicable to a single-camera parking system, but it gives a useful design principle: robust perception systems should avoid over-dependence on one fragile signal. In an RGB-only parking project, the analogous strategy is to combine multiple cues: vehicle boxes, slot polygons, temporal consistency, scene masks, and confidence smoothing.

## 6. Comparison Table

| Paper | Venue / Year | Main Task | Method | Dataset / Benchmark | Strengths | Limitations | Relevance to Parking Occupancy System |
|---|---|---|---|---|---|---|---|
| YOLO-World | CVPR 2024 | Real-time open-vocabulary object detection | YOLO detector plus vision-language pretraining and region-text contrastive learning | LVIS and downstream detection/segmentation tasks | Fast, flexible categories, suitable for deployment-oriented detection | May need prompt tuning and fine-tuning for overhead parking views | Detect vehicles and unexpected objects without a rigid fixed class set |
| Multi-Object Tracking in the Dark | CVPR 2024 | Low-light multi-object tracking | LMOT dataset plus LTrack with low-light robust feature learning | LMOT and night low-light scenes | Directly addresses night surveillance and sensor noise | Not designed for parking-slot occupancy labels | Important for nighttime parking-lot tracking |
| DiffMOT | CVPR 2024 | Real-time MOT with nonlinear motion prediction | Decoupled diffusion-based motion predictor | DanceTrack, SportsMOT | Handles nonlinear motion and remains real-time in reported benchmarks | Diffusion-based prediction may be heavier than simple Kalman-based trackers | Useful for vehicles turning, stopping, reversing, and moving irregularly inside lots |
| Towards Generalizable MOT | CVPR 2024 | Domain-generalized multi-object tracking | Scenario-attribute analysis and GeneralTrack relation modeling | Multiple MOT benchmarks | Emphasizes cross-scenario robustness | Still needs parking-lot validation | Helps transfer across different parking lots and camera placements |
| TrackTrack | CVPR 2025 | Online multi-object tracking | Track-perspective association and track-aware initialization | MOT17, MOT20, DanceTrack | Strong online association and fewer spurious tracks | Occupancy still needs slot-level reasoning | Good candidate for real-time occupancy update pipelines |
| MOTIP | CVPR 2025 | MOT as ID prediction | In-context ID prediction for object association | Multiple MOT benchmarks | Reduces handcrafted association rules | Needs reliable object features and likely domain adaptation | Promising for learning tracking behavior from parking videos |
| AODRaw | CVPR 2025 | Object detection in diverse lighting/weather | RAW detection dataset and RAW-domain pretraining with distillation | AODRaw | Covers low light, rain, fog, diverse scenes and categories | RAW data may not be available from common surveillance video | Strong guidance for weather/lighting robustness and evaluation design |
| OW-OVD | CVPR 2025 | Open-world and open-vocabulary detection | Attribute selection plus attribute-uncertainty fusion | M-OWODB, S-OWODB | Detects known and unknown objects with flexible vocabulary | Unknown-object scores may require calibration in fixed scenes | Useful for obstacles, unusual vehicles, and scene changes |

## 7. Implications for the Proposed System

The reviewed work suggests that a strong parking occupancy system should use a modular video pipeline:

1. Use a real-time detector such as a YOLO-style model, with open-vocabulary methods considered for flexible categories.
2. Calibrate parking-space polygons or use a segmentation/line-detection module to map detections to individual spaces.
3. Apply online multi-object tracking to smooth frame-level detections and maintain vehicle identities.
4. Add temporal occupancy logic, such as requiring consistent evidence across several frames before switching a space from free to occupied.
5. Evaluate separately on daytime, nighttime, rain, fog, shadows, glare, and partially occluded vehicles.
6. Use domain adaptation or fine-tuning when moving to a new parking lot.

The most immediately useful core references are YOLO-World for detection, Multi-Object Tracking in the Dark for night robustness, TrackTrack or MOTIP for online tracking, and AODRaw for weather/lighting evaluation. DiffMOT and GeneralTrack provide additional insight into nonlinear motion and cross-scene generalization.

## 8. Research Gap

The main gap is the absence of a recent top-tier method that directly combines overhead parking-space geometry, vehicle detection, temporal tracking, and adverse-weather robustness into one end-to-end parking occupancy framework. Most top-tier work solves one component well. Therefore, this project can contribute by integrating these components and evaluating them on parking-specific metrics such as per-space occupancy accuracy, occupancy transition latency, false occupied/free rate, and robustness across environmental conditions.

## References

- Cheng et al. YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- Wang et al. Multi-Object Tracking in the Dark. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Multi-Object_Tracking_in_the_Dark_CVPR_2024_paper.html
- Lv et al. DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html
- Qin et al. Towards Generalizable Multi-Object Tracking. CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html
- Shim et al. Focusing on Tracks for Online Multi-Object Tracking. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html
- Gao et al. Multiple Object Tracking as ID Prediction. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html
- Li et al. Towards RAW Object Detection in Diverse Conditions. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_RAW_Object_Detection_in_Diverse_Conditions_CVPR_2025_paper.html
- Xi et al. OW-OVD: Unified Open World and Open Vocabulary Object Detection. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Xi_OW-OVD_Unified_Open_World_and_Open_Vocabulary_Object_Detection_CVPR_2025_paper.html
- Park et al. Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html
- Pan et al. Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions. ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Pan_Exploring_Weather-aware_Aggregation_and_Adaptation_for_Semantic_Segmentation_under_Adverse_ICCV_2025_paper.html

