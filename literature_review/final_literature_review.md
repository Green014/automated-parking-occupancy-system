# Literature Review: Automated Parking Lot Occupancy and Tracking System

## Abstract

Automated parking lot occupancy analysis aims to estimate whether each parking space is available or occupied from surveillance video, usually captured by fixed overhead or high-angle cameras. The task is related to object detection and multi-object tracking, but it also requires space-level reasoning, temporal stability, and robustness to illumination, weather, occlusion, and camera viewpoint changes. This review surveys recent top-tier computer vision papers published mainly in CVPR 2024 and CVPR 2025, with one ICCV 2025 auxiliary reference. Since few recent top-tier papers directly target parking occupancy detection, this review focuses on the technical components that would form a practical system: real-time and open-vocabulary detection, robust multi-object tracking, low-light and adverse-weather perception, and deployment-oriented robustness. Eight papers are selected as core references, and eight additional papers are retained as auxiliary candidates.

## 1. Introduction

Parking occupancy detection is often treated as a simple binary classification problem at the level of each parking slot. In real deployments, however, the problem is more complex. A surveillance-based system must detect vehicles from a fixed camera, associate detections across frames, map detections to predefined parking-slot regions, and decide when a slot changes from free to occupied or from occupied to free. It must also remain reliable under changing sunlight, shadows, rain, fog, wet ground reflections, nighttime lighting, partial occlusion, and camera-specific distortions.

The project considered here is an Automated Parking Lot Occupancy and Tracking System. The goal is to analyze overhead camera footage in real time and estimate the availability of each parking space. This makes the project a natural combination of five research areas:

- object detection for vehicles and obstacles
- open-vocabulary or open-world recognition for flexible scene understanding
- multi-object tracking for temporal consistency
- adverse-condition perception for lighting and weather robustness
- deployment-aware inference for real-time operation

Recent top-tier computer vision conferences contain many strong papers on these subproblems, even though few focus directly on parking lots. Therefore, the review adopts a component-based selection strategy: papers are selected if they provide methods, datasets, or design principles that can improve one or more parts of the parking occupancy pipeline.

## 2. Paper Collection and Screening Criteria

The initial candidate pool contains 16 papers. All PDFs are saved locally in `literature_review/papers/`, and their metadata is stored in `literature_review/metadata/`.

The inclusion criteria were:

- Publication date: 2024-2025.
- Venue priority: CVPR, ICCV, ECCV, ICLR, NeurIPS, ICML, or SIGGRAPH. The final pool is dominated by CVPR because CVF Open Access provided the strongest set of recent detection and tracking papers.
- Relevance to the project: detection, tracking, open-world/open-vocabulary recognition, low-light robustness, adverse-weather robustness, multi-view or wide-angle tracking, or real-time deployment.
- Transferability: the paper should provide methods or evaluation ideas that can reasonably be adapted to overhead parking-lot surveillance.

Eight papers were selected as the core review set:

1. YOLO-World: Real-Time Open-Vocabulary Object Detection
2. Multi-Object Tracking in the Dark
3. DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction
4. Towards Generalizable Multi-Object Tracking
5. Focusing on Tracks for Online Multi-Object Tracking
6. Multiple Object Tracking as ID Prediction
7. Towards RAW Object Detection in Diverse Conditions
8. OW-OVD: Unified Open World and Open Vocabulary Object Detection

The remaining eight papers are useful auxiliary references, especially for dense perception, long-tail detection, wide-angle or multi-view tracking, event-camera sensing, weather-aware segmentation, multi-modal robustness, and edge-device latency.

## 3. Detection and Open-Vocabulary Perception

Reliable vehicle detection is the first major component of a parking occupancy system. In a simple pipeline, the detector outputs bounding boxes for cars, trucks, motorcycles, and other relevant objects; these detections are then matched to parking-space polygons. However, closed-set detectors can be brittle when real lots contain unusual vehicles, temporary objects, trailers, cones, construction materials, or ambiguous vehicle-like objects.

YOLO-World [1] addresses this limitation by combining a YOLO-style real-time detector with open-vocabulary recognition. Its central value for this project is not only speed, but category flexibility. A parking system could use prompts such as "car", "truck", "motorcycle", "van", "traffic cone", "barrier", and "construction equipment" to detect objects that affect parking-space availability. This is especially useful during early prototyping, when collecting a large labeled parking-specific dataset may not be practical.

The limitation is that open-vocabulary detection does not remove the need for domain adaptation. Overhead parking-lot cameras have unusual viewing angles, small vehicles, repeated slot patterns, and strong shadows. Therefore, YOLO-World should be viewed as a strong detection backbone or initialization method rather than a complete solution. Parking-specific fine-tuning, prompt calibration, and confidence thresholds would still be needed.

OW-OVD [8] is another relevant detection paper because it unifies open-world and open-vocabulary detection. In a long-term deployment, a parking system must handle both known classes and unknown objects. For example, an unknown object occupying a parking space should still be flagged even if it is not recognized as a standard vehicle class. OW-OVD is useful conceptually because it treats unknown-object discovery and vocabulary-driven recognition in one framework. For a parking lot, however, open-world detection must be carefully calibrated. A fixed surveillance scene contains many static objects, such as lamp posts, painted markings, signs, and background structures, which could otherwise generate unnecessary alerts.

Several auxiliary candidates extend this direction. DeCLIP explores open-vocabulary dense perception, which could help if the project later uses segmentation to identify parking lines, drivable areas, or slot boundaries. Search and Detect explores training-free long-tail object detection through web-image retrieval, which is relevant when the system needs to detect rare or user-specified object categories without collecting a new labeled dataset.

## 4. Multi-Object Tracking for Temporal Consistency

Frame-level detection alone is usually not stable enough for parking occupancy estimation. A detector may miss a vehicle for one or two frames because of glare, occlusion, rain streaks, headlights, or compression artifacts. If occupancy is updated from single-frame predictions, the system may flicker between free and occupied states. Multi-object tracking is therefore essential.

Multi-Object Tracking in the Dark [2] is highly relevant because parking lots frequently operate at night. The paper introduces a low-light multi-object tracking setting and proposes a tracker designed for degraded visibility. For this project, the main lesson is that night scenes should not be treated as a minor edge case. Low-light tracking should be evaluated separately because detection confidence, appearance features, and motion association can all degrade at night.

DiffMOT [3] focuses on nonlinear motion prediction using a diffusion-based tracker. This is relevant because vehicles in parking lots do not follow simple constant-velocity trajectories. They slow down, reverse, turn sharply, stop at intersections, and maneuver into spaces. Trackers based only on simple motion assumptions can produce identity switches during these maneuvers. DiffMOT suggests that stronger motion modeling can improve tracking in such cases. The tradeoff is computational cost: a parking occupancy system may run on an edge device or local server, so any advanced tracker must be tested under real-time constraints.

Towards Generalizable Multi-Object Tracking [4] studies how trackers generalize across different scenes. This is important because parking lots vary substantially in camera height, viewing angle, lighting, slot layout, traffic density, and background texture. A tracker tuned on one site may fail when deployed elsewhere. The paper supports a deployment strategy that emphasizes cross-scene validation and avoids overfitting the tracking module to a single parking lot.

TrackTrack [5] and MOTIP [6] are recent online tracking methods from CVPR 2025. TrackTrack improves online association by focusing on tracks, while MOTIP formulates tracking as ID prediction. Both papers are relevant because a parking occupancy system needs online updates rather than offline batch processing. TrackTrack is especially practical for continuous monitoring, while MOTIP is conceptually interesting because it reduces reliance on hand-designed association rules. For parking lots, learned association could be useful when vehicles are visually similar, parked close together, or partially occluded.

The key limitation of general MOT papers is that they track objects, not parking spaces. A parking system needs an additional layer that maps vehicle tracks to fixed parking-slot polygons and applies temporal rules to update occupancy. Thus, MOT should be considered a stabilizing component rather than the final output layer.

## 5. Robustness to Lighting, Weather, and Sensor Conditions

Lighting and weather changes are central challenges for parking surveillance. A system that works only in clear daytime conditions is not sufficient for real-world deployment.

AODRaw [7] directly addresses object detection in diverse conditions, including daylight, low light, rain, and fog. The paper introduces a RAW-image detection benchmark and investigates RAW-domain pretraining with distillation. Although many parking cameras provide compressed RGB video rather than RAW data, the paper is still highly relevant because it emphasizes evaluation across environmental conditions. For this project, the practical takeaway is to build a validation split that explicitly separates clear daytime, night, rain, fog, glare, shadows, and wet pavement. This makes robustness measurable instead of anecdotal.

Multi-Object Tracking in the Dark [2] complements AODRaw from the tracking side. Together, they show that robustness should be evaluated at both detection and tracking levels. A detector may remain acceptable in low light while the tracker fails because appearance embeddings become unreliable. Conversely, tracking may hide short detection failures but fail during long occlusions or severe noise.

Two auxiliary papers provide additional design ideas. Weather-aware adaptation for semantic segmentation [10] is useful if the project incorporates segmentation for parking-space boundaries, road surfaces, or lane-like markings. Multi-modal expert fusion under adverse sensor failures [9] is less directly applicable to an RGB-only parking system, but it offers a useful principle: robust perception should avoid depending on a single fragile signal. In a camera-only parking system, this principle can be implemented by combining detection boxes, slot polygons, temporal smoothing, background priors, confidence calibration, and scene-specific masks.

## 6. Real-Time and Deployment Considerations

A parking lot occupancy system is usually expected to run continuously and provide near real-time status updates. Therefore, model accuracy alone is not enough. Latency, memory use, hardware variability, and robustness under load are also important.

YOLO-World [1] is attractive because it combines open-vocabulary detection with real-time inference. TrackTrack [5] and DiffMOT [3] are also relevant because they are designed for online tracking. However, the full system must consider cumulative latency: frame decoding, object detection, tracking, slot assignment, temporal smoothing, visualization, logging, and alert generation.

The auxiliary paper Can't Slow Me Down studies robust and hardware-adaptive object detectors against latency attacks for edge devices. While its threat model is not exactly the same as parking surveillance, its deployment focus is relevant. Parking systems may run on edge hardware with limited compute, and performance may vary with video resolution, frame rate, number of cameras, or simultaneous workloads. A practical system should therefore include benchmark targets such as frames per second, end-to-end latency, maximum camera count, and performance under night/rain scenes.

Event-camera object detection is also an auxiliary direction. Event cameras can perform well in high dynamic range and low-light motion settings, which are relevant to parking lots with headlights and nighttime scenes. However, most parking lots already have RGB surveillance cameras, so event cameras are better treated as a future extension rather than a default design requirement.

## 7. Comparison Table

| Paper | Venue / Year | Main Task | Method | Dataset / Benchmark | Strengths | Limitations | Relevance to Parking Occupancy |
|---|---|---|---|---|---|---|---|
| YOLO-World | CVPR 2024 | Real-time open-vocabulary object detection | YOLO detector with vision-language pretraining and region-text learning | LVIS and downstream detection tasks | Fast; flexible categories; useful before collecting large parking-specific labels | Still needs prompt calibration or fine-tuning for overhead camera views | Strong candidate detector for vehicles, obstacles, cones, barriers, and unusual objects |
| Multi-Object Tracking in the Dark | CVPR 2024 | Low-light MOT | Low-light tracking dataset plus LTrack | LMOT and low-light tracking benchmarks | Directly addresses night surveillance degradation | Does not solve parking-slot state estimation | Important for night occupancy stability |
| DiffMOT | CVPR 2024 | Real-time MOT with nonlinear motion | Diffusion-based motion prediction | DanceTrack, SportsMOT | Better suited to irregular motion than simple linear models | Diffusion components may increase compute cost | Useful for vehicles stopping, reversing, and turning into spaces |
| Towards Generalizable MOT | CVPR 2024 | Cross-scene MOT generalization | Scenario-aware relation modeling with GeneralTrack | Multiple MOT benchmarks | Emphasizes transfer across scenarios | Needs validation on parking-specific cameras | Helps deployment across different lots and camera heights |
| TrackTrack | CVPR 2025 | Online MOT | Track-perspective association and track-aware initialization | MOT17, MOT20, DanceTrack | Online and association-focused; reduces unstable tracks | Still requires slot-level mapping | Good fit for continuous real-time occupancy updates |
| MOTIP | CVPR 2025 | MOT as ID prediction | In-context ID prediction for association | Multiple MOT benchmarks | Reduces handcrafted matching rules | Needs strong features and parking-domain data for best use | Promising for learning identity association in crowded lots |
| AODRaw | CVPR 2025 | Detection under diverse conditions | RAW detection benchmark and RAW-domain pretraining/distillation | AODRaw | Covers low light, rain, fog, and diverse scenes | RAW video may not be available in deployed surveillance systems | Strong reference for robustness evaluation and data design |
| OW-OVD | CVPR 2025 | Open-world and open-vocabulary detection | Unified known/unknown object detection with attribute-based modeling | M-OWODB, S-OWODB | Can detect known classes and flag unknown objects | Unknown-object scores may create false positives in fixed scenes | Useful for unusual vehicles, obstacles, or temporary objects in spaces |

## 8. Proposed System Implications

Based on the reviewed literature, a practical parking occupancy system should be designed as a modular video analytics pipeline:

1. Camera calibration and slot definition: define parking-space polygons either manually or through a segmentation/line-detection module.
2. Vehicle and obstacle detection: use a real-time detector, with open-vocabulary detection as a flexible starting point.
3. Multi-object tracking: track detected vehicles across frames to reduce flicker and preserve identity.
4. Slot assignment: map detections or tracks to parking spaces using intersection-over-area, vehicle center location, or learned spatial assignment.
5. Temporal smoothing: update occupancy only after consistent evidence across several frames or seconds.
6. Condition-aware evaluation: report performance separately for clear daytime, nighttime, rain, fog, glare, shadow, and wet pavement.
7. Deployment profiling: measure end-to-end latency, FPS, memory use, and performance under multi-camera operation.

The reviewed papers suggest that the best initial technical stack would combine a YOLO-style detector, an online MOT method, and parking-slot geometry. YOLO-World is a strong candidate for detection, while TrackTrack, MOTIP, DiffMOT, and low-light tracking methods provide different tracking options. AODRaw and Multi-Object Tracking in the Dark should guide the robustness evaluation protocol.

## 9. Research Gap

The main research gap is that recent top-tier literature does not yet provide a unified solution for parking occupancy detection that jointly handles:

- overhead parking-space geometry
- vehicle and obstacle detection
- online multi-object tracking
- low-light and adverse-weather robustness
- real-time deployment constraints
- per-slot occupancy metrics

Most recent papers solve one component well, but parking occupancy requires these components to interact. This creates a clear opportunity for the project: build an integrated pipeline and evaluate it with parking-specific metrics. Suitable metrics include per-space occupancy accuracy, false occupied rate, false free rate, occupancy transition latency, ID switch rate near parking spaces, nighttime performance, adverse-weather performance, and system FPS.

## 10. Conclusion

The literature indicates that modern parking occupancy systems should move beyond single-frame vehicle detection. Recent open-vocabulary detectors make the system more flexible, recent MOT methods improve temporal stability, and adverse-condition detection/tracking papers provide guidance for robust evaluation. The most relevant core papers are YOLO-World, Multi-Object Tracking in the Dark, AODRaw, TrackTrack, MOTIP, DiffMOT, Generalizable MOT, and OW-OVD. Together, they support a modular architecture that combines real-time detection, online tracking, parking-slot geometry, and condition-aware evaluation.

For this project, the most defensible direction is to implement a detector-tracker-slot assignment pipeline, then evaluate it under multiple environmental conditions. The contribution can come not from inventing an entirely new detector, but from integrating recent perception methods into a parking-specific real-time system and measuring performance at the level that matters most: individual parking-space availability.

## Appendix A. Full Candidate Pool

| ID | Paper | Venue | Year | Role |
|---|---|---:|---:|---|
| P1 | YOLO-World: Real-Time Open-Vocabulary Object Detection | CVPR | 2024 | Core |
| P2 | Multi-Object Tracking in the Dark | CVPR | 2024 | Core |
| P3 | DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction | CVPR | 2024 | Core |
| P4 | Towards Generalizable Multi-Object Tracking | CVPR | 2024 | Core |
| P5 | Focusing on Tracks for Online Multi-Object Tracking | CVPR | 2025 | Core |
| P6 | Multiple Object Tracking as ID Prediction | CVPR | 2025 | Core |
| P7 | Towards RAW Object Detection in Diverse Conditions | CVPR | 2025 | Core |
| P8 | Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion | CVPR | 2025 | Auxiliary |
| P9 | OW-OVD: Unified Open World and Open Vocabulary Object Detection | CVPR | 2025 | Core |
| P10 | Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions | ICCV | 2025 | Auxiliary |
| P11 | DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception | CVPR | 2025 | Auxiliary |
| P12 | Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval | CVPR | 2025 | Auxiliary |
| P13 | Omnidirectional Multi-Object Tracking | CVPR | 2025 | Auxiliary |
| P14 | MITracker: Multi-View Integration for Visual Object Tracking | CVPR | 2025 | Auxiliary |
| P15 | Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset | CVPR | 2025 | Auxiliary |
| P16 | Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices | CVPR | 2025 | Auxiliary |

## References

[1] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html

[2] X. Wang, K. Ma, Q. Liu, Y. Zou, and Y. Fu, "Multi-Object Tracking in the Dark," CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Multi-Object_Tracking_in_the_Dark_CVPR_2024_paper.html

[3] W. Lv, Y. Huang, N. Zhang, R.-S. Lin, M. Han, and D. Zeng, "DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction," CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html

[4] Z. Qin, L. Wang, S. Zhou, P. Fu, G. Hua, and W. Tang, "Towards Generalizable Multi-Object Tracking," CVPR, 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html

[5] K. Shim, K. Ko, Y. Yang, and C. Kim, "Focusing on Tracks for Online Multi-Object Tracking," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html

[6] R. Gao, J. Qi, and L. Wang, "Multiple Object Tracking as ID Prediction," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html

[7] Z.-Y. Li, X. Jin, B.-Y. Sun, C.-L. Guo, and M.-M. Cheng, "Towards RAW Object Detection in Diverse Conditions," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_RAW_Object_Detection_in_Diverse_Conditions_CVPR_2025_paper.html

[8] X. Xi, Y. Huang, R. Luo, and Y. Qiu, "OW-OVD: Unified Open World and Open Vocabulary Object Detection," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Xi_OW-OVD_Unified_Open_World_and_Open_Vocabulary_Object_Detection_CVPR_2025_paper.html

[9] K. Park, Y. Kim, D. Kim, and J. W. Choi, "Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html

[10] Y. Pan, R. Sun, W. Li, and T. Zhang, "Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions," ICCV, 2025. https://openaccess.thecvf.com/content/ICCV2025/html/Pan_Exploring_Weather-aware_Aggregation_and_Adaptation_for_Semantic_Segmentation_under_Adverse_ICCV_2025_paper.html

[11] J. Wang, B. Chen, Y. Li, B. Kang, Y. Chen, and Z. Tian, "DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Wang_DeCLIP_Decoupled_Learning_for_Open-Vocabulary_Dense_Perception_CVPR_2025_paper.html

[12] M. Sidhu, H. Chopra, A. Blume, J. Kim, R. G. Reddy, and H. Ji, "Search and Detect: Training-Free Long Tail Object Detection via Web-Image Retrieval," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Sidhu_Search_and_Detect_Training-Free_Long_Tail_Object_Detection_via_Web-Image_CVPR_2025_paper.html

[13] K. Luo, H. Shi, S. Wu, F. Teng, M. Duan, C. Huang, Y. Wang, K. Wang, and K. Yang, "Omnidirectional Multi-Object Tracking," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Omnidirectional_Multi-Object_Tracking_CVPR_2025_paper.html

[14] M. Xu, Y. Zhu, H. Jiang, J. Li, Z. Shen, S. Wang, H. Huang, X. Wang, H. Zhang, Q. Yang, and Q. Wang, "MITracker: Multi-View Integration for Visual Object Tracking," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Xu_MITracker_Multi-View_Integration_for_Visual_Object_Tracking_CVPR_2025_paper.html

[15] X. Wang, Y. Jin, W. Wu, W. Zhang, L. Zhu, B. Jiang, and Y. Tian, "Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Object_Detection_using_Event_Camera_A_MoE_Heat_Conduction_based_CVPR_2025_paper.html

[16] T. Wang, Z. Wang, C. Wang, Y. Shu, R. Deng, P. Cheng, and J. Chen, "Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices," CVPR, 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Cant_Slow_Me_Down_Learning_Robust_and_Hardware-Adaptive_Object_Detectors_CVPR_2025_paper.html

