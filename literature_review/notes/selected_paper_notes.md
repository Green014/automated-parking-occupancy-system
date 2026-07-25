# Selected Paper Notes

## P1. YOLO-World: Real-Time Open-Vocabulary Object Detection

- Venue/year: CVPR 2024.
- Local PDF: `literature_review/papers/2024_CVPR_YOLO-World.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html
- Core idea: Combines YOLO-style efficient detection with vision-language pretraining, enabling open-vocabulary detection at real-time speed.
- Relevance: Useful for a parking system that may need to detect cars, trucks, motorcycles, empty spaces, barriers, cones, or abnormal objects without repeatedly retraining a closed-set detector.
- Limitation: Open-vocabulary detection still needs careful prompt design and may require fine-tuning for overhead parking-lot views.

## P2. Multi-Object Tracking in the Dark

- Venue/year: CVPR 2024.
- Local PDF: `literature_review/papers/2024_CVPR_Multi-Object_Tracking_in_the_Dark.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Multi-Object_Tracking_in_the_Dark_CVPR_2024_paper.html
- Core idea: Builds a low-light MOT dataset and proposes LTrack for tracking under night scenes and sensor noise.
- Relevance: Parking lots often operate at night. The paper directly addresses low-light surveillance-like video, making it highly relevant for nighttime occupancy tracking.
- Limitation: The paper focuses on low-light tracking rather than parking-space state estimation.

## P3. DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction

- Venue/year: CVPR 2024.
- Local PDF: `literature_review/papers/2024_CVPR_DiffMOT.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Lv_DiffMOT_A_Real-time_Diffusion-based_Multiple_Object_Tracker_with_Non-linear_Prediction_CVPR_2024_paper.html
- Core idea: Uses a diffusion-based motion predictor for nonlinear motion while maintaining real-time tracking.
- Relevance: Parking-lot motion includes stops, turns, backing up, and irregular trajectories that are not well modeled by simple constant-velocity assumptions.
- Limitation: Diffusion components may be heavier than classical trackers and should be benchmarked on edge hardware.

## P4. Towards Generalizable Multi-Object Tracking

- Venue/year: CVPR 2024.
- Local PDF: `literature_review/papers/2024_CVPR_Generalizable_MOT.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html
- Core idea: Studies tracking generalization across scenarios and proposes GeneralTrack.
- Relevance: A parking occupancy system may be deployed across different camera heights, lot layouts, weather conditions, and traffic densities.
- Limitation: Generalization is still evaluated on standard MOT benchmarks, not parking-lot-specific data.

## P5. Focusing on Tracks for Online Multi-Object Tracking

- Venue/year: CVPR 2025.
- Local PDF: `literature_review/papers/2025_CVPR_TrackTrack.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html
- Core idea: TrackTrack performs track-perspective association and track-aware initialization to improve online MOT.
- Relevance: Online tracking is important for real-time lot monitoring where occupancy must update continuously.
- Limitation: The paper optimizes identity association, while parking occupancy also requires stable mapping between vehicles and parking-slot polygons.

## P6. Multiple Object Tracking as ID Prediction

- Venue/year: CVPR 2025.
- Local PDF: `literature_review/papers/2025_CVPR_MOTIP.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html
- Core idea: Reformulates association as in-context ID prediction with a simple end-to-end framework.
- Relevance: Reduces reliance on hand-designed association rules and may adapt better when parking-lot camera geometry differs from generic MOT data.
- Limitation: Requires strong object-level features and may need domain-specific training data for parking scenes.

## P7. Towards RAW Object Detection in Diverse Conditions

- Venue/year: CVPR 2025.
- Local PDF: `literature_review/papers/2025_CVPR_AODRaw.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_RAW_Object_Detection_in_Diverse_Conditions_CVPR_2025_paper.html
- Core idea: Introduces AODRaw, a RAW-image object detection benchmark with diverse lighting and weather conditions, and explores RAW pretraining with distillation.
- Relevance: Directly targets detection under daylight, low light, rain, and fog, matching the environmental concerns of parking surveillance.
- Limitation: Many deployed surveillance cameras expose compressed video rather than RAW data, so the main lesson may be robustness-oriented data design rather than direct RAW deployment.

## P8. OW-OVD: Unified Open World and Open Vocabulary Object Detection

- Venue/year: CVPR 2025.
- Local PDF: `literature_review/papers/2025_CVPR_OW-OVD.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2025/html/Xi_OW-OVD_Unified_Open_World_and_Open_Vocabulary_Object_Detection_CVPR_2025_paper.html
- Core idea: Unifies open-world and open-vocabulary detection to detect unknown objects while retaining zero-shot class flexibility.
- Relevance: Parking lots may contain unexpected obstacles, temporary signage, construction materials, or unusual vehicle types.
- Limitation: Open-world methods can increase false positives if unknown-object scoring is not calibrated for a fixed surveillance scene.

## Auxiliary A. Resilient Sensor Fusion Under Adverse Sensor Failures via Multi-Modal Expert Fusion

- Venue/year: CVPR 2025.
- Local PDF: `literature_review/papers/2025_CVPR_MoME.pdf`
- Official page: https://openaccess.thecvf.com/content/CVPR2025/html/Park_Resilient_Sensor_Fusion_Under_Adverse_Sensor_Failures_via_Multi-Modal_Expert_CVPR_2025_paper.html
- Use as auxiliary reference for robust perception design, especially if the project later expands beyond RGB cameras.

## Auxiliary B. Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions

- Venue/year: ICCV 2025.
- Local PDF: `literature_review/papers/2025_ICCV_WeatherAware_Segmentation.pdf`
- Official page: https://openaccess.thecvf.com/content/ICCV2025/html/Pan_Exploring_Weather-aware_Aggregation_and_Adaptation_for_Semantic_Segmentation_under_Adverse_ICCV_2025_paper.html
- Use as auxiliary reference for weather-aware adaptation, especially for segmenting parking spaces, road surface, or drivable regions under adverse conditions.

