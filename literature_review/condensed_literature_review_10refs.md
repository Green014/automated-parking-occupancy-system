# Condensed Literature Review: Automated Parking Lot Occupancy and Tracking System

## Abstract

This condensed literature review focuses on ten key references for an Automated Parking Lot Occupancy and Tracking System. The selected papers balance two needs: direct relevance to parking occupancy systems and high-quality recent computer vision methods that can support implementation. Parking-specific studies are used to understand existing solution types, datasets, and metrics, while recent CVPR papers are used to justify modern detection, tracking, and robustness modules. To avoid relying on preprints as core evidence, this version uses peer-reviewed parking-specific papers and recent top-conference papers. The review supports a practical pipeline that combines vehicle/object detection, parking-slot ROI mapping, optional multi-object tracking, temporal smoothing, and slot-level evaluation.

## 1. Problem Context

The goal of a camera-based parking occupancy system is to determine whether each parking space is occupied or available from surveillance or overhead video. A complete system must detect vehicles or obstacles, relate them to parking-space regions, stabilize predictions over time, and evaluate results quantitatively. The coursework brief requires comparison with existing systems and explanation of commonly used performance metrics, so this review prioritizes papers that either directly address parking occupancy or provide important implementation modules.

## 2. Existing Parking Occupancy Approaches

Slot-patch classification is one common approach. Improved MobileNetV3 [2] classifies cropped parking-space patches as occupied or vacant and reports strong performance on PKLot and CNRPark-EXT. This approach is easy to evaluate because its output directly matches the final task. However, it requires predefined parking slot locations and usually treats frames independently.

APSD-OC [1] extends beyond simple patch classification by automatically detecting parking slots from vehicle detections over an image sequence and then classifying occupancy. It is highly relevant because it connects vehicle detection, slot geometry, and occupancy classification in one pipeline. Its limitation is implementation complexity: automatic slot discovery requires enough vehicle observations and a more careful geometric setup.

YOLO-based parking detection is another practical route. CMCA-YOLO [3] improves YOLO for parking-lot surveillance imagery, especially for small and overlapping objects, and evaluates the model using precision, recall, AP, mAP, and speed-related indicators. Car Parking Space Detection Using YOLOv8 [4] provides a simpler example of using YOLOv8 for parking-space detection and compares standard detection metrics such as precision, recall, mAP@0.5, and mAP@0.5:0.95. These works show that YOLO-based detectors are realistic implementation baselines, but detection alone does not guarantee stable slot-level occupancy.

Recent panoramic-image parking detection work also provides a useful implementation reference. Wu et al. [5] propose a real-time parking space detection method based on deep learning and panoramic images. The paper constructs the PSEX dataset and improves a PP-Yoloe-based model for parking corner and occupancy recognition, reporting improvements in mAP50 and mAP50:95 while maintaining real-time performance on edge hardware. This is more recent than classic parking datasets and is useful for discussing detection accuracy, real-time speed, and complex parking environments.

## 3. Modern Computer Vision Modules

YOLO-World [6] introduces real-time open-vocabulary object detection. It is not a parking occupancy system by itself, but it can serve as a flexible detector in the pipeline. For example, the system could search for cars, trucks, motorcycles, cones, barriers, or other obstacles using text prompts. This is useful because real parking lots may contain non-standard objects that still block spaces.

Multi-object tracking can stabilize video-based occupancy estimates. TrackTrack [7] is a recent online MOT method that focuses on track-level association. MOTIP [8] formulates tracking as ID prediction, reducing reliance on hand-designed association rules. These methods justify adding tracking or at least temporal smoothing to avoid rapid occupied/available flickering caused by missed detections or occlusions.

Robustness is also important. Multi-Object Tracking in the Dark [9] focuses on low-light tracking and introduces LMOT, which is relevant because parking lots often operate at night. AODRaw [10] studies object detection under diverse conditions such as low light, rain, and fog. Although it is not parking-specific, it supports evaluating the system under different lighting and weather conditions.

## 4. Comparison of Solution Types

| Solution type | Representative references | Main idea | Strengths | Limitations | Fit for this project |
|---|---|---|---|---|---|
| Slot-patch classification | Improved MobileNetV3 [2] | Crop each slot and classify occupied/vacant | Direct slot-level output; easy metrics | Needs predefined slot crops; weak temporal reasoning | Good baseline |
| Automatic slot detection + classification | APSD-OC [1] | Infer slot locations from vehicle detections, then classify occupancy | Reduces manual slot labeling | More complex; needs enough observations | Relevant future improvement |
| YOLO-based detection | CMCA-YOLO [3], YOLOv8 parking detection [4] | Detect cars, empty spaces, or parking-related objects | Real-time and implementable | Needs mapping from detections to spaces | Strong implementation baseline |
| Recent panoramic parking detection | Real-time panoramic parking detection [5] | Detect parking corners and occupancy status from panoramic images | Recent, peer-reviewed, reports mAP and FPS | Uses panoramic/AVM images rather than fixed overhead CCTV | Useful for modern parking detection and deployment discussion |
| Detector + tracker + slot mapping | YOLO-World [6], TrackTrack [7], MOTIP [8] | Detect objects, track over time, map to slot polygons | More stable video output | More implementation effort | Best final pipeline |

## 5. Evaluation Metrics

The most important evaluation level is slot-level occupancy, because the final system output is whether each parking space is available.

Recommended slot-level metrics:

- Accuracy
- Precision
- Recall
- F1-score
- False occupied rate
- False free rate
- Occupancy transition latency

False free rate is particularly important because incorrectly marking an occupied space as free may mislead users.

If a detector is trained or evaluated, object detection metrics should also be reported:

- IoU
- precision and recall
- AP
- mAP@0.5
- mAP@0.5:0.95

If tracking is implemented and track labels are available, tracking metrics such as IDF1, MOTA, HOTA, ID switches, and track fragmentation can be used. If track labels are not available, a simpler temporal stability metric such as occupancy flicker count can be used.

Runtime should also be reported because the system is intended for real-time monitoring:

- FPS
- average processing time per frame
- end-to-end latency
- memory usage

## 6. Proposed Pipeline

The most suitable implementation is a combined pipeline rather than a single isolated algorithm:

```text
Input video frame
-> define parking-space polygons
-> detect vehicles and obstacles
-> optionally track detected objects over time
-> assign detections or tracks to slot polygons
-> apply temporal smoothing
-> output occupied/available state for each slot
```

This design is supported by the literature. Parking-specific studies show that the final goal should be evaluated at the slot level. YOLO-based and panoramic detection methods show that real-time parking detection is practical. Tracking papers justify temporal stabilization, while low-light and adverse-condition papers justify robustness testing.

## 7. Research Gap

Existing methods often solve only part of the problem. Patch classifiers need predefined slot crops and usually ignore tracking. YOLO-based methods detect objects or spaces but may not provide stable video-level occupancy. Automatic slot detection reduces manual annotation but is more complex. Recent MOT and open-vocabulary methods are powerful but not parking-specific.

The proposed project can address this gap by integrating known methods into one system and evaluating it using slot-level metrics, detection metrics, optional temporal stability metrics, and runtime measurements.

## 8. Conclusion

The selected ten references are sufficient to support a coursework literature review with a clear implementation direction. Parking-specific papers provide existing systems, datasets, and metrics. Recent CVPR papers provide stronger detection, tracking, and robustness modules. The most defensible project approach is a detector-plus-ROI/slot-mapping system with optional tracking and temporal smoothing, evaluated primarily by slot-level F1-score, false free rate, and real-time performance.

## References

[1] R. Grbic and B. Koch, "Automatic Vision-Based Parking Slot Detection and Occupancy Classification," Expert Systems with Applications, vol. 225, 120147, 2023.

[2] Y. Yuldashev, M. Mukhiddinov, A. B. Abdusalomov, R. Nasimov, and J. Cho, "Parking Lot Occupancy Detection with Improved MobileNetV3," Sensors, vol. 23, no. 17, 7642, 2023.

[3] N. Zhao, K. Wang, J. Yang, F. Luan, L. Yuan, and H. Zhang, "CMCA-YOLO: A Study on a Real-Time Object Detection Model for Parking Lot Surveillance Imagery," Electronics, vol. 13, no. 8, 1557, 2024.

[4] M. Sobirin, Tiorivaldi, and C. Mufit, "Car Parking Space Detection Using YOLOv8," Proceedings of the 4th International Seminar and Call for Paper, pp. 394-398, 2024.

[5] W. Wu, H. Chen, J. Gong, K. Che, W. Ren, and B. Zhang, "Real-Time Parking Space Detection Based on Deep Learning and Panoramic Images," Sensors, vol. 25, no. 20, 6449, 2025.

[6] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," CVPR, 2024.

[7] K. Shim, K. Ko, Y. Yang, and C. Kim, "Focusing on Tracks for Online Multi-Object Tracking," CVPR, 2025.

[8] R. Gao, J. Qi, and L. Wang, "Multiple Object Tracking as ID Prediction," CVPR, 2025.

[9] X. Wang, K. Ma, Q. Liu, Y. Zou, and Y. Fu, "Multi-Object Tracking in the Dark," CVPR, 2024.

[10] Z.-Y. Li, X. Jin, B.-Y. Sun, C.-L. Guo, and M.-M. Cheng, "Towards RAW Object Detection in Diverse Conditions," CVPR, 2025.
