# Coursework-Aligned Literature Review: Automated Parking Lot Occupancy and Tracking System

## Abstract

This literature review examines computer vision approaches for automated parking lot occupancy detection and tracking. The coursework brief requires a comparison of existing systems or algorithms and an explanation of commonly used performance metrics. Therefore, this revised review combines two groups of literature. The first group consists of parking-specific systems, including slot-patch classification, automatic parking-slot detection, YOLO-based parking detection, and ROI-based smart parking systems. The second group consists of recent high-level computer vision methods that can support implementation, including open-vocabulary detection, online multi-object tracking, low-light tracking, and adverse-condition object detection. The review concludes that a practical system should combine vehicle/object detection, slot-region mapping, optional multi-object tracking, and temporal smoothing. Evaluation should prioritize slot-level occupancy metrics such as accuracy, precision, recall, F1-score, false free rate, and transition latency, while also reporting detector, tracker, and runtime metrics.

## 1. Problem Context

An Automated Parking Lot Occupancy and Tracking System aims to analyze surveillance or overhead camera footage and determine whether each parking space is occupied or available. This is a practical computer vision problem because camera-based systems can be cheaper and easier to retrofit than sensor-based parking systems. However, the task is not simply object detection. A complete system must solve several connected problems:

- identify vehicles and relevant obstacles in each frame
- define or detect parking-space regions
- assign detected vehicles or objects to specific parking spaces
- stabilize occupancy predictions over time
- handle lighting, weather, occlusion, shadows, and camera viewpoint changes
- evaluate performance quantitatively

The coursework brief specifically asks the literature review to cover existing computer vision systems or algorithms and the performance metrics commonly used in the field. For this reason, this review places direct parking occupancy papers before more general top-conference detection and tracking methods.

## 2. Existing Parking Occupancy Solutions

### 2.1 Slot-Patch Classification

One common solution is to crop each parking space into a patch and classify the patch as occupied or vacant. This is the approach used by many PKLot and CNRPark-EXT style systems. Improved MobileNetV3 [2] is a recent example. It processes individual parking-space patches from video feeds and classifies each patch as occupied or available. The paper reports strong performance on PKLot and CNRPark-EXT, including high AUC and an average accuracy of 98.01% on combined datasets.

The strength of patch classification is that the output directly matches the final task: each parking space receives a binary label. It is easy to evaluate with accuracy, precision, recall, F1-score, and AUC. It is also relatively simple to implement in Python and OpenCV once the parking-space regions are known.

The limitation is that it requires predefined parking slot locations or cropped patches. If the camera moves, is replaced, or has a different angle, the slot definitions may need to be manually updated. Patch classification also treats each frame independently unless a temporal smoothing module is added.

### 2.2 Automatic Slot Detection and Occupancy Classification

APSD-OC [1] addresses one major weakness of patch classification: the need to manually label parking-slot locations. The paper proposes automatic parking-slot detection using vehicle detections from a sequence of images. The detected vehicle positions are transformed into a bird's-eye view and clustered to infer parking-slot positions. Once slots are detected, a ResNet34 classifier determines whether each slot is occupied or vacant.

This is highly relevant to the project because it connects detection, parking-slot geometry, and occupancy classification in one pipeline. It also evaluates the method on PKLot and CNRPark-EXT, two standard public datasets for parking occupancy.

The limitation is that automatic slot discovery requires enough observations of vehicles occupying different spaces. It may also be harder to implement than manually defining polygons. For a coursework project, a practical compromise would be to manually define parking-space polygons first, then mention APSD-OC as a more automated future improvement.

### 2.3 YOLO-Based Parking Detection

Several recent parking papers use YOLO-style object detection. Car Parking Space Detection Using YOLOv8 [4] detects cars and available spaces from a custom parking dataset collected from camera footage. The paper compares YOLOv8 and YOLOv5 using precision, recall, mAP@0.5, and mAP@0.5:0.95. This paper is useful because it is close to a straightforward implementation route: train or fine-tune a YOLO model to detect cars and available spaces directly.

CMCA-YOLO [3] is another YOLO-based parking surveillance paper. It proposes a real-time detection model optimized for small and overlapping objects in parking lot imagery, using cross-attention and multi-spectral channel attention. The paper constructs a custom parking-lot scene dataset with 4502 images and evaluates the model using precision, recall, AP, mAP, and speed-related considerations. Its focus on small and overlapping targets is relevant because surveillance cameras often view vehicles from a distance.

Optimizing YOLOv8 for Parking Space Detection [6] compares YOLOv8 backbone choices such as ResNet-18, VGG16, EfficientNetV2, and Ghost on PKLot. It is useful for understanding trade-offs between accuracy and computational efficiency. However, as an arXiv paper, it should be treated as a supporting source rather than the strongest peer-reviewed evidence.

YOLO-based approaches are attractive for our project because they are easy to implement with Python, OpenCV, and existing deep learning libraries. They also provide standard detection metrics. Their weakness is that object detection alone does not guarantee stable slot-level occupancy. A detected car must still be mapped to a parking slot, and predictions may flicker across frames.

### 2.4 ROI-Based Smart Parking Systems

Smart Parking with Pixel-Wise ROI Selection [5] combines YOLO-family detectors with pixel-wise ROI post-processing. Instead of relying only on detection boxes, it uses a region-of-interest strategy to determine whether vehicles fall inside relevant parking areas. It compares YOLOv8, YOLOv9, YOLOv10, and YOLOv11 and reports balanced accuracy and inference-time results on a custom dataset.

This type of work is especially relevant to the project because it resembles the likely implementation pipeline:

```text
camera frame
→ vehicle detection
→ ROI or slot-region filtering
→ occupancy decision
→ real-time performance reporting
```

The limitation is that ROI design can still be camera-specific, and preprint results should be interpreted carefully. Still, the general idea of combining object detection with ROI post-processing is a strong fit for the coursework implementation.

## 3. Modern Computer Vision Modules for Implementation

The parking-specific literature explains how existing systems approach the task. However, it may not provide the strongest recent methods for detection, tracking, and robustness. The following papers are therefore used as implementation-oriented references rather than direct parking occupancy solutions.

### 3.1 Flexible Object Detection

YOLO-World [7] introduces real-time open-vocabulary object detection. Its value for this project is that it can detect categories specified by text prompts, such as cars, trucks, motorcycles, cones, barriers, and other obstacles. It should not be described as a parking occupancy system by itself. Instead, it can serve as a flexible detector inside a parking pipeline.

OW-OVD [8] extends this idea by addressing both open-world and open-vocabulary detection. This is useful for abnormal occupancy cases, where a parking space is blocked by an object that is not a standard vehicle class.

### 3.2 Multi-Object Tracking

Frame-by-frame occupancy decisions can flicker because of missed detections, shadows, or occlusions. Multi-object tracking can stabilize the system by linking detected vehicles across time.

TrackTrack [9] and MOTIP [10] are recent online MOT methods. TrackTrack focuses on track-perspective association, while MOTIP formulates association as ID prediction. DiffMOT [11] adds stronger nonlinear motion prediction, which is relevant when vehicles reverse, turn, stop, or maneuver into spaces. Generalizable MOT [12] is relevant when deploying across different parking lots or camera views.

For this coursework project, it may not be necessary to fully reproduce these advanced trackers. A simpler tracker can be implemented first, but these papers justify the design choice of adding temporal tracking rather than relying only on single-frame detection.

### 3.3 Low-Light and Weather Robustness

Parking lots operate under changing conditions. Multi-Object Tracking in the Dark [13] directly addresses low-light MOT and introduces the LMOT dataset. AODRaw [14] addresses object detection under diverse lighting and weather conditions such as low light, rain, and fog. These papers are not parking-specific, but they are useful for defining robustness requirements and evaluation splits.

For the project, their main lesson is that results should be reported separately across conditions if data is available:

- daytime
- nighttime
- rainy/cloudy
- shadow
- glare/reflection

## 4. Comparison of Existing Solution Types

| Category | Representative papers | Core idea | Common datasets | Common metrics | Strengths | Limitations | Suitability for our project |
|---|---|---|---|---|---|---|---|
| Slot-patch classification | Improved MobileNetV3; PKLot/CNRPark-style methods | Crop each parking slot and classify occupied/vacant | PKLot, CNRPark-EXT | Accuracy, precision, recall, F1, AUC | Directly predicts occupancy; simple to evaluate | Requires predefined slot crops; weak temporal reasoning | Good baseline |
| Automatic slot detection + classification | APSD-OC | Infer slot positions from vehicle detections, then classify slots | PKLot, CNRPark-EXT | Slot detection precision/recall, classification accuracy | Reduces manual slot labeling | More complex; needs enough vehicle observations | Very relevant but may be future work |
| YOLO-based parking detection | YOLOv8 parking detection; CMCA-YOLO; Optimized YOLOv8 | Detect cars, empty spaces, or parking-related objects | Custom datasets, PKLot | Precision, recall, mAP, FPS | Real-time and implementable | Needs bounding-box labels; may flicker | Strong implementation baseline |
| ROI + detector smart parking | Pixel-wise ROI + YOLO family | Detect vehicles, then use ROI/slot regions for occupancy | Custom dataset | Balanced accuracy, inference time | Similar to practical deployment | ROI is camera-specific | Recommended project direction |
| Detector + tracker + slot mapping | Proposed direction using YOLO/YOLO-World + MOT | Detect vehicles/obstacles, track over time, map to slots | Custom video plus public support datasets | Slot F1, false free rate, latency, FPS | Stable video-based occupancy | More implementation effort | Best final direction |

## 5. Datasets

The literature shows that no single dataset perfectly matches the proposed project. Relevant datasets fall into three groups.

### 5.1 Parking Occupancy Datasets

PKLot is a classic dataset for parking-space classification, with images from different parking lots and weather conditions such as sunny, cloudy/overcast, and rainy scenes. CNRPark-EXT is another widely used dataset for visual parking occupancy detection, containing roughly 150,000 labeled patches of vacant and occupied spaces. These datasets are suitable for slot-patch classification experiments and for comparing with existing parking occupancy methods.

### 5.2 Parking Detection or Counting Datasets

Some papers use custom parking datasets collected from surveillance cameras. For example, CMCA-YOLO uses a custom 4502-image parking-lot scene dataset, while the YOLOv8 parking detection paper uses a custom UTA'45 Jakarta parking dataset. CARPK and PUCPR+ are useful for parking-lot vehicle counting and aerial or high-angle vehicle detection, but they do not directly provide slot-level occupancy labels.

### 5.3 Robustness and Tracking Datasets

LMOT is useful for low-light tracking, while AODRaw is useful for adverse lighting and weather object detection. These datasets do not directly provide parking occupancy labels, but they can guide robustness testing.

For the actual coursework implementation, the most practical approach is likely to create a small custom test set from selected parking-lot video clips. This custom set should include parking-space polygons and frame-level or interval-level occupancy labels.

## 6. Performance Metrics and Evaluation

The coursework brief emphasizes quantitative evaluation. Therefore, the literature review should identify metrics at four levels.

### 6.1 Slot-Level Occupancy Metrics

These are the most important because the final system output is whether each parking slot is occupied or available.

- Accuracy: overall percentage of correctly predicted slot states.
- Precision: among predicted occupied slots, how many are truly occupied.
- Recall: among truly occupied slots, how many are detected.
- F1-score: balance between precision and recall.
- False occupied rate: available spaces incorrectly marked as occupied.
- False free rate: occupied spaces incorrectly marked as available.
- Occupancy transition latency: delay between real state change and predicted state change.

For a parking guidance system, false free rate is especially important. Sending a driver to a supposedly free space that is actually occupied is a serious failure.

### 6.2 Detection Metrics

If the system uses YOLO or another detector, it should report:

- IoU
- precision
- recall
- AP
- mAP@0.5
- mAP@0.5:0.95

These metrics are standard in YOLO-based parking detection papers and modern object detection papers.

### 6.3 Tracking Metrics

If vehicle tracking is implemented and track labels are available, standard MOT metrics include:

- IDF1
- MOTA
- HOTA
- ID switches
- track fragmentation

If track labels are not available, the project can still evaluate temporal stability using occupancy flicker count, which measures how often a slot rapidly switches between occupied and available without a true state change.

### 6.4 Runtime Metrics

Because the system is intended for real-time surveillance, runtime should also be reported:

- FPS
- average processing time per frame
- end-to-end latency
- memory usage
- optional multi-camera throughput

These metrics are important because a high-accuracy method may be unsuitable if it is too slow for real-time use.

## 7. Proposed Coursework Pipeline

Based on the reviewed literature, the most suitable project direction is not a pure patch classifier and not a pure detector. Instead, the project should combine existing algorithms into a complete pipeline:

```text
Input video frame
→ define parking-space polygons
→ detect vehicles and obstacles
→ optionally track detected objects across frames
→ assign detections/tracks to slot polygons
→ apply temporal smoothing
→ output occupied/available state per parking slot
```

This design is justified by the literature:

- Patch classifiers show that slot-level occupancy is the correct final target.
- APSD-OC shows the importance of connecting vehicle detections with parking-slot geometry.
- YOLO-based parking papers show that real-time object detection is practical for parking scenes.
- ROI-based smart parking papers show how region filtering can turn detection into occupancy status.
- MOT papers justify temporal tracking to reduce flicker.
- Low-light and weather papers justify robustness evaluation.

## 8. Research Gaps

The main gap in existing work is that many systems solve only one part of the parking occupancy problem:

- Patch classifiers classify slots but need predefined crops and usually ignore video tracking.
- YOLO-based systems detect vehicles or spaces but may not stabilize occupancy over time.
- Slot detection methods reduce manual labeling but are more complex and may require multiple observations.
- Advanced MOT and open-vocabulary detection papers provide strong modules but are not parking-specific.
- Robustness to night, rain, shadows, glare, and camera changes is often not evaluated in a unified parking occupancy pipeline.

Therefore, the proposed project can contribute by combining known techniques into an integrated system and evaluating it at the slot level. Even if the final performance does not exceed every existing method, the coursework brief allows this as long as the reasons are analyzed clearly.

## 9. Conclusion

The revised literature review supports a clear implementation plan. Parking-specific papers show that the core task is slot-level occupied/vacant classification and that common metrics include accuracy, precision, recall, F1-score, AUC, mAP, and inference time. Recent top-conference papers show how the system can be strengthened with flexible object detection, tracking, and robustness-aware design.

For the coursework project, the most defensible approach is a detector-plus-ROI/slot-mapping pipeline with optional tracking and temporal smoothing. Evaluation should prioritize slot-level metrics, especially F1-score and false free rate, while also reporting detector performance and runtime. This structure satisfies the coursework requirement to compare existing systems, identify design considerations, and learn how to quantitatively evaluate the final system.

## References

[1] R. Grbic and B. Koch, "Automatic Vision-Based Parking Slot Detection and Occupancy Classification," Expert Systems with Applications, vol. 225, 120147, 2023.

[2] Y. Yuldashev, M. Mukhiddinov, A. B. Abdusalomov, R. Nasimov, and J. Cho, "Parking Lot Occupancy Detection with Improved MobileNetV3," Sensors, vol. 23, no. 17, 7642, 2023.

[3] N. Zhao, K. Wang, J. Yang, F. Luan, L. Yuan, and H. Zhang, "CMCA-YOLO: A Study on a Real-Time Object Detection Model for Parking Lot Surveillance Imagery," Electronics, vol. 13, no. 8, 1557, 2024.

[4] M. Sobirin, Tiorivaldi, and C. Mufit, "Car Parking Space Detection Using YOLOv8," Proceedings of the 4th International Seminar and Call for Paper, pp. 394-398, 2024.

[5] G. P. C. P. da Luz, G. M. Sato, L. F. G. Gonzalez, and J. F. Borin, "Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11," arXiv:2412.01983, 2024.

[6] A. Pokhrel and G. Dao, "Optimizing YOLOv8 for Parking Space Detection: Comparative Analysis of Custom YOLOv8 Architecture," arXiv:2505.17364, 2025.

[7] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," CVPR, 2024.

[8] X. Xi, Y. Huang, R. Luo, and Y. Qiu, "OW-OVD: Unified Open World and Open Vocabulary Object Detection," CVPR, 2025.

[9] K. Shim, K. Ko, Y. Yang, and C. Kim, "Focusing on Tracks for Online Multi-Object Tracking," CVPR, 2025.

[10] R. Gao, J. Qi, and L. Wang, "Multiple Object Tracking as ID Prediction," CVPR, 2025.

[11] W. Lv, Y. Huang, N. Zhang, R.-S. Lin, M. Han, and D. Zeng, "DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction," CVPR, 2024.

[12] Z. Qin, L. Wang, S. Zhou, P. Fu, G. Hua, and W. Tang, "Towards Generalizable Multi-Object Tracking," CVPR, 2024.

[13] X. Wang, K. Ma, Q. Liu, Y. Zou, and Y. Fu, "Multi-Object Tracking in the Dark," CVPR, 2024.

[14] Z.-Y. Li, X. Jin, B.-Y. Sun, C.-L. Guo, and M.-M. Cheng, "Towards RAW Object Detection in Diverse Conditions," CVPR, 2025.

---

# 中文版本：与 Coursework 要求对齐的文献综述

## 摘要

本文献综述讨论了用于自动停车场占用检测与车辆跟踪的计算机视觉方法。Coursework 要求文献综述不仅要介绍相关算法，还要比较已有系统，并说明该领域常用的性能评价指标。因此，这一版综述将文献分成两类：第一类是直接面向停车场占用检测的系统，例如车位图像块分类、自动车位检测、基于 YOLO 的停车检测，以及基于 ROI 的智能停车系统；第二类是可以支持系统实现的现代计算机视觉方法，例如开放词汇检测、在线多目标跟踪、弱光跟踪和复杂天气下的目标检测。综述认为，一个实际可行的系统应结合车辆/障碍物检测、车位区域映射、可选的多目标跟踪和时间平滑机制。系统验收应优先使用车位级占用指标，例如 accuracy、precision、recall、F1-score、false free rate 和 occupancy transition latency，同时也应报告检测器、跟踪器和运行速度相关指标。

## 1. 问题背景

Automated Parking Lot Occupancy and Tracking System 的目标是分析监控摄像头或俯视摄像头拍摄的视频，并判断每个停车位当前是被占用还是空闲。相比在每个停车位安装硬件传感器，基于摄像头的系统通常成本更低，也更容易部署到已有停车场中。

不过，这个任务并不只是简单的目标检测。一个完整系统需要解决多个相互关联的问题：

- 在每一帧中识别车辆和相关障碍物；
- 定义或检测停车位区域；
- 将检测到的车辆或物体分配到具体停车位；
- 在时间维度上稳定占用状态预测；
- 处理光照、天气、遮挡、阴影和摄像头视角变化；
- 使用定量指标评估系统表现。

Coursework 明确要求 literature review 覆盖已有计算机视觉系统或算法，并说明该领域常用的性能指标。因此，本综述先讨论直接面向停车场占用检测的论文，再讨论更通用的顶会检测和跟踪方法。

## 2. 已有停车场占用检测方案

### 2.1 车位图像块分类

一种常见方案是先把每个停车位裁剪成一个小图像块，然后将该图像块分类为 occupied 或 vacant。许多基于 PKLot 和 CNRPark-EXT 的系统都采用这种思路。Improved MobileNetV3 [2] 是一个较新的例子。该方法从实时视频中提取单个停车位图像块，再判断每个车位是否被占用。论文在 PKLot 和 CNRPark-EXT 上取得了较强表现，包括较高 AUC，以及在合并数据集上 98.01% 的平均 accuracy。

这种方法的优点是输出结果直接对应最终任务：每个停车位得到一个二分类标签。它也很容易用 accuracy、precision、recall、F1-score 和 AUC 进行评估。只要停车位区域已经确定，用 Python 和 OpenCV 实现也相对直接。

它的局限是需要预先知道停车位位置，或者需要已经裁剪好的车位图像块。如果摄像头移动、更换或角度变化，车位定义可能需要重新标注。另外，单帧图像块分类本身没有时间连续性，除非额外加入 temporal smoothing。

### 2.2 自动车位检测与占用分类

APSD-OC [1] 解决了图像块分类中的一个重要问题：手动标注停车位位置。该论文提出通过一系列图像中的车辆检测结果自动推断停车位位置。方法大致是：先检测车辆，将车辆位置变换到鸟瞰视角，再通过聚类推断停车位位置。车位位置确定后，再使用 ResNet34 分类器判断每个车位是 occupied 还是 vacant。

这篇论文与本项目高度相关，因为它把车辆检测、停车位几何和占用分类连接成了一个完整流程。它还在 PKLot 和 CNRPark-EXT 这两个标准公开数据集上进行了评估。

它的局限是，自动发现车位需要足够多车辆停放在不同车位中的历史观测。如果数据不够丰富，车位推断可能不稳定。此外，它比手动定义车位多边形更复杂。对于 coursework 项目来说，一个更实际的折中方案是先手动定义停车位 polygons，再把 APSD-OC 作为未来自动化改进方向。

### 2.3 基于 YOLO 的停车检测

近年的一些停车场论文使用 YOLO 风格的目标检测器。Car Parking Space Detection Using YOLOv8 [4] 使用自采集停车场数据检测车辆和可用车位，并用 precision、recall、mAP@0.5 和 mAP@0.5:0.95 比较 YOLOv8 与 YOLOv5。这篇论文接近一个直接可实现的路线：训练或微调 YOLO 模型，让它直接检测 cars 和 available spaces。

CMCA-YOLO [3] 也是一篇面向停车场监控的 YOLO 改进论文。它提出了一个面向停车场监控场景的小目标和重叠目标检测模型，引入 cross-attention 和 multi-spectral channel attention。该论文构建了一个包含 4502 张图像的自定义停车场数据集，并使用 precision、recall、AP、mAP 和速度相关指标进行评估。它关注小目标和重叠目标，这一点对高位监控摄像头下的车辆检测很有意义。

Optimizing YOLOv8 for Parking Space Detection [6] 比较了 YOLOv8 与不同 backbone 组合在 PKLot 上的表现，例如 ResNet-18、VGG16、EfficientNetV2 和 Ghost。这篇论文有助于理解 accuracy 与 computational efficiency 之间的权衡。不过，由于它是 arXiv 论文，应作为辅助参考，而不是最强的同行评审证据。

基于 YOLO 的方案对本项目很有吸引力，因为它们容易用 Python、OpenCV 和现有深度学习库实现，也能提供标准检测指标。它们的问题是，仅有目标检测并不能保证稳定的车位级占用判断。检测到车辆之后，还必须判断该车辆属于哪个停车位，并且单帧检测结果可能在视频中抖动。

### 2.4 基于 ROI 的智能停车系统

Smart Parking with Pixel-Wise ROI Selection [5] 将 YOLO 系列检测器与 pixel-wise ROI 后处理结合起来。它不是只依赖检测框，而是使用 ROI 策略判断车辆是否落在相关停车区域内。该论文比较了 YOLOv8、YOLOv9、YOLOv10 和 YOLOv11，并在自定义数据集上报告 balanced accuracy 和 inference time。

这类工作与本项目特别接近，因为它类似于我们可能采用的实现流程：

```text
camera frame
-> vehicle detection
-> ROI or slot-region filtering
-> occupancy decision
-> real-time performance reporting
```

它的局限是 ROI 设计仍然依赖具体摄像头位置和停车场布局，且 preprint 结果需要谨慎解读。不过，将目标检测与 ROI 后处理结合起来的思想非常适合 coursework 实现。

## 3. 支持实现的现代计算机视觉模块

直接面向停车场的文献说明了已有系统如何设计，但它们未必包含最新、最强的检测、跟踪和鲁棒性方法。因此，以下顶会论文应被视为实现模块参考，而不是直接的停车场占用检测系统。

### 3.1 灵活目标检测

YOLO-World [7] 提出了实时开放词汇目标检测。它对本项目的价值在于，可以根据文本提示检测指定类别，例如 cars、trucks、motorcycles、cones、barriers 和其他障碍物。它不应该被描述成一个完整的停车场占用系统，而应该被视为停车场 pipeline 中的灵活检测器。

OW-OVD [8] 进一步结合了 open-world 和 open-vocabulary detection。它适合处理异常占用情况，例如某个停车位被非标准车辆类别的物体挡住。

### 3.2 多目标跟踪

逐帧占用判断容易受到漏检、阴影和遮挡影响，导致车位状态在 occupied 和 available 之间快速跳变。多目标跟踪可以通过跨帧连接车辆检测结果来稳定系统输出。

TrackTrack [9] 和 MOTIP [10] 是近年的在线多目标跟踪方法。TrackTrack 强调从轨迹角度进行关联，MOTIP 将目标关联建模为 ID prediction。DiffMOT [11] 加入了更强的非线性运动预测，这对车辆倒车、转弯、停车和驶入车位等动作很有意义。Generalizable MOT [12] 则与跨停车场或跨摄像头视角部署相关。

对于 coursework 项目来说，未必需要完整复现这些高级 tracker。可以先实现一个较简单的 tracker 或时间平滑方法，但这些论文可以用来论证为什么本系统不应只依赖单帧检测，而应加入时间信息。

### 3.3 弱光和天气鲁棒性

停车场会受到光照和天气变化影响。Multi-Object Tracking in the Dark [13] 直接研究弱光多目标跟踪，并提出 LMOT 数据集。AODRaw [14] 研究低光、雨、雾等复杂条件下的目标检测。这些论文不是停车场专用论文，但可以帮助定义系统的鲁棒性需求和评估划分。

对本项目来说，它们的主要启发是：如果数据允许，结果应该按不同环境条件分别报告，例如：

- 白天；
- 夜间；
- 雨天或阴天；
- 阴影；
- 眩光或反光。

## 4. 已有方案类型对比

| 类型 | 代表论文 | 核心思想 | 常用数据集 | 常用指标 | 优点 | 局限 | 对本项目的适用性 |
|---|---|---|---|---|---|---|---|
| 车位图像块分类 | Improved MobileNetV3；PKLot/CNRPark 方法 | 裁剪每个车位并分类 occupied/vacant | PKLot, CNRPark-EXT | Accuracy, precision, recall, F1, AUC | 直接预测车位状态，评价简单 | 需要预定义车位裁剪，时间连续性弱 | 适合作为 baseline |
| 自动车位检测 + 分类 | APSD-OC | 从车辆检测结果推断车位位置，再分类车位状态 | PKLot, CNRPark-EXT | 车位检测 precision/recall，分类 accuracy | 减少手动车位标注 | 更复杂，需要足够车辆观测 | 非常相关，但可作为未来工作 |
| 基于 YOLO 的停车检测 | YOLOv8 parking detection；CMCA-YOLO；Optimized YOLOv8 | 检测车辆、空车位或停车相关对象 | 自定义数据集, PKLot | Precision, recall, mAP, FPS | 实时、容易实现 | 需要框标注，结果可能抖动 | 强实现 baseline |
| ROI + 检测器智能停车 | Pixel-wise ROI + YOLO 系列 | 先检测车辆，再用 ROI/车位区域判断占用 | 自定义数据集 | Balanced accuracy, inference time | 接近实际部署 | ROI 依赖摄像头设置 | 推荐项目方向 |
| 检测器 + 跟踪 + 车位映射 | 使用 YOLO/YOLO-World + MOT 的 proposed direction | 检测车辆/障碍物，跨帧跟踪，再映射到车位 | 自定义视频 + 公开辅助数据集 | Slot F1, false free rate, latency, FPS | 视频结果更稳定 | 实现工作量更大 | 最适合作为最终方向 |

## 5. 数据集

文献显示，没有一个公开数据集能完全覆盖本项目的全部需求。相关数据集大致可以分为三类。

### 5.1 停车位占用数据集

PKLot 是经典的停车位分类数据集，包含不同停车场和不同天气条件下的图像，例如 sunny、cloudy/overcast 和 rainy。CNRPark-EXT 也是常用的视觉停车占用检测数据集，包含约 150,000 个标注好的 occupied/vacant 车位图像块。这些数据集适合用于车位图像块分类实验，也适合与已有停车占用方法进行比较。

### 5.2 停车检测或计数数据集

一些论文使用从监控摄像头采集的自定义停车场数据集。例如，CMCA-YOLO 使用包含 4502 张图像的自定义停车场场景数据集；YOLOv8 停车检测论文使用 UTA'45 Jakarta 停车场数据集。CARPK 和 PUCPR+ 适合停车场车辆计数和高位/航拍车辆检测，但它们不直接提供车位级占用标签。

### 5.3 鲁棒性和跟踪数据集

LMOT 对弱光车辆跟踪有参考价值，AODRaw 对复杂光照和天气条件下的目标检测有参考价值。它们不直接提供停车位占用标签，但可以帮助设计鲁棒性测试。

对于 coursework 实现，最实际的方法可能是从选定停车场视频中创建一个小型自定义测试集。这个测试集应包含停车位 polygons，以及帧级或时间段级的 occupied/available 标签。

## 6. 性能指标与系统验收

Coursework 明确强调定量评价。因此，文献综述应识别四个层面的评价指标。

### 6.1 车位级占用指标

这是最重要的一类指标，因为系统最终输出的是每个停车位是否被占用。

- Accuracy：所有车位状态中预测正确的比例。
- Precision：预测为 occupied 的车位中，有多少确实 occupied。
- Recall：真实 occupied 的车位中，有多少被检测出来。
- F1-score：precision 和 recall 的平衡。
- False occupied rate：空车位被错误标记为 occupied 的比例。
- False free rate：已占用车位被错误标记为 available 的比例。
- Occupancy transition latency：真实状态变化到系统预测变化之间的延迟。

对于停车引导系统来说，false free rate 尤其重要。因为如果系统把一个已占用车位错误标记为空闲，用户可能会被引导到不可用车位。

### 6.2 检测指标

如果系统使用 YOLO 或其他目标检测器，应报告：

- IoU；
- precision；
- recall；
- AP；
- mAP@0.5；
- mAP@0.5:0.95。

这些指标是 YOLO 停车检测论文和现代目标检测论文中常见的标准指标。

### 6.3 跟踪指标

如果实现车辆跟踪，并且有 track labels，可以使用标准 MOT 指标：

- IDF1；
- MOTA；
- HOTA；
- ID switches；
- track fragmentation。

如果没有 track labels，仍然可以用 occupancy flicker count 评价时间稳定性。它表示某个车位在没有真实状态变化时，预测状态快速来回切换的次数。

### 6.4 运行性能指标

由于系统面向实时监控，还应报告运行速度：

- FPS；
- average processing time per frame；
- end-to-end latency；
- memory usage；
- optional multi-camera throughput。

这些指标很重要，因为一个高准确率模型如果运行太慢，也不适合实时停车场系统。

## 7. 建议的 Coursework 系统流程

根据综述结果，最适合本项目的方向不是纯图像块分类，也不是纯目标检测，而是把多个已有算法组合成完整 pipeline：

```text
Input video frame
-> define parking-space polygons
-> detect vehicles and obstacles
-> optionally track detected objects across frames
-> assign detections/tracks to slot polygons
-> apply temporal smoothing
-> output occupied/available state per parking slot
```

这个设计可以由文献支持：

- 图像块分类论文说明车位级 occupied/vacant 是最终目标；
- APSD-OC 说明车辆检测与停车位几何之间的连接很重要；
- YOLO 停车论文说明实时目标检测适合停车场场景；
- ROI-based smart parking 论文说明区域过滤可以把检测结果转化为占用状态；
- MOT 论文说明时间跟踪可以减少状态抖动；
- 弱光和天气论文说明系统需要鲁棒性评价。

## 8. 研究空白

现有研究的主要空白在于，许多系统只解决了停车占用问题的一部分：

- 车位图像块分类能分类车位，但需要预定义裁剪区域，并且通常忽略视频跟踪；
- YOLO 系统能检测车辆或车位，但未必能稳定跨帧维护占用状态；
- 自动车位检测方法减少了手动标注，但实现更复杂，并且需要足够多车辆观测；
- 高级 MOT 和开放词汇检测论文提供了强模块，但不是停车场专用方案；
- 夜间、雨天、阴影、眩光和摄像头变化下的鲁棒性，通常没有在统一停车占用 pipeline 中充分评估。

因此，本项目可以通过整合已有技术形成一个完整系统，并在车位级别进行评价。即使最终性能没有超过所有已有方法，只要能清楚分析原因，也符合 coursework 对项目评估的要求。

## 9. 结论

这一版 literature review 支持一个清晰的实现计划。直接停车场论文表明，本任务的核心输出是车位级 occupied/vacant 分类，常用指标包括 accuracy、precision、recall、F1-score、AUC、mAP 和 inference time。近年的顶会论文则说明，系统可以通过灵活目标检测、车辆跟踪和鲁棒性感知进一步增强。

对于 coursework 项目，最稳妥的方案是 detector + ROI/slot mapping pipeline，并可加入 tracking 和 temporal smoothing。评估应优先报告车位级指标，尤其是 F1-score 和 false free rate，同时补充检测性能和运行速度。这样的结构能够满足 coursework 对 existing systems comparison、design considerations 和 quantitative evaluation 的要求。

## 参考文献

[1] R. Grbic and B. Koch, "Automatic Vision-Based Parking Slot Detection and Occupancy Classification," Expert Systems with Applications, vol. 225, 120147, 2023.

[2] Y. Yuldashev, M. Mukhiddinov, A. B. Abdusalomov, R. Nasimov, and J. Cho, "Parking Lot Occupancy Detection with Improved MobileNetV3," Sensors, vol. 23, no. 17, 7642, 2023.

[3] N. Zhao, K. Wang, J. Yang, F. Luan, L. Yuan, and H. Zhang, "CMCA-YOLO: A Study on a Real-Time Object Detection Model for Parking Lot Surveillance Imagery," Electronics, vol. 13, no. 8, 1557, 2024.

[4] M. Sobirin, Tiorivaldi, and C. Mufit, "Car Parking Space Detection Using YOLOv8," Proceedings of the 4th International Seminar and Call for Paper, pp. 394-398, 2024.

[5] G. P. C. P. da Luz, G. M. Sato, L. F. G. Gonzalez, and J. F. Borin, "Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11," arXiv:2412.01983, 2024.

[6] A. Pokhrel and G. Dao, "Optimizing YOLOv8 for Parking Space Detection: Comparative Analysis of Custom YOLOv8 Architecture," arXiv:2505.17364, 2025.

[7] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," CVPR, 2024.

[8] X. Xi, Y. Huang, R. Luo, and Y. Qiu, "OW-OVD: Unified Open World and Open Vocabulary Object Detection," CVPR, 2025.

[9] K. Shim, K. Ko, Y. Yang, and C. Kim, "Focusing on Tracks for Online Multi-Object Tracking," CVPR, 2025.

[10] R. Gao, J. Qi, and L. Wang, "Multiple Object Tracking as ID Prediction," CVPR, 2025.

[11] W. Lv, Y. Huang, N. Zhang, R.-S. Lin, M. Han, and D. Zeng, "DiffMOT: A Real-time Diffusion-based Multiple Object Tracker with Non-linear Prediction," CVPR, 2024.

[12] Z. Qin, L. Wang, S. Zhou, P. Fu, G. Hua, and W. Tang, "Towards Generalizable Multi-Object Tracking," CVPR, 2024.

[13] X. Wang, K. Ma, Q. Liu, Y. Zou, and Y. Fu, "Multi-Object Tracking in the Dark," CVPR, 2024.

[14] Z.-Y. Li, X. Jin, B.-Y. Sun, C.-L. Guo, and M.-M. Cheng, "Towards RAW Object Detection in Diverse Conditions," CVPR, 2025.
