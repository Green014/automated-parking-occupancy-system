# Role 2: Method-Based Literature Review, System Comparison, and Design Implications

## Method-Based Review of Existing Systems and Algorithms

This section reviews existing computer vision methods for parking lot occupancy detection and tracking by method category rather than by summarizing papers one by one. The purpose is to compare how different approaches process input data, what outputs they produce, what assumptions they require, and what design lessons they provide for the proposed Automated Parking Lot Occupancy and Tracking System.

### 1. Slot-Based Occupancy Classification

Slot-based occupancy classification treats parking occupancy as a direct image classification problem. The usual input is a cropped image patch corresponding to one predefined parking space, and the output is a binary label such as occupied or vacant. This approach directly matches the final goal of a parking occupancy system because each parking slot receives its own status prediction.

Yuldashev et al. [2] follow this type of approach using an improved MobileNetV3 model for parking lot occupancy detection. Their method processes parking slot image patches from datasets such as PKLot and CNRPark-EXT and classifies each patch as occupied or vacant. The main advantage of this method category is its clear slot-level output. Since the output is already aligned with the final task, the method can be evaluated using standard classification metrics such as accuracy, precision, recall, F1-score, and AUC, although the detailed definitions of these metrics are left for Role 3.

However, the method depends on predefined parking slot locations. In practice, each slot must be manually labelled, cropped, or otherwise extracted before classification can occur. If the camera position changes, or if the system is deployed in another parking lot, these regions may need to be redefined. Another limitation is that patch classification is often frame-based. Without a temporal module, predictions may fluctuate when shadows, reflections, partial occlusion, or passing vehicles affect a cropped slot image.

For our project, slot-based classification is useful as a baseline and as a reminder that the final system output must be slot-level occupancy. However, relying only on patch classification may be too restrictive for a video-based surveillance system. Our system is likely to benefit from combining slot-level reasoning with object detection and temporal stabilization.

### 2. Automatic Parking Slot Detection and Occupancy Classification

Automatic parking slot detection methods attempt to reduce or remove the need for manual parking slot annotation. Instead of assuming that each slot polygon is already known, these systems infer parking slot positions from visual evidence.

APSD-OC [1] is an important example of this category. The method uses vehicle detections from a sequence of images and transforms detected vehicle positions into a bird's-eye view. Clustering is then used to infer parking slot positions. After the parking slots are detected, occupancy classification is performed for each slot. In other words, the system combines vehicle detection, geometric transformation, parking slot discovery, and occupancy classification.

The main advantage of this approach is that it addresses one of the practical weaknesses of patch-based systems: manual parking slot labelling. If a system can automatically infer slot positions, it becomes easier to deploy in new parking lots. This is highly relevant to real-world smart parking systems because manual annotation can be time-consuming when the number of spaces is large.

The limitation is that automatic slot detection is more complex than using manually defined slot polygons. It requires enough vehicle observations to reveal the typical parking positions, a stable camera setup, and suitable geometric processing. If a camera is moved, if the dataset contains few occupied examples for some slots, or if the parking layout is irregular, slot discovery may become unreliable.

For our project, APSD-OC [1] provides an important design lesson: parking occupancy detection is not only about recognizing vehicles, but also about connecting vehicle locations with parking space geometry. Given the limited scope of a coursework implementation, manually defining parking slot polygons may be more feasible. However, APSD-OC can be discussed as a more automated alternative and a potential future improvement.

### 3. YOLO-Based Parking or Vehicle Detection

YOLO-based methods treat parking analysis as an object detection problem. The input is usually a full image or video frame, and the output consists of bounding boxes for objects such as vehicles, parking spaces, or available spaces. These methods are attractive because YOLO models are designed for real-time detection and are widely supported by existing implementations.

Zhao et al. [3] propose CMCA-YOLO for parking lot surveillance imagery. Their work focuses on improving detection in parking scenes where vehicles may be small, dense, or overlapping. The method uses a custom parking-lot scene dataset and reports detection-oriented results such as precision, recall, AP, mAP, and speed-related performance. This paper is parking-specific and useful because it studies the surveillance setting rather than a generic object detection benchmark.

Sobirin et al. [4] provide a more direct YOLOv8 parking-space detection example. Their work compares YOLOv8 with YOLOv5 and detects parking-related targets from camera imagery. This type of system is relatively easy to understand and implement because the detector produces bounding boxes that can be inspected visually and evaluated with common object detection metrics.

Wu et al. [5] present a recent peer-reviewed parking space detection method based on deep learning and panoramic images. Their system uses a parking-specific dataset and reports both detection performance and real-time deployment considerations. Although the input scenario is panoramic or AVM-style imagery rather than a fixed overhead CCTV camera, the paper is useful because it shows how modern parking detection systems consider both detection accuracy and practical runtime.

Despite their practicality, YOLO-based detectors do not automatically solve slot-level occupancy. A detector can identify a vehicle or an available-space candidate, but the system still needs to decide which parking slot the detection belongs to. A vehicle driving through the lane may be detected by the model but should not necessarily be counted as occupying a parking space. Similarly, a partially visible vehicle may overlap several slot regions. Therefore, an additional detection-to-slot assignment step is required to convert object detections into final parking slot availability.

For our project, YOLO-based detection is a strong implementation baseline. It provides a realistic first stage for detecting vehicles or obstacles in each frame. However, it should be combined with parking slot geometry before it can produce stable occupancy output. This design lesson is supported by slot-based systems, where cropped slot patches are produced from predefined parking-space regions [2], and by APSD-OC, which shows that vehicle detections must be linked to parking slot geometry before occupancy can be determined [1]. In our implementation, this connection can be handled pragmatically by manually defining parking slot polygons and assigning detected vehicles to those polygons using overlap or center-point rules. This is not presented as a separate literature category, but as an implementation step required when adapting object detection to slot-level parking occupancy.

### 4. Tracking and Temporal Stabilization

Multi-object tracking papers should be treated as supporting computer vision modules rather than direct parking occupancy solutions. They do not by themselves prove improved parking occupancy performance. However, they provide mechanisms for maintaining object identities across frames, which may be adapted to stabilize a video-based parking occupancy system.

TrackTrack [7] is an online multi-object tracking method that focuses on track-level association. Its relevance to this project is that a parking system processes video frames sequentially and must update occupancy status continuously. A track-aware association strategy may help reduce unstable outputs when detections are briefly missed or when multiple similar vehicles appear close together.

MOTIP [8] formulates multi-object tracking as ID prediction. Instead of relying only on hand-designed matching rules, it uses contextual information to predict object identities. This is relevant to parking scenes because vehicles may move slowly, stop temporarily, or become partially occluded by other vehicles or scene structures. While MOTIP is not parking-specific, it supports the idea that temporal association can be learned or modelled more robustly than simple frame-by-frame matching.

Tracking or temporal smoothing can reduce unstable predictions caused by missed detections, passing vehicles, short-term occlusion, and frame-level classification errors. For example, if a vehicle is detected in a slot for several consecutive frames, the system can be more confident that the slot is occupied. If a vehicle appears in a slot region for only one frame while passing through a lane, temporal logic can prevent a false occupancy update.

For our project, fully implementing an advanced tracker may not be necessary at the first stage. A simpler temporal smoothing rule or lightweight tracker may be sufficient. Nevertheless, TrackTrack [7] and MOTIP [8] justify the design decision to include temporal information rather than treating each frame independently.

### 5. Robustness Under Low-Light or Adverse Conditions

Real parking lots operate under variable conditions, including night-time illumination, rain, fog, shadows, reflections, headlights, and camera noise. Robustness papers are therefore useful even when they are not parking-specific.

Wang et al. [9] study multi-object tracking in low-light conditions and introduce the LMOT setting. This is relevant because parking lots often operate at night, and both detection confidence and appearance features can degrade under poor illumination. The paper should not be presented as a parking occupancy method, but it supports the need to consider night-time monitoring when designing and evaluating the system.

Li et al. [10] study object detection under diverse conditions using RAW and sRGB data. Their work covers challenging conditions such as low light, rain, and fog. Although the method is not designed for parking lots, it highlights the importance of testing detection systems under different visual conditions rather than only in clear daytime imagery.

For our project, these papers suggest that robustness should influence system design. The detector and slot-mapping logic should be tested on frames with shadows, low illumination, glare, and weather variation where possible. If such data is not available, the limitation should be stated clearly by Role 3 in the metrics and gaps discussion.

## Design Implications for the Proposed System

The reviewed methods suggest that our project should not rely on a single isolated algorithm. Slot-based classification provides direct occupancy output but depends on predefined slot regions. Automatic slot detection reduces manual annotation but adds geometric complexity. YOLO-based detection is practical and suitable for real-time implementation, but it must be connected to parking slot geometry before it can produce final occupancy status. Tracking methods are not parking-specific, but they provide mechanisms for temporal stabilization. Robustness studies do not directly solve parking occupancy, but they show why night-time and adverse-condition testing should be considered.

Based on these design lessons, the proposed system should follow a modular pipeline:

```text
Input surveillance frame
-> vehicle or obstacle detection
-> parking slot polygon assignment
-> optional tracking or temporal smoothing
-> slot-level occupied / available output
```

This pipeline satisfies the coursework requirement to combine multiple known algorithms into a new approach. It also keeps the implementation feasible because each module can be developed and tested separately using Python and OpenCV-compatible tools.

## Comparison Table of Selected Methods and Systems

| Paper / Method | Main Task | Core Technique | Dataset / Scenario | Output | Advantages | Limitations | Relevance to Our Project |
|---|---|---|---|---|---|---|---|
| APSD-OC [1] | Automatic parking slot detection and occupancy classification | Vehicle detection, perspective transformation, bird's-eye mapping, clustering, ResNet34 classification | PKLot and CNRPark-EXT parking occupancy datasets | Detected parking slot positions and occupied/vacant labels | Reduces manual parking slot annotation; connects detection with slot geometry | Requires enough vehicle observations, geometric processing, and stable camera setup | Strong design reference for linking detections to parking-space geometry |
| Improved MobileNetV3 [2] | Parking slot occupancy classification | Lightweight CNN classification on cropped parking slot patches | PKLot and CNRPark-EXT | Slot-level occupied/vacant classification | Directly matches final occupancy task; simple evaluation and efficient inference | Needs predefined or cropped slot regions; weak temporal reasoning | Useful baseline for slot-level classification |
| CMCA-YOLO [3] | Parking lot surveillance object detection | YOLO-based detector with attention modules for parking imagery | Custom parking-lot surveillance dataset | Vehicle/object detection boxes | Parking-specific; addresses small and overlapping objects; suitable for real-time detection discussion | Detection output still needs slot assignment; custom dataset may limit generalization | Useful detector reference for surveillance-style parking scenes |
| YOLOv8 parking detection [4] | Parking space or vehicle detection | YOLOv8 object detection and comparison with YOLOv5 | Custom parking camera dataset | Detection boxes for parking-related targets | Practical and easy to implement; uses common YOLO metrics | Does not by itself ensure stable slot-level occupancy | Supports using YOLO as an implementation baseline |
| Panoramic parking detection [5] | Real-time parking space detection | Deep learning detector based on improved PP-Yoloe and panoramic images | PSEX panoramic parking dataset | Parking corner and occupancy-related detections | Recent, peer-reviewed, parking-specific, considers accuracy and runtime | Uses panoramic/AVM images rather than fixed overhead CCTV | Useful for modern parking detection and deployment considerations |
| YOLO-World [6] | Open-vocabulary object detection | Vision-language detector for text-specified object categories | General detection benchmarks, not parking-specific | Detection boxes for prompted categories | Flexible categories such as vehicle types or obstacles; real-time design | Not validated as a parking occupancy system; prompt and domain adaptation may be needed | Supporting detector module for flexible vehicle/obstacle recognition |
| TrackTrack [7] | Online multi-object tracking | Track-perspective association and track-aware initialization | Standard MOT benchmarks | Object tracks across frames | Supports continuous video processing and identity consistency | Not parking-specific; does not output slot occupancy | Supporting reference for temporal stabilization |
| MOTIP [8] | Multi-object tracking as ID prediction | Context-based identity prediction for object association | Standard MOT benchmarks | Object identities and trajectories | Reduces reliance on hand-designed association rules | Needs reliable detections and is not parking-specific | Supporting reference for learned temporal association |
| Multi-Object Tracking in the Dark [9] | Low-light multi-object tracking | Low-light tracking framework and LMOT setting | Low-light MOT scenarios | Object tracks under low illumination | Addresses night-time tracking, which is realistic for parking lots | Does not solve parking slot occupancy | Supports robustness discussion for night-time deployment |
| AODRaw [10] | Object detection under diverse conditions | RAW/sRGB detection under low light, rain, fog, and diverse scenes | AODRaw adverse-condition detection dataset | Object detection boxes | Highlights detection robustness under weather and lighting changes | Not parking-specific and may require RAW data for full method use | Supports adverse-condition testing and robustness design |

## Transition to Role 3

The reviewed methods differ in their assumptions, inputs, outputs, implementation complexity, and evaluation focus. Slot-based classifiers directly predict occupancy but require predefined slot regions; YOLO-based detectors are practical for real-time use but require a detection-to-slot assignment step; tracking methods may improve temporal stability but are mainly supporting modules; and robustness studies highlight deployment challenges rather than providing a complete parking solution. Therefore, the next section should explain the standard performance metrics in more detail and then summarize the main limitations and research gaps that remain for the proposed system.

---

# 中文版本：Role 2 方法分类文献综述、系统对比与设计启发

## 基于方法类别的已有系统与算法综述

本节按照方法类别，而不是逐篇论文罗列的方式，综述与停车场占用检测和跟踪相关的计算机视觉方法。这样组织的目的，是比较不同方法如何处理输入数据、产生什么输出、依赖哪些假设，以及它们对我们 proposed Automated Parking Lot Occupancy and Tracking System 有什么设计启发。

### 1. 基于车位图像块的占用分类

基于车位图像块的占用分类方法，将停车位占用检测看作一个直接的图像分类问题。它通常以某个预定义停车位的裁剪图像 patch 作为输入，输出 occupied 或 vacant 这样的二分类标签。这个方法和停车场占用检测的最终目标非常贴合，因为每个停车位都会得到自己的状态预测。

Yuldashev 等人 [2] 使用 improved MobileNetV3 进行停车场占用检测，就属于这一类方法。该方法处理来自 PKLot 和 CNRPark-EXT 等数据集的停车位图像 patch，并判断每个 patch 是 occupied 还是 vacant。这一类方法的主要优势是输出非常清晰，直接是 slot-level result。由于输出已经和最终任务对齐，因此可以使用 accuracy、precision、recall、F1-score 和 AUC 等标准分类指标进行评价，但这些指标的详细定义应留给 Role 3 说明。

不过，这类方法依赖预定义停车位位置。实际应用中，每个停车位必须先被手动标注、裁剪或以其他方式提取出来，模型才能分类。如果摄像头位置变化，或者系统部署到另一个停车场，这些区域可能需要重新定义。另一个局限是，patch classification 通常是 frame-based 的。如果没有时间模块，阴影、反光、局部遮挡或经过车辆都可能导致预测结果在帧之间波动。

对我们的项目来说，slot-based classification 适合作为 baseline，也提醒我们最终输出必须是车位级占用状态。然而，如果只依赖 patch classification，对于基于视频监控的系统来说可能过于受限。我们的系统更适合将车位级判断、目标检测和时间稳定机制结合起来。

### 2. 自动车位检测与占用分类

自动车位检测方法试图减少或移除手动停车位标注。它们不是默认每个 slot polygon 已经存在，而是从视觉证据中推断停车位位置。

APSD-OC [1] 是这一类方法的重要例子。该方法使用一系列图像中的车辆检测结果，并将检测到的车辆位置变换到 bird's-eye view。随后使用 clustering 推断停车位位置。停车位被检测出来后，再对每个停车位进行 occupancy classification。换句话说，这个系统结合了 vehicle detection、geometric transformation、parking slot discovery 和 occupancy classification。

这种方法的主要优势是，它解决了 patch-based systems 的一个实际问题：手动停车位标注。如果系统能够自动推断停车位位置，那么部署到新停车场会更容易。这对真实 smart parking systems 很重要，因为当停车位数量很多时，手动标注会很耗时。

它的局限是，自动车位检测比手动定义 slot polygons 更复杂。它需要足够多的车辆观测来显现典型停车位置，也需要稳定摄像头和合适的几何处理。如果摄像头移动、某些车位缺少 occupied 样本，或者停车场布局不规则，slot discovery 可能不可靠。

对我们的项目来说，APSD-OC [1] 提供了一个重要设计启发：停车占用检测不只是识别车辆，还必须把车辆位置和停车位几何连接起来。考虑 coursework 实现范围，手动定义停车位 polygons 可能更可行。但 APSD-OC 可以作为更自动化的替代方案和未来改进方向。

### 3. 基于 YOLO 的停车位或车辆检测

基于 YOLO 的方法将停车场分析看作目标检测问题。输入通常是完整图像或视频帧，输出是车辆、停车位或可用车位等目标的 bounding boxes。这类方法很有吸引力，因为 YOLO 模型面向实时检测，并且已有大量成熟实现。

Zhao 等人 [3] 提出 CMCA-YOLO，用于停车场监控图像。该方法关注停车场中车辆可能较小、密集或重叠的情况。论文使用自定义停车场监控数据集，并报告 precision、recall、AP、mAP 和速度相关表现。这篇论文是 parking-specific 的，因此比通用 object detection benchmark 更接近我们的场景。

Sobirin 等人 [4] 提供了一个更直接的 YOLOv8 parking-space detection 示例。该工作比较 YOLOv8 和 YOLOv5，并从摄像头图像中检测停车相关目标。这类系统比较容易理解和实现，因为 detector 输出可视化的 bounding boxes，也可以用常见 object detection metrics 评价。

Wu 等人 [5] 提出了一种基于深度学习和 panoramic images 的实时停车位检测方法。该系统使用停车相关数据集，并同时报告检测表现和实时部署因素。虽然它的输入是 panoramic 或 AVM-style image，而不是固定 overhead CCTV camera，但它展示了现代停车检测系统如何同时考虑 detection accuracy 和 runtime。

尽管 YOLO-based detectors 很实用，它们并不能自动解决 slot-level occupancy。检测器可以识别车辆或可用车位候选区域，但系统仍然需要判断该检测属于哪个具体停车位。一辆驶过车道的车可能被模型检测到，但不应被算作占用某个车位。同样，部分可见车辆可能与多个 slot regions 重叠。因此，还需要额外的 detection-to-slot assignment 步骤，将 object detections 转换成最终车位可用性。

对我们的项目来说，YOLO-based detection 是很强的 implementation baseline。它可以作为每一帧检测车辆或障碍物的第一阶段。但为了得到稳定的 occupancy output，它必须和 parking slot geometry 结合。这个设计思路由 slot-based systems 间接支持，因为 cropped slot patches 来自预定义停车位区域 [2]；APSD-OC 也支持这一点，因为它说明车辆检测必须与停车位几何连接起来，才能判断 occupancy [1]。在我们的实现中，可以更务实地手动定义 parking slot polygons，再用 overlap 或 center-point rules 将检测到的车辆分配到具体车位。这里不再把 ROI mapping 当作独立文献类别，而是把它作为将 object detection 适配到 slot-level occupancy 的必要实现步骤。

### 4. 跟踪与时间稳定

多目标跟踪论文应被视为 supporting computer vision modules，而不是直接的停车场占用检测方案。它们本身不能证明停车场占用检测性能会提高。但它们提供了跨帧维护 object identities 的机制，这些机制可以被改造用于 video-based parking occupancy system 的时间稳定。

TrackTrack [7] 是一种 online multi-object tracking 方法，重点关注 track-level association。它与本项目的关系在于，停车场系统需要按视频帧连续处理，并不断更新 occupancy status。track-aware association 可以帮助减少由于短暂漏检或多个相似车辆靠近而造成的不稳定输出。

MOTIP [8] 将多目标跟踪建模为 ID prediction。它不是只依赖手工匹配规则，而是使用上下文信息预测 object identities。这和停车场场景相关，因为车辆可能低速移动、短暂停止，或被其他车辆和场景结构部分遮挡。虽然 MOTIP 不是 parking-specific，但它支持这样一个思想：temporal association 可以比简单逐帧匹配更稳健。

Tracking 或 temporal smoothing 可以减少由 missed detections、passing vehicles、short-term occlusion 和 frame-level classification errors 导致的不稳定预测。例如，如果某辆车连续多帧被检测到位于某个 slot 内，系统可以更有信心判断该车位 occupied。如果某辆车只在一帧中经过某个 slot region，时间逻辑可以避免错误地更新为 occupied。

对我们的项目来说，第一阶段未必需要完整实现高级 tracker。更简单的 temporal smoothing rule 或 lightweight tracker 可能已经足够。不过，TrackTrack [7] 和 MOTIP [8] 可以支持我们在设计中加入 temporal information，而不是把每一帧独立处理。

### 5. 弱光或复杂条件下的鲁棒性

真实停车场会面对多种变化条件，包括夜间光照、雨、雾、阴影、反光、车灯和相机噪声。因此，即使某些鲁棒性感知论文不是 parking-specific，它们仍然有参考价值。

Wang 等人 [9] 研究弱光条件下的多目标跟踪，并提出 LMOT 设置。这与停车场相关，因为停车场经常在夜间运行，而弱光会降低检测置信度和外观特征质量。这篇论文不应被描述成停车占用检测方法，但它支持我们在设计和评价系统时考虑 night-time monitoring。

Li 等人 [10] 研究 RAW 和 sRGB 数据下的复杂条件目标检测，覆盖低光、雨、雾等挑战。虽然该方法不是为停车场设计的，但它强调了检测系统不应只在白天清晰图像上测试，而应考虑不同视觉条件。

对我们的项目来说，这些论文说明鲁棒性应该影响系统设计。如果数据允许，检测器和 slot-mapping logic 应在阴影、弱光、眩光和天气变化条件下测试。如果没有这类数据，Role 3 应在 metrics 和 gaps discussion 中清楚说明这一限制。

## 对 proposed system 的设计启发

以上方法说明，我们的项目不应依赖单一算法。Slot-based classification 提供直接的 occupancy output，但依赖预定义 slot regions。Automatic slot detection 可以减少手动标注，但带来几何处理复杂度。YOLO-based detection 适合实时实现，但必须与 parking slot geometry 连接起来才能产生最终 occupancy status。Tracking methods 不是 parking-specific，但提供了时间稳定机制。Robustness studies 不直接解决停车占用问题，但说明为什么需要考虑夜间和复杂条件测试。

基于这些设计启发，proposed system 应采用模块化 pipeline：

```text
Input surveillance frame
-> vehicle or obstacle detection
-> parking slot polygon assignment
-> optional tracking or temporal smoothing
-> slot-level occupied / available output
```

这个 pipeline 符合 coursework 中“combine multiple known algorithms into a new approach”的要求。同时，它也保持了实现可行性，因为每个模块都可以用 Python 和 OpenCV-compatible tools 独立开发和测试。

## 所选方法与系统对比表

| Paper / Method | Main Task | Core Technique | Dataset / Scenario | Output | Advantages | Limitations | Relevance to Our Project |
|---|---|---|---|---|---|---|---|
| APSD-OC [1] | 自动车位检测与占用分类 | 车辆检测、透视变换、鸟瞰映射、聚类、ResNet34 分类 | PKLot 和 CNRPark-EXT 停车占用数据集 | 检测到的停车位位置和 occupied/vacant 标签 | 减少手动停车位标注；连接检测与车位几何 | 需要足够车辆观测、几何处理和稳定相机 | 强设计参考，说明如何连接检测结果和车位几何 |
| Improved MobileNetV3 [2] | 停车位占用分类 | 对 cropped parking slot patches 进行轻量 CNN 分类 | PKLot 和 CNRPark-EXT | 车位级 occupied/vacant 分类 | 直接匹配最终任务；评价简单；推理效率较好 | 需要预定义或裁剪好的 slot regions；时间推理较弱 | 可作为 slot-level classification baseline |
| CMCA-YOLO [3] | 停车场监控目标检测 | 面向停车图像的 YOLO-based detector 和 attention modules | 自定义停车场监控数据集 | 车辆/物体检测框 | parking-specific；关注小目标和重叠目标；适合实时检测讨论 | 检测输出仍需 slot assignment；自定义数据集可能限制泛化 | 适合作为监控场景下的 detector reference |
| YOLOv8 parking detection [4] | 停车位或车辆检测 | YOLOv8 目标检测，并与 YOLOv5 比较 | 自定义停车摄像头数据集 | 停车相关目标检测框 | 实用、容易实现；使用常见 YOLO 指标 | 本身不能保证稳定的 slot-level occupancy | 支持使用 YOLO 作为实现 baseline |
| Panoramic parking detection [5] | 实时停车位检测 | 基于 improved PP-Yoloe 和 panoramic images 的深度检测器 | PSEX panoramic parking dataset | parking corner 和 occupancy-related detections | 近期、同行评审、parking-specific，同时考虑准确率和 runtime | 使用 panoramic/AVM images，不是固定 overhead CCTV | 支持现代停车检测和部署讨论 |
| YOLO-World [6] | 开放词汇目标检测 | 根据文本类别进行检测的 vision-language detector | 通用检测 benchmark，不是 parking-specific | prompted categories 的检测框 | 类别灵活，可检测车辆类型或障碍物；面向实时检测 | 未验证为停车占用系统；可能需要 prompt 和 domain adaptation | 可作为灵活车辆/障碍物识别的 supporting detector module |
| TrackTrack [7] | 在线多目标跟踪 | track-perspective association 和 track-aware initialization | 标准 MOT benchmarks | 跨帧 object tracks | 支持连续视频处理和身份一致性 | 不是 parking-specific；不输出 slot occupancy | temporal stabilization 的 supporting reference |
| MOTIP [8] | 将多目标跟踪建模为 ID prediction | 基于上下文的 object association 身份预测 | 标准 MOT benchmarks | object identities 和 trajectories | 减少对手工关联规则的依赖 | 需要可靠检测，且不是 parking-specific | learned temporal association 的 supporting reference |
| Multi-Object Tracking in the Dark [9] | 弱光多目标跟踪 | low-light tracking framework 和 LMOT setting | 弱光 MOT 场景 | 弱光下的 object tracks | 关注夜间跟踪，符合停车场真实需求 | 不解决 parking slot occupancy | 支持夜间部署鲁棒性讨论 |
| AODRaw [10] | 复杂条件下目标检测 | RAW/sRGB 数据下的低光、雨、雾、多场景检测 | AODRaw adverse-condition detection dataset | 目标检测框 | 强调天气和光照变化下的检测鲁棒性 | 不是 parking-specific，完整方法可能依赖 RAW 数据 | 支持复杂条件测试和鲁棒性设计 |

## 过渡到 Role 3

上述方法在假设、输入、输出、实现复杂度和评价重点上存在差异。Slot-based classifiers 可以直接预测 occupancy，但需要预定义 slot regions；YOLO-based detectors 适合实时使用，但需要 detection-to-slot assignment；tracking methods 可能提升时间稳定性，但主要是 supporting modules；robustness studies 强调部署挑战，而不是提供完整停车场解决方案。因此，下一节应更详细说明标准 performance metrics，并总结 proposed system 仍然存在的主要 limitations 和 research gaps。
