# Coursework Alignment Assessment

The coursework brief requires the literature review to focus on:

1. Existing computer vision systems or algorithms relevant to the selected topic.
2. Performance metrics commonly used in the field.
3. Design considerations that help the later implementation.
4. Quantitative evaluation methods for the final system.

## Assessment of the earlier review

The previous review was strong for modern computer vision components:

- open-vocabulary detection
- multi-object tracking
- low-light tracking
- adverse-condition object detection
- cross-scene generalization

However, it was not yet fully aligned with the coursework requirement because it did not sufficiently compare existing parking occupancy systems. The previous core papers explain how to build a strong system, but they do not fully answer:

- How have existing parking occupancy systems been designed?
- What datasets are normally used for parking occupancy?
- What metrics do parking occupancy papers report?
- How should our system be accepted or evaluated?

## Updated review strategy

The literature review should be restructured into four layers:

1. **Parking-specific existing systems**  
   Examples: APSD-OC, Improved MobileNetV3, CMCA-YOLO, YOLOv8 parking detection, ROI+YOLO smart parking.

2. **Modern implementation modules**  
   Examples: YOLO-World for flexible detection, TrackTrack/MOTIP/DiffMOT for tracking, AODRaw/LMOT for robustness.

3. **Datasets and evaluation metrics**  
   Examples: PKLot, CNRPark-EXT, custom parking datasets, accuracy, precision, recall, F1, AUC, mAP, FPS, latency.

4. **Gap and proposed pipeline**  
   Existing parking-specific papers often solve either slot classification or object detection. The proposed project can combine vehicle detection, tracking, slot mapping, and temporal smoothing for video-based occupancy tracking.

## Key recommendation

The final coursework literature review should not present YOLO-World or MOTIP as direct parking occupancy solutions. Instead, it should present them as modern modules that can improve a parking-specific pipeline after comparing direct parking occupancy approaches.

