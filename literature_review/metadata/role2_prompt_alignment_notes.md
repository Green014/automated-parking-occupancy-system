# Role 2 Prompt Alignment Notes

## Does the prompt match the coursework?

Yes. The prompt matches the coursework requirements because the coursework asks the literature review to cover:

- existing computer vision systems or algorithms relevant to the topic
- commonly used performance metrics
- comparison of existing algorithms and systems
- gaps, limitations, and design considerations
- preparation for later implementation and evaluation

The prompt also correctly limits Role 2 so it does not duplicate Role 1 or Role 3.

## Does the prompt match our current research?

Mostly yes. Our current literature set contains:

- parking-specific systems: APSD-OC, Improved MobileNetV3, CMCA-YOLO, YOLOv8 parking detection, panoramic parking detection
- supporting CV modules: YOLO-World, TrackTrack, MOTIP, Multi-Object Tracking in the Dark, AODRaw

This supports a method-based Role 2 structure:

- slot-based classification
- automatic slot detection
- YOLO-based detection
- ROI/slot mapping
- temporal tracking
- low-light and adverse-condition robustness

## Adjustments made

The draft uses cautious wording for supporting CVPR papers. It does not claim that YOLO-World, TrackTrack, MOTIP, LMOT, or AODRaw directly solve parking occupancy. Instead, it explains that they provide modules or design ideas that may be adapted into a parking occupancy pipeline.

The draft avoids:

- full introduction
- full conclusion
- formulas for metrics
- full final research gaps section
- overclaiming results not shown in the papers

## Recommended placement in final report

This section can be placed after Role 1's introduction/background and before Role 3's performance metrics and final gap analysis.

