# Condensed 10-Reference Selection Rationale

This file explains why the condensed literature review uses 10 references.

## Selected references

| No. | Paper | Reason for inclusion |
|---|---|---|
| 1 | APSD-OC | Best direct reference for connecting vehicle detection, parking-slot geometry, and occupancy classification. |
| 2 | Improved MobileNetV3 | Strong direct reference for slot-patch occupancy classification and PKLot/CNRPark-EXT metrics. |
| 3 | CMCA-YOLO | Direct parking surveillance detection paper; useful for YOLO-based real-time design. |
| 4 | Car Parking Space Detection Using YOLOv8 | Simple and practical YOLO parking detection baseline. |
| 5 | Real-Time Parking Space Detection Based on Deep Learning and Panoramic Images | Recent peer-reviewed parking detection paper; replaces the older PKLot citation in the core 10-reference version. |
| 6 | YOLO-World | High-quality CVPR 2024 detector; supports flexible object/obstacle detection. |
| 7 | TrackTrack | High-quality CVPR 2025 online tracking method. |
| 8 | MOTIP | High-quality CVPR 2025 tracking-by-ID-prediction method. |
| 9 | Multi-Object Tracking in the Dark | High-quality CVPR 2024 low-light tracking reference and LMOT dataset. |
| 10 | AODRaw | High-quality CVPR 2025 adverse-condition detection reference and dataset. |

## Excluded from condensed version

| Paper | Reason for exclusion from the 10-reference version |
|---|---|
| Pixel-Wise ROI + YOLO Smart Parking | Very close to the proposed implementation idea, but it is an arXiv preprint, so it was removed from the core 10-reference version and kept only as an auxiliary candidate. |
| PKLot | Foundational and peer-reviewed, but 2015 is too old for the condensed core version given the coursework preference for recent references. It can still be mentioned as background for datasets if needed. |
| OW-OVD | Useful for unknown-object detection, but less necessary than YOLO-World for the basic pipeline. |
| DiffMOT | Useful for nonlinear motion, but TrackTrack and MOTIP already cover tracking in the condensed review. |
| Generalizable MOT | Useful for cross-scene generalization, but less central to initial coursework implementation. |
| Optimizing YOLOv8 for Parking Space Detection | Useful but arXiv and partly overlaps with YOLOv8 parking detection. |
| DeCLIP, SearchDet, OmniTrack, MITracker, Event Camera Detection, Can't Slow Me Down | Good extension directions but not essential for a concise coursework literature review. |

## Balance

The 10-reference version keeps:

- 5 parking-specific or parking-dataset references for existing systems, datasets, and metrics.
- 5 high-quality CVPR references for implementation modules and robustness.

## Change from the previous condensed version

The first condensed review used the arXiv paper "Smart Parking with Pixel-Wise ROI Selection for Vehicle Detection Using YOLOv8, YOLOv9, YOLOv10, and YOLOv11" as reference [5]. It was first replaced with the peer-reviewed PKLot dataset paper to remove preprints from the core set. However, PKLot was considered too old for a concise recent-review version.

The final replacement is:

W. Wu, H. Chen, J. Gong, K. Che, W. Ren, and B. Zhang, "Real-Time Parking Space Detection Based on Deep Learning and Panoramic Images," Sensors, 2025.

This keeps the core set recent, peer-reviewed, and parking-specific.
