# Evaluation Metrics and Acceptance Plan

This file defines the evaluation logic for the Automated Parking Lot Occupancy and Tracking System.

## 1. Slot-level occupancy metrics

These are the most important metrics because the final user-facing output is whether each parking space is occupied or available.

| Metric | Meaning | Why it matters |
|---|---|---|
| Accuracy | Percentage of correctly classified slot states | Simple overall correctness measure |
| Precision | Among predicted occupied slots, how many are truly occupied | Avoids falsely marking available spaces as occupied |
| Recall | Among truly occupied slots, how many are detected as occupied | Avoids missing occupied spaces |
| F1-score | Harmonic mean of precision and recall | Useful when occupied/free classes are imbalanced |
| False occupied rate | Available spaces incorrectly marked occupied | Bad for drivers because it hides usable spaces |
| False free rate | Occupied spaces incorrectly marked available | Worse failure because drivers may be sent to unavailable spaces |
| Occupancy transition latency | Time delay between real state change and system update | Important for real-time parking guidance |

Recommended primary metric: **slot-level F1-score**, supported by false free rate and transition latency.

## 2. Detection metrics

These evaluate the vehicle/object detector.

| Metric | Meaning | Use |
|---|---|---|
| IoU | Overlap between predicted and ground-truth boxes | Box quality |
| Precision | Correct positive detections / all detections | False detection control |
| Recall | Detected objects / all ground-truth objects | Missed vehicle control |
| AP | Area under precision-recall curve for one class | Standard object detection metric |
| mAP@0.5 | Mean AP at IoU threshold 0.5 | Common YOLO-style reporting |
| mAP@0.5:0.95 | Mean AP across IoU thresholds | Stricter detection quality |

Detection metrics are important, but they are secondary to slot-level occupancy metrics.

## 3. Tracking metrics

These evaluate temporal identity and track stability.

| Metric | Meaning | Use |
|---|---|---|
| IDF1 | Identity preservation quality | Important when tracking vehicles over time |
| MOTA | Combines false positives, false negatives, and ID switches | Traditional MOT metric |
| HOTA | Balances detection and association quality | More balanced tracking metric |
| ID switches | Number of times a tracked identity changes | Indicates unstable tracking |
| Track fragmentation | Number of broken trajectories | Indicates temporal instability |

Tracking metrics are useful if the project annotates track IDs. If track IDs are not annotated, use simpler temporal stability metrics such as occupancy flicker count.

## 4. System performance metrics

These evaluate deployability.

| Metric | Meaning | Target |
|---|---|---|
| FPS | Processed frames per second | Aim for real-time or near-real-time |
| End-to-end latency | Time from frame input to occupancy output | Lower is better |
| Per-frame processing time | Breakdown of detection, tracking, and slot assignment time | Helps identify bottlenecks |
| Memory usage | GPU/CPU memory consumption | Important for edge deployment |
| Multi-camera throughput | Number of cameras supported simultaneously | Optional extension |

## 5. Suggested acceptance test

Create a small labeled parking-lot test set:

- 3-5 short video clips or image sequences
- at least 20-50 parking spaces if possible
- labels for occupied/available spaces
- separate clips for daytime, night, shadow, and rain/cloudy conditions if available

Suggested minimum report:

- slot-level accuracy, precision, recall, F1-score
- false free rate and false occupied rate
- FPS and average latency
- qualitative examples of success/failure

If the system includes tracking:

- occupancy flicker count before and after temporal smoothing
- optional ID switch count if track IDs are annotated

