# Stage V Integration Audit

Date: 2026-07-30  
Stage: **V — Multi-Backend Occupancy Integration**  
Reference: `https://github.com/prestzy/OpenCV-Car-Parking`  
Audited reference commit: `12271576be39a4ac0eb456526eca122685799e8c`

## Decision

The reference repository is usable as design context but not as an inbound
source-code or model dependency. It contains no `LICENSE`, `LICENCE`,
`COPYING`, or `NOTICE` file at the audited commit. Stage V therefore copies
none of its source, templates, static files, configuration, or committed
MobileNetV3 checkpoint. The new Classic path is a clean-room implementation
described as:

> OpenCV pixel-count baseline inspired by the reference project

The member must add a licence or provide written permission before any direct
reuse is reconsidered. No licence or authorization is inferred from public
GitHub visibility.

## What the audited repository actually implements

The current repository does **not** implement the exact fixed-rectangle,
adaptive-threshold, non-zero-pixel-count pipeline assumed in the Stage V
request. Its README explicitly says occupancy is based on tracked vehicles
rather than thresholded-pixel counts.

| Audit item | Audited implementation |
|---|---|
| Inputs | Local MP4, webcam index, or RTSP URL through OpenCV `VideoCapture` |
| Slot definition | Manually drawn polygons saved in `data/parking_spaces.json`; coordinates scale with processing resolution |
| Default occupancy source | Camera-specific committed MobileNetV3 slot classifier |
| Other occupancy source | YOLOv8n vehicle detections with ByteTrack and centre/bottom-centre polygon association |
| Overhead fallback | Per-polygon grayscale intensity standard deviation plus Canny edge ratio |
| Temporal logic | Confirmation frames, dwell time, movement limit, vacancy grace, degraded-weather multiplier, and `UNKNOWN` state |
| Outputs | Flask dashboard/MJPEG, JSON status endpoints, event CSV, and parking-session CSV |
| Video support | Yes; local video, webcam, and RTSP |
| Main dependencies | Flask, NumPy, OpenCV, Ultralytics, PyTorch, torchvision; pytest is also in runtime requirements |
| Repository licence | **Missing** |

The committed `config.json` supplies fixed engineering thresholds, including
YOLO confidence `0.25`, classifier occupied probability `0.75`, appearance
standard deviation `30.0`, Canny edge ratio `0.07`, dwell `1.5 s`, and several
frame-confirmation values. The repository does not document a held-out
calibration procedure for all of those runtime defaults.

## Requested preprocessing checklist

| Technique | Present? | Precise finding |
|---|---:|---|
| Grayscale conversion | Yes | Used by frame-quality assessment and overhead appearance fallback |
| Blur | Yes, but conditional | Gaussian `3x3` denoising is used only for dark/noisy conditions |
| Adaptive threshold | **No** | No adaptive-threshold occupancy path at the audited commit |
| Median blur | **No** | Not used |
| Dilation | **No** | Not used |
| ROI non-zero pixel count | **No** | The fallback computes edge ratio and intensity standard deviation, not a count-threshold rule |
| Fixed rectangular slots | **No** | Slots are polygons with at least three points |
| Fixed pixel-count threshold | **No** | Fixed probability, edge-ratio, texture, dwell, and confirmation thresholds exist instead |

## Code and model provenance boundary

| Category | Stage V treatment |
|---|---|
| Directly reused reference code | None |
| Directly reused reference model/assets | None |
| Idea-level reference | Manual fixed-camera ROIs and an interpretable classical per-slot baseline |
| Independently implemented Stage V code | Grayscale, Gaussian blur, adaptive Gaussian inverse threshold, median blur, dilation, polygon mask, foreground ratio, and explicit uncalibrated ratio threshold |
| Reused project-original code | `Detection`, `ParkingSlot`, slot maps, D1 adapter, B1 one-to-one overlap mapping, E1b adapter, F2 asymmetric gate, output/evaluation conventions, and optional existing E4/tracker integrations |

The clean-room Classic defaults are recorded in every
`configuration_snapshot.yaml`. The foreground-ratio threshold `0.30` is an
**uncalibrated reference default**, not a best or validated parameter.

## Third-party and public-release boundary

OpenCV and NumPy have permissive upstream licences, while Ultralytics publishes
AGPL-3.0 and enterprise licensing options. This repository currently has no
top-level project licence. Consequently:

1. Stage V may remain in local research use.
2. Reference-repository source or weights must not be copied.
3. A public release must select and add a project licence compatible with the
   actual Ultralytics use, or document the applicable enterprise permission.
4. Third-party notices and the chosen model-distribution boundary must be
   reviewed before a public release.

This is an engineering audit, not legal advice.
