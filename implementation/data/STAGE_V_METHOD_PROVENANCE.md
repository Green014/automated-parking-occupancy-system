# Stage V Method Provenance

Date: 2026-07-30

## Provenance matrix

| Stage V item | Origin | Code treatment | Frozen parameter treatment |
|---|---|---|---|
| `C0` Classic | General OpenCV per-ROI foreground-statistics pattern; design context from the member reference | Clean-room implementation in `parking_occupancy.stage_v`; no reference code copied | New `0.30` foreground-ratio default is explicitly uncalibrated |
| D1 detector | Existing project-produced fine-tuned YOLOv8n | Existing `create_detector` / `UltralyticsSequenceAdapter` reused | SHA-256 and size must match frozen D1; inference settings unchanged |
| B1 mapping | Existing project-original one-to-one polygon coverage mapper | Existing `map_detections_to_slots` reused | Coverage `0.40` unchanged |
| E1b classifier | Existing project-produced MobileNetV3-CBAM slot classifier | Existing `E1bClassifierAdapter` reused | SHA-256, patch size, perspective warp, and threshold `0.76` unchanged |
| F2 fusion | Existing project-original asymmetric uncertainty gate | Existing `uncertainty_gated_fusion` reused | Detector positives remain final; only detector-negative slots are reviewed |
| E4 | Existing project optional temporal filter | Available only by explicit `--temporal` for a single continuous fusion video | Default off; frozen values unchanged |
| ByteTrack/TrackTrack | Existing project optional tracker adapters | Available through the existing detector adapter | Default `none`; controlled C0/C1/C2 comparison rejects tracking |
| Unified result/output | Stage V project-original integration | New backend-neutral result, runner, schemas, visualization invariant, and cache | No Stage J–U.1 artifact or result changed |

## Why F2 is not simple concatenation

F2 has asymmetric authority:

1. D1 detections are mapped by B1.
2. A detector-positive slot is emitted occupied immediately; E1b is not called
   for that slot and cannot overturn the result.
3. E1b is called for every and only detector-negative slot.
4. E1b recovers the slot only when its frozen occupied probability reaches
   `0.76`; otherwise the slot remains vacant.

This is a detector-first gate, not score averaging, voting, or two independent
pipelines joined after the fact.

## Model artifact identity

| Artifact | Bytes | Required SHA-256 | Stage V status |
|---|---:|---|---|
| D1 | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` | Verified before real smoke |
| E1b | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` | Verified before real smoke |

Stage V raises an explicit error for a missing, wrong-size, or wrong-hash
artifact. It never downloads or substitutes a model.

## Licence and authorization record

- Reference repository at the audited commit: no licence file; direct reuse is
  prohibited.
- Stage V direct reference-code reuse: none.
- Stage V reference-weight reuse: none.
- The current project: no top-level project licence found during the Stage V
  public-release audit.
- Ultralytics: upstream AGPL-3.0 or enterprise licensing boundary must be
  resolved consistently with the intended public/commercial use.
- OpenCV/NumPy/PyTorch and all other dependencies retain their own licences and
  notices.

No unrecorded permission is claimed.
