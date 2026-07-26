# Continuous Parking-Video Dataset Audit

Audit date: 26 July 2026 (Asia/Shanghai)

## Outcome

No source is currently both legally accessible and verified to contain the
required fixed marked bays, both occupancy classes, and an arrival/departure
in two leakage-safe sequences. Accordingly, no temporal development or
holdout sequence has been falsely declared.

VIRAT Ground Release 2.0 is the conditional primary source. It is the closest
match because it contains about 8.5 hours of stationary ground video from 11
outdoor scenes. It still requires two gates: the user must personally accept
the VIRAT Usage Agreement, and the downloaded clips must pass visual screening
for complete marked bays and real occupancy transitions. The repository will
not start E4, E5, or Fusion V2 before those gates pass.

## Audit table

| Dataset | Camera Type | Continuous Video | Slot Occupancy Labels | Transitions | Low-light/Adverse | License | Size | Access | Decision |
|---|---|---|---|---|---|---|---|---|---|
| VIRAT Ground 2.0 | Stationary ground surveillance | Yes | No; manual truth required | Activity/track annotations make transitions plausible, not guaranteed | Uncontrolled outdoor scenes; not a dedicated night set | Individual VIRAT Usage Agreement; research/commercial use, restricted redistribution and PII duties | About 8.5 h, 11 scenes; official original-video folder 37.63 GB | Catalog public; every user must accept terms before data use | **Conditional primary**; screen only after user acceptance |
| Dragon Lake Parking | Overhead drone | Yes | No slot-state truth | Abundant parking maneuvers and dense trajectories | Not claimed | Non-commercial research/teaching/personal use; cite; no redistribution | 3.5 h, 4K/25 FPS, 30 scenes; 168 GB raw package, 8.02 GB JSON | Raw video requires request; sample link currently stops at OneDrive permission validation | Defer; motion/stabilization and access unresolved |
| EPFL Multi-view Multi-class | Six fixed calibrated cameras | Described, but full sequence not currently exposed | No | Vehicles and parking slots described | Occlusions; no low-light claim | Copyrighted; explicit research use with citation | 23:57 at 25 FPS; 242 non-consecutive annotated frames | Current official page exposes only the two ground-truth archives | Reject for temporal evaluation until full video access exists |
| ISLab-PVD | Urban surveillance | 16 sequences | No | Illegal-parking events, not bay occupancy transitions | Crowds, illumination changes, night IR | Citation request only; no explicit dataset license located | Not stated | Download link published, but terms/size unverified | Reject for formal evaluation |
| LMOT | Paired city outdoor cameras | Described in paper | No | MOT tracks only | Dedicated real/paired low light | No dataset license published in official repository | 32 videos, 35,120 frames | Official repository still says release is forthcoming | Low-light tracking reference only; not occupancy data |
| Grand Bassin | Fixed-looking high aerial surveillance | Yes at 2 FPS | Seven local occupied-only bays | No valid arrival/departure after review | Glare/occlusion | CC BY-NC-SA 4.0 | 1,349 reviewed frames across three sequences | Local and hash-recorded | Preserve negative/positive-only result; reject for full temporal metrics |
| CNR-EXT | Fixed surveillance | No | Yes | Sparse snapshots only | Weather/light variation | ODbL 1.0 | 4,081 images, 144,965 slot labels | Local; once-only evaluation already consumed | Frozen static external result only; never retune |
| PKLot selected 27 | Fixed surveillance | No | Yes | Roughly five-minute sampling | Weather, no night | CC BY 4.0 | 27 local development images | Local | Method development only; never temporal truth |

The machine-readable form is
`data/manifests/temporal_dataset_audit_20260726.yaml`.

## Why the seemingly easiest alternatives were rejected

- A video being public is not enough. ISLab-PVD and UFPArk-like sources cannot
  be formal evaluation data until reuse terms are explicit.
- DLP has excellent parking maneuvers, but the camera is a drone. Fixed slot
  ROIs are invalid unless residual motion is measured and corrected. The raw
  media also requires an author-approved request.
- EPFL is scientifically promising, but the current page only links the 242
  non-consecutive ground-truth frames/annotations. Those frames cannot support
  dwell time, flicker, latency, or tracking.
- LMOT can support a future low-light detector/tracker diagnostic, but it has
  no parking-slot occupancy truth and is not currently released with a clear
  data license.
- Grand Bassin was already searched thoroughly. Reinterpreting detector
  dropouts or lanes as transitions would be invalid.

## Conditional VIRAT selection protocol

After the user accepts the agreement, acquisition remains deliberately small:

1. Record acceptance date locally; do not commit the agreement or personal
   details.
2. Download official annotations plus a sub-1-GB screening set of short clips
   from distinct official scene IDs rather than the whole 37.63-GB folder.
3. Record URL/item ID, byte size, SHA-256, resolution, FPS, duration, and
   scene/camera key for every file.
4. Screen full clips for 5-20 complete marked bays, both classes, at least one
   true arrival/departure, and no camera movement that invalidates fixed ROIs.
5. Prefer two distinct scenes/cameras. Only if that fails may one long source
   be split with an explicit temporal guard and no adjacent-frame leakage.
6. Lock the holdout identity and checksum before any development label is used
   for dwell, threshold, or hysteresis selection.
7. Change `configs/temporal_protocol_pending.yaml` to `status: frozen` only
   when its validator reports `ready_for_experiment: true`.

The protocol validator explicitly rejects slot/frame grouping, same-scene use
without permission, overlapping/adjacent clips that violate the guard, missing
frozen hashes, and unrecorded required agreement acceptance.
