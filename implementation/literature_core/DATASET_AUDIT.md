# Continuous Parking-Video Dataset Audit

Audit date: 26 July 2026 (Asia/Shanghai)

## Outcome

No source currently provides the required fixed marked bays, both occupancy
classes, and a verified arrival/departure in two leakage-safe physical scenes.
Accordingly, no temporal development or holdout pair has been falsely
declared.

The user accepted the VIRAT Usage Agreement, and a bounded official subset was
screened. Twenty-one clips totaling 961,643,821 bytes were downloaded without
overwriting any prior file. One clip from physical scene `0502` contains a
clear departure from a complete marked slot and remains eligible pending exact
frame truth. The other 20 clips were rejected. No second physical scene passed,
so the repository will not start E4, E5, or Fusion V2.

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

The candidate audit is in
`data/manifests/temporal_dataset_audit_20260726.yaml`; exact downloaded-file
metadata and every negative decision are in
`data/manifests/virat_screening_20260726.yaml`.

## VIRAT bounded screening result

The official Release 2.0 introduction defines `XXYY` (the first four digits
after `VIRAT_S_`) as the scene and `ZZ` as the sequence. This matters:
`000201`, `000203`, `000205`, and `000207` are all sequences from physical
scene `0002`, not independent scenes. The acquisition helper and regression
tests now enforce this grouping.

The screening covered 21 clips from physical scenes `0002`, `0100`, `0101`,
`0102`, `0400`, `0401`, `0500`, and `0502`. Most either lacked complete marked
parking bays or showed only vehicles moving through lanes. Vehicle-interaction
event counts were useful for prioritization but were not treated as occupancy
truth because VIRAT events describe human actions and its static-object
annotations are optional.

Only `VIRAT_S_050202_10_002159_002233.mp4` passed visual eligibility: a dark
vehicle begins in a complete diagonal marked bay, backs out at approximately
45-55 seconds, and leaves the bay vacant. This is not yet an experiment
partition. Exact slot polygon coordinates, the per-frame state boundary, and
an independent second scene remain required.

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

The bounded-screening workflow is:

1. Record acceptance date locally; do not commit the agreement or personal
   details. **Completed.**
2. Download official annotations plus a sub-1-GB screening set of short clips
   from distinct official scene IDs rather than the whole 37.63-GB folder.
   **Completed: 961,643,821 bytes.**
3. Record URL/item ID, byte size, SHA-256, resolution, FPS, duration, and
   scene/camera key for every file.
4. Screen full clips for complete marked bays, both classes, at least one
   true arrival/departure, and no camera movement that invalidates fixed ROIs.
   **Completed for the current subset: one clip eligible, 20 rejected.**
5. Prefer two distinct scenes/cameras. Only if that fails may one long source
   be split with an explicit temporal guard and no adjacent-frame leakage.
6. Lock the holdout identity and checksum before any development label is used
   for dwell, threshold, or hysteresis selection.
7. Change `configs/temporal_protocol_pending.yaml` to `status: frozen` only
   when its validator reports `ready_for_experiment: true`.

The protocol validator explicitly rejects slot/frame grouping, same-scene use
without permission, overlapping/adjacent clips that violate the guard, missing
frozen hashes, and unrecorded required agreement acceptance.
