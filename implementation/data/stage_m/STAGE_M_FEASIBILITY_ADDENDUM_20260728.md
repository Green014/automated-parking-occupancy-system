# Stage M feasibility addendum

Audit date: 2026-07-28
Scope: new Stage M work only. This file does not amend or replace a Stage L
artifact.

## Runtime and implementation decision

The active experiment environment contains:

- Python 3.12.13;
- `ultralytics==8.4.104`;
- `torch==2.13.0+cu130`;
- OpenCV 4.13.0.92;
- NumPy 2.4.4;
- PyYAML 6.0.3;
- LAP 0.5.13;
- Shapely 2.1.2 (BSD-3-Clause).

Ultralytics 8.4.104 contains official
`ultralytics.solutions.ParkingManagement`, `tracktrack.yaml`,
`ultralytics.trackers.track.TRACKER_MAP["tracktrack"]`, and
`ultralytics.trackers.track_tracker.TRACKTRACK`. The TrackTrack predictor
registration path also attaches the raw-prediction hook used to recover
detections removed by tight NMS. Stage M therefore calls
`model.track(..., persist=True, tracker=...)` and does not instantiate
`TRACKTRACK` directly.

The paper authors' repository is
[`kamkyu94/TrackTrack`](https://github.com/kamkyu94/TrackTrack), licensed MIT.
The executable implementation in this project is the separately distributed
Ultralytics implementation, licensed AGPL-3.0. The method provenance label is:

> paper method executed through Ultralytics implementation

The project does not claim that author code was copied into Ultralytics or
into this repository.

## OS0 decision

`OS0-Controlled` is feasible with the frozen D1 checkpoint, D1 inference
settings, scene polygons, and local TrackTrack configuration. The adapter
constructs the official `ParkingManagement` object without changing
site-packages. Official centre-point-in-polygon decisions remain authoritative.
The local adapter only replays that same rule to attach a slot ID and write
the common evaluation artifacts.

For consecutive frames from one video, the official object's tracker state is
retained. For unrelated images or a new source, a fresh official object is
constructed. Static output must be described as an **OS0 static centre-point
diagnostic**, never as temporal evidence.

## T0-T3 decision

The new sequence adapter keeps the existing detector path intact and supplies
three inference streams:

- one D1 `predict` stream shared by T0 and T1;
- one D1 `track` stream with frozen ByteTrack for T2;
- one D1 `track` stream with frozen TrackTrack for T3.

Every stream shares D1 weights, confidence, IoU, image size, classes, B1
mapping coverage, E1b checkpoint/threshold, and E4 parameters. A source change
constructs a fresh YOLO model, preventing tracker state and TrackTrack
raw-prediction hooks from leaking between videos.

## Data feasibility

No formal new continuous or low-light slot-occupancy test is currently
eligible:

- VIRAT 0502 is already consumed development evidence and is not a new
  independent validation scene.
- Grand Bassin has incomplete project truth.
- NDISPark is licensed and locally present, but it supports detector boxes or
  count metrics, not parking-slot state or tracking.
- AODRaw's repository explicitly assigns CC BY-NC-SA 4.0 to **code**; it does
  not explicitly assign that license to the images and annotations. The
  dataset was not downloaded.
- LMOT's current author repository states CC BY-NC 4.0 for the dataset and MIT
  for code, but only train and validation are released. No local copy is
  present, and LMOT has no parking-slot polygons or occupied/vacant truth.

Consequently, Stage M may execute controlled smoke tests now. Formal video,
low-light occupancy, AODRaw, and LMOT measurements remain gated by the
machine-readable audit in `STAGE_M_DATA_GATES_20260728.yaml`.

## Smoke dependency addendum

The first OS0 smoke exposed that Shapely was not installed:
Ultralytics' requirement check installed Shapely 2.1.2 during that run. The
`stage_m_smoke_20260728_v1` output is retained as dependency-discovery
evidence and is not used for a timing claim. Shapely's version, BSD-3-Clause
licence file (1,612 bytes) and licence SHA-256
`e8cd7d34604d2758ff589f0f34d6ed79c3f7fb14aaf5e12fb09221421391a3bf`
were then added to the Stage M runtime freeze.

The clean, registered `stage_m_smoke_20260728_v2` run completed all three
paths: four continuous OS0 frames, two OS0 static images with two fresh
official instances, and four T0--T3 frames. It has no truth and therefore
produces no accuracy or temporal metrics.
