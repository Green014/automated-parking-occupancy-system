# Stage M Open-source Tracking Robustness Report

Date: 28 July 2026

Protocol: `STAGE-M-OPEN-SOURCE-TRACKING-ROBUSTNESS-20260728-01`

Status: implementation and reproducible smoke complete; formal continuous,
low-light slot-occupancy, AODRaw and LMOT runs remain gated.

## 1. Scope and evidence boundary

Stage M adds an official open-source reference path and a TrackTrack ablation
without rerunning, editing or selecting parameters from Stage L. It does not
turn the consumed VIRAT 0502 case into a new validation scene, and it does not
use the viewed Stage K result to choose thresholds.

The three registered Stage L records were checked before and after Stage M:

| Preserved record | Bytes | SHA-256 |
|---|---:|---|
| `configs/stage_l_integrated_workflow_frozen_20260728.yaml` | 6,268 | `29e70e76fe7a8c92fe4022520a47c2cbe69f159749960e5e170116df1ea039e3` |
| `data/STAGE_L_INTEGRATED_WORKFLOW_REPORT.md` | 7,340 | `a5b0034accdd5fa0e69ab3c1b31b19c781c62ce03af6149626107c358353a7fa` |
| `data/comparisons/stage_l_integrated_workflow_20260728.yaml` | 5,495 | `b6536131011338f2576ba6f1d8cf415748cee513943d4c82795e1c6832415408` |

No Stage L prediction, training run or artifact was executed or rewritten.

## 2. Source, implementation and licence audit

The executed environment is Python 3.12.13, Ultralytics 8.4.104,
Torch 2.13.0+cu130, OpenCV 4.13.0.92, NumPy 2.4.4, PyYAML 6.0.3,
lap 0.5.13 and Shapely 2.1.2 on the existing RTX 3060.

| Frozen item | SHA-256 |
|---|---|
| Ultralytics `solutions/parking_management.py` | `e910c5a08156ee590bec3aefa3cd3d58ca748cb997d6756679d9ab7119cc715b` |
| Ultralytics `solutions/solutions.py` | `54dd5c39e25404f075dd6a93650bc7e9950ec6d69da637f6377f00f75c410cf7` |
| Ultralytics `trackers/track.py` | `493cdecc24f71feb17091fd9935363712412a7b3a4f4a76693f783f4c8e07419` |
| Ultralytics `trackers/track_tracker.py` | `c9ccd5ed3275093a75b874813acde95fc2cd4af251df17cee00c98ff9415977a` |
| Installed upstream `tracktrack.yaml` | `33fcd0cf42023fabc1cff93e8769f949e3f92a39219a8b7de157fc612a1de37a` |
| Local frozen TrackTrack configuration | `54a158728a3dd41b523c7d9054fa0e187548075f563adb71c5db72e797328f37` |
| Local frozen ByteTrack configuration | `9c46e5796711065c1e3e2784f9ed2c2775833801c01b1a548c0ee0b092331f1f` |
| Installed Ultralytics licence | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |
| Installed Shapely licence | `e8cd7d34604d2758ff589f0f34d6ed79c3f7fb14aaf5e12fb09221421391a3bf` |

Ultralytics is recorded as AGPL-3.0 and Shapely as BSD-3-Clause. TrackTrack's
authors publish an MIT-licensed reference repository, but this project did
not copy or directly instantiate that repository. The precise relationship
is **paper method executed through Ultralytics implementation**:
`ultralytics.trackers.track_tracker.TRACKTRACK` is reached through
`model.track(..., persist=True, tracker=...)`. This retains the Ultralytics
predictor callbacks, including the raw-prediction path that a partial direct
tracker wrapper could omit.

Sources:

- TrackTrack paper:
  <https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html>
- author repository: <https://github.com/kamkyu94/TrackTrack>
- Ultralytics tracking documentation:
  <https://docs.ultralytics.com/modes/track/>
- Ultralytics source and licence:
  <https://github.com/ultralytics/ultralytics>

## 3. Frozen controlled methods

All methods share D1 checkpoint
`0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`,
confidence 0.30, NMS IoU 0.70, image size 640, source class 0, a unified
`vehicle` class, class-agnostic NMS and `max_det=300`. T0--T3 retain the
frozen B1 0.40 one-to-one polygon-coverage mapping, E1b threshold 0.76 and
Stage L E4 parameters.

| Method | Frozen definition |
|---|---|
| OS0-Controlled | Official `ultralytics.solutions.ParkingManagement`, D1, shared polygons/settings, explicit TrackTrack and official centre-point-in-polygon occupancy |
| T0 | P3 asymmetric E1b gate; no temporal filter and no tracker |
| T1 | T0 plus E4 asymmetric EMA/hysteresis |
| T2 | T1 plus the unchanged Stage L ByteTrack parameters |
| T3 | T1 plus TrackTrack through the full Ultralytics tracking entry point |

OS0's count and annotation come from the official object. The local adapter
replays the same centre-point rule only to attach a state to each named slot,
checks that its filled-slot total equals the official total, and writes the
common evaluation contract. It never patches `site-packages`.

One official object/model session persists only for consecutive frames from
the same continuous source. A source switch creates a new instance. Static
diagnostics create a new `ParkingManagement` object for every image; their
output is labelled `OS0 static centre-point diagnostic` and has no temporal
claim.

## 4. Data gates and task-specific conclusions

No newly qualified two-scene continuous parking dataset was present locally.
The formal gate requires verified licence, fixed cameras, human-reviewed
polygons and interval truth, development and physically distinct test scenes,
distinct hash-bound video/polygon/truth bundles, at least one relevant event,
configuration freeze before prediction and zero prior test executions.

| Gate | State | Allowed conclusion |
|---|---|---|
| New continuous parking test | Blocked | Audit only; VIRAT 0502 is consumed development and Grand Bassin truth is incomplete |
| End-to-end low-light slot occupancy | Blocked | Audit only; no local low-light parking data has both slot polygons and occupancy truth |
| NDISPark low-light evidence | Eligible | Detector/count-only supporting evidence; it has no slot geometry or occupied/vacant truth |
| AODRaw | Blocked | Audit only; the repository explicitly states a code licence, but the image/annotation dataset licence was not verified and no data was downloaded |
| LMOT | Deferred | Validation tracking diagnostic only; the author release currently provides train/validation, persistent IDs and normal/low-light pairs, but no parking-slot polygons or occupancy truth |

AODRaw therefore cannot support slot occupancy or tracking claims. No
MMDetection installation, RAW pretraining or cross-domain distillation was
attempted. If its data licence is later verified, the maximum presently
approved scope is the approximately 4.3 GB downsampled sRGB detector-only
diagnostic with vehicle-class mapping and condition-stratified box metrics.

LMOT cannot establish parking occupancy improvement. If its released
validation data is later acquired, ByteTrack versus TrackTrack may be
compared with a reliable TrackEval path using HOTA, AssA and IDF1, keeping
paired normal/low-light videos in the same split. LTrack reproduction remains
out of scope.

Sources:

- AODRaw repository: <https://github.com/lzyhha/AODRaw>
- LMOT author repository and licence:
  <https://github.com/xinzwang/LMOT>
- LMOT CVPR 2024 paper:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Multi-Object_Tracking_in_the_Dark_CVPR_2024_paper.html>

## 5. Executed smoke evidence

The registered smoke is
`outputs/stage_m_smoke_20260728_v2/`. It repeats four copies of one consumed
PKLot development image over a five-slot subset. The image SHA-256 is
`de17c03f641df869e258ef96afb1eb8baccc986b123e35ac08d802a6ec8d6452`;
the polygon SHA-256 is
`3f44717453a7445cb7f1167be0884a4c69afd02ab82a1e862824646819e9fd05`.
There is intentionally no truth file.

| Run | Frames | Slots | Rows | Events | Output files | Result scope |
|---|---:|---:|---:|---:|---:|---|
| OS0-Controlled continuous | 4 | 5 | 20 | 0 | 7/7 non-empty | Interface and persistence smoke only |
| OS0 static diagnostic | 2 | 5 | 10 | 0 | 7/7 non-empty | Per-image reset/centre-point diagnostic only |
| T0--T3 ablation | 4 | 5 | 20 | 8 | 7/7 non-empty | Common-schema execution smoke only |

The static OS0 metadata records `generation=2`, proving that two images used
two fresh official objects. Continuous OS0 and all three T0--T3 adapters
record `generation=1`, proving one session within the declared source. T2 and
T3 reached `model.track` with their respective frozen tracker files. Unit
tests separately exercise stable track-ID mapping, a moving box, a short
no-detection interval and source-switch resets.

The repeated image produced no official OS0 centre-point occupancy and two
E1b recoveries per T0--T3 frame. Those values are not compared with truth.
Every `metrics.json` correctly reports `not_computed_no_truth`; no Macro F1,
transition metric, tracking metric or improvement claim is generated.

Recorded smoke timings are diagnostic only. T0--T3 averaged 137.29 ms for the
combined four-branch frame path (7.28 FPS from that mean); TrackTrack alone
averaged 38.01 ms and ByteTrack 28.77 ms in this four-frame run. The first
OS0 frame includes model/solution warm-up, so its mean and percentile are not
a steady-state benchmark. The five-slot repeated-image smoke cannot support a
full-lot or real-time claim.

The first output directory, `stage_m_smoke_20260728_v1`, is retained rather
than overwritten. It exposed that `ParkingManagement` required Shapely and
Ultralytics installed version 2.1.2 during the run, contaminating that run's
timing. Shapely version/licence was then added to the runtime freeze and the
clean `v2` smoke was registered. No method parameter changed.

## 6. Outputs, tests and reproduction

Each run exports:

- `occupancy.csv`
- `events.csv`
- `detections.jsonl`
- `annotated.mp4`
- `metrics.json`
- `summary.json`
- `runtime_metadata.json`

The implementation refuses an existing output directory. A formal run also
refuses to start without a truth file and an eligible, hash-verifiable
machine-readable gate.

From `implementation/`:

```powershell
.\.venv\Scripts\python.exe scripts\check_stage_m_data_gates.py
.\.venv\Scripts\python.exe scripts\run_stage_m_smoke.py `
  --output-root outputs\stage_m_smoke_<new-id> --device 0
```

Formal execution, after a future gate becomes eligible:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_m.py `
  --mode ablation --video <test-video> --regions <test-regions> `
  --truth <frozen-truth> --formal-gate <frozen-gate> `
  --claim-scope formal_test --output-root outputs\<new-id> --device 0
```

The pre-Stage-M baseline was 216 passing tests. Ten new targeted Stage M
tests pass, and the final combined suite is 226 passing tests. Every Stage M
input, implementation and registered smoke-output hash is recorded in the
adjacent artifact registry.

## 7. Interpretation and next decision

Stage M completes the runnable OS0 and TrackTrack paths, but it does not
produce a formal comparison because the required independent data does not
exist locally. This is a scientific gate, not an implementation failure.

Even if TrackTrack later improves HOTA or IDF1, that alone will not establish
a slot Macro F1 improvement. Similarly, AODRaw box AP cannot establish P3
occupancy robustness. Stage L already showed the relevant failure mode:
tracking can stabilize a wrong B1 slot--vehicle association, but it cannot
repair the geometry that created the association.

The next justified experiment is to acquire or create a clearly licensed,
fixed-camera, two-scene parking bundle with polygons and human-reviewed
transition truth, freeze it, and run the test scene once. The existing RTX
3060 is sufficient for OS0, TrackTrack, AODRaw sRGB inference or an LMOT
tracking diagnostic; no A100 rental is justified.
