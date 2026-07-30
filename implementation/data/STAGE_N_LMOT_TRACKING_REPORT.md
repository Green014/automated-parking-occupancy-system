# Stage N LMOT low-light tracking diagnostic report

Protocol: `STAGE-N-LMOT-TRACKING-DIAGNOSTIC-20260728-01`

Status: implementation and synthetic metric verification complete; LMOT
acquisition and formal LMOT execution blocked.

Allowed conclusion only: **“LMOT validation上的低光多目标跟踪诊断。”** No
actual LMOT result is available in this Stage N record.

## 1. Preserved evidence

Stage L and Stage M frozen results were not edited. At closeout, the Stage M
report remained
`92ba9d2d5d901e07184d32c383036d797c85aa23d589ba12903f7105991127bb`
and its registry remained
`dd12482338df3c4b9dccac36863dd85151645a1ce7845c5fcec5fa6168b998bd`.
Stage N uses new configuration, data, output, report, and registry names.

## 2. N1 acquisition outcome

The official [LMOT repository](https://github.com/xinzwang/LMOT) confirms
CC BY-NC 4.0 dataset terms, train/validation-only current availability,
paired normal/low-light RAW and sRGB streams, four validation videos, the
nine-column MOT row, and a single Baidu Netdisk entry.

The official page does not publish the share package size or state whether
validation and sRGB can be selected without RAW/train. Login and desktop
client requirements could not be checked because the interactive share was
rejected by the browser security policy, which also prohibits bypassing that
decision. Available space was 117.10 GiB on `C:` and 77.24 GiB on `D:`. These
facts are insufficient to authorize an unknown potentially large package.

Consequently:

- LMOT downloaded bytes: **0**
- LMOT videos/frames/files acquired: **0**
- archive and extracted-file hashes: not applicable
- gate: `blocked_before_download`

The acquisition script never downloads from the network. It accepts a local
ZIP/TAR only, hashes it, checks every member, rejects any train/test/real,
RAW/TIFF, or non-approved path, and refuses to extract an archive with even
one prohibited member.

## 3. N2 format and truth handling

`stage_n_lmot.py` implements strict parsing of:

```text
fn,id,x,y,width,height,ignore,classid,visibility
```

It validates integer identity fields, positive boxes, visibility bounds,
unique frame/track keys, sequence length, exact dark/light frame-number
pairing, GT coverage, track-ID gaps, duplicate/missing frames, unexpected
directories, and image decoding.

The official README lists six names but does not explicitly bind numeric
`classid` values to them, nor define evaluated `ignore` values. Stage N does
not infer either from ordering. Production conversion therefore rejects every
map that is not marked `official_verified` with a hashed evidence source.

Once verified, `car`, `motorcycle`, `bus`, and `truck` are unified as
`motor_vehicle`. Person, bicycle, and explicit-ignore GT remain
prediction-suppression regions, so D1 unified-vehicle boxes on excluded
objects do not become motor-vehicle false positives. Car-only evaluation is
optional and secondary because D1 has one unified vehicle output class.

## 4. N3 official TrackEval

Official TrackEval was cloned with permission from
`https://github.com/JonathonLuiten/TrackEval` and frozen at commit
`12c8791b303e0a0b50f753af204249e622d0281a` (Git tree
`ce583e59d96b33aad5f8f62149d96bc6bc4d8f96`). The 86 tracked files total
904,443 bytes; their deterministic manifest hash is
`2f490a82239cf20f3c36a923b94ba9ef1ffe010eb1e5cc51e3d9c2136d2ff494`.
The MIT licence hash is
`e7d2399dd1ce12eb5f5bb555f708bb0fdc1a95c1490cdb85b2f755b1a6b32019`.

SciPy 1.16.1 was installed from the pinned 38,508,060-byte CPython 3.12
Windows wheel after verifying SHA-256
`f7b8013c6c066609577d910d1a2a077021727af07b6fab0ee22c2f901f22352a`.
The official TrackEval commit predates NumPy 2 and references `np.float` and
`np.int`; the adapter supplies scalar aliases without changing vendor source
or metric logic. Missing optional `pycocotools` disables BURST only and does
not affect HOTA, CLEAR, or Identity.

HOTA, DetA, AssA, IDF1, ID switches, and MOTA all call official TrackEval
metric classes. HOTA and IDF1 were not reimplemented locally.

## 5. N4/N5 frozen run path

L0--L3 all use the same D1 weights, `conf=0.30`, `IoU=0.70`, `imgsz=640`,
agnostic NMS, and `max_det=300`. No training, fine-tuning, threshold search,
or LMOT-driven tracker tuning is allowed.

| Method | Input | Tracker |
|---|---|---|
| L0 | `img_light_rgb` | ByteTrack |
| L1 | `img_light_rgb` | TrackTrack |
| L2 | `img_dark_rgb` | ByteTrack |
| L3 | `img_dark_rgb` | TrackTrack |

Both backends execute through complete Ultralytics
`model.track(..., persist=True, tracker=<yaml>)`. Because TrackTrack may
recover raw pre-NMS candidates, this is a controlled end-to-end tracker
backend comparison, not a claim that both trackers receive an identical
post-NMS detection log.

The runner exports per-sequence and aggregate detection/tracking/runtime
metrics, paired light/dark deltas, backend deltas, JSONL detections/tracks,
runtime metadata, qualitative frames, and a configuration snapshot. Official
TrackEval `combine_sequences` produces tracking aggregates.

## 6. Executed evidence

No LMOT metric was computed. The only Stage N output is
`outputs/stage_n_lmot_synthetic_adapter_20260728_v2/`, explicitly marked
`synthetic_adapter_verification_only`. It contains the required output
contract and 13 SHA-256-recorded files.

| Synthetic fixture | HOTA | DetA | AssA | IDF1 | IDSW | MOTA |
|---|---:|---:|---:|---:|---:|---:|
| Perfect | 100.000 | 100.000 | 100.000 | 100.000 | 0 | 100.000 |
| ID switch | 70.711 | 100.000 | 50.000 | 50.000 | 1 | 75.000 |
| Missed detection | 75.000 | 75.000 | 75.000 | 85.714 | 0 | 75.000 |
| False positive | 89.443 | 80.000 | 100.000 | 88.889 | 0 | 75.000 |

These numbers test plumbing only. They are not LMOT accuracy, a low-light
delta, tracker evidence, or parking-occupancy evidence.

## 7. N6 formal parking data

Local VIRAT 0502, Grand Bassin, PKLot, CNRPark-EXT, and NDISPark were
rechecked. Public CNRPark-EXT, PKLot, DLP, ACPDS, UFPArk, and the AGH Parking
Database were also screened. None simultaneously provides an explicit
licence, fixed cameras in at least two physical scenes, continuous relevant
events, defensible slot polygons and interval/transition truth, and
physically isolated development/test roles.

The formal data gate is therefore
`blocked_no_qualifying_new_data`. No OS0/T0--T3 prediction was run, and VIRAT
0502 was not relabelled as a new test.

## 8. Verification

The complete implementation suite passes 156 tests, including 12 new Stage N
tests. The independent `literature_core` suite passes 82 tests. The new tests
cover parser rejection, the frozen protocol, all four unified motor-vehicle
classes, excluded/ignore suppression, perfect tracking, ID switch, missed
detection, false positive, detection AP, complete Ultralytics tracker
invocation, and passing/failing dark-light frame alignment.

## 9. Limits and next action

The next LMOT step is metadata, not transfer: manually inspect the official
share for archive/member sizes and validation/sRGB selectivity, then obtain
explicit permission before any login, client installation, or large
download. Numeric class-ID and ignore-value semantics must also be supported
by official evidence before conversion.

The next parking step is acquisition or recording of two licensed,
physically separate fixed-camera continuous scenes, followed by frozen
polygons and human-reviewed truth before any prediction. The current RTX 3060
is sufficient; no A100 rental is authorized or needed.
