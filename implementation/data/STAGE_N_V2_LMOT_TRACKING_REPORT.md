# Stage N-v2 LMOT low-light tracking diagnostic

Protocol: `STAGE-N-V2-LMOT-TRACKING-DIAGNOSTIC-20260729-01`

Status: complete. The official LMOT validation RGB pairs were acquired,
verified, and evaluated without changing detector or tracker parameters.

## 1. Purpose and claim boundary

This experiment tests whether the frozen D1 vehicle detector and two frozen
tracking backends remain effective when the same scenes change from well-lit
to low-light input. It is a supporting robustness experiment motivated by the
LMOT paper from Part I.

LMOT contains moving road users rather than parking-slot polygons and
slot-occupancy truth. Consequently, these results do not measure parking
occupancy Macro F1, do not replace the parking data gate, and do not show that
TrackTrack improves slot occupancy.

## 2. Acquired evidence

The user supplied the official Baidu Netdisk release. All 13 RGB split-tar
parts and the annotation tar were hashed before use. The split archives contain
train and validation members, but the extractor streamed them without creating
a joined tar and wrote only the four validation sequences:

| Sequence | Frames per RGB stream | GT rows |
|---|---:|---:|
| LMOT-05 | 1,210 | 16,663 |
| LMOT-13 | 1,210 | 27,133 |
| LMOT-14 | 1,210 | 19,251 |
| LMOT-25 | 1,210 | 68,734 |
| **Total** | **4,840 light + 4,840 dark** | **131,781** |

The extracted set contains 9,688 files and 14,013,374,410 bytes. Every file
has a frozen SHA-256 record. The dark and light frame numbers align exactly,
all images decode, and no RAW, train, test, or LMOT-real file was extracted.

## 3. Truth interpretation

The released README lists the category order as person, bicycle, car,
motorcycle, bus, and truck. The numeric map `1..6` was not accepted from order
alone. Three boxed samples per class were inspected, followed by every
distinct validation track carrying ID 6. The resulting empirical map is:

| ID | Released category | Validation boxes |
|---:|---|---:|
| 1 | person | 52,585 |
| 2 | bicycle | 10,309 |
| 3 | car | 44,371 |
| 4 | motorcycle | 20,129 |
| 5 | bus | 1,830 |
| 6 | truck | 2,557 |

ID 6 includes conventional trucks and the release's broader commercial-vehicle
boundary, including vans, an ambulance, and a cargo tricycle. No label was
changed. All 131,781 rows use value `1` in the column called `ignore` by the
LMOT README. Since LMOT declares MOTChallenge17 organization, and treating
`1` as ignored would leave no evaluable truth, Stage N-v2 treats it as the
positive active/evaluated mark.

The primary diagnostic merges car, motorcycle, bus, and truck into
`motor_vehicle`. Person and bicycle boxes are retained only as
prediction-suppression regions.

## 4. Frozen comparison

All methods use the same D1 weights, `conf=0.30`, NMS IoU `0.70`,
`imgsz=640`, agnostic NMS, and `max_det=300`. No LMOT training, fine-tuning,
threshold selection, or tracker tuning was performed.

| Method | Illumination | Tracker |
|---|---|---|
| L0 | Well-lit RGB | ByteTrack |
| L1 | Well-lit RGB | TrackTrack |
| L2 | Low-light RGB | ByteTrack |
| L3 | Low-light RGB | TrackTrack |

HOTA, DetA, AssA, IDF1, ID switches, and MOTA are calculated through official
TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`.

## 5. Aggregate results

| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Emitted-box recall | AP50 | Latency (ms) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 26.613 | 23.422 | 30.582 | 34.147 | 22.039 | 481 | 0.228 | 0.192 | 13.336 | 75.20 |
| L1 | 22.940 | 17.524 | 30.207 | 29.840 | 20.635 | 147 | 0.185 | 0.169 | 36.826 | 27.17 |
| L2 | 6.028 | 2.053 | 17.894 | 4.442 | 2.114 | 39 | 0.042 | 0.038 | 13.157 | 76.03 |
| L3 | 3.454 | 0.650 | 18.553 | 1.552 | 0.814 | 7 | 0.044 | 0.040 | 41.371 | 24.22 |

The detection columns describe boxes emitted by the complete
`model.track(...)` path after excluded-class suppression. They are not raw
detector-only AP. This distinction matters because the tracker backends can
emit different subsets of the shared detector candidates.

## 6. Findings

Low light is the dominant failure mode. ByteTrack HOTA falls by 20.585 points
from L0 to L2, while emitted-box recall falls from 0.228 to 0.042. TrackTrack
HOTA falls by 19.485 points from L1 to L3. The higher dark precision is caused
by conservative output and must not be interpreted as improved robustness.

TrackTrack reduces identity switches from 481 to 147 under well-lit input and
from 39 to 7 under low light. This benefit comes with lower detection coverage:
against ByteTrack, HOTA is 3.673 points lower in well-lit scenes and 2.574
points lower in low light. Under low light, TrackTrack AssA is 0.659 points
higher, but DetA is 1.404 points lower. Thus, the frozen TrackTrack settings
favor continuity among a small set of confident tracks rather than recovering
the many targets missed by D1.

ByteTrack inference/tracking runs at about 75 FPS. TrackTrack runs at 27.17 FPS
for well-lit and 24.22 FPS for low-light frames, approximately 2.77 and 3.14
times slower respectively. These timings exclude image decoding and file I/O,
but both remain above LMOT's 20 FPS source rate on the RTX 3060 Laptop GPU.

## 7. Limitations

The D1 detector was fine-tuned for parking-lot vehicles, not LMOT road scenes,
so the low absolute scores combine illumination shift with dataset and viewpoint
shift. LMOT's four validation sequences are paired and valuable for controlled
diagnosis, but they are not an untouched parking test set.

Some released track IDs contain long temporal gaps. They were preserved
exactly because the official total of 626 unique validation IDs matches the
parsed annotations; silently splitting or renumbering them would create a
different benchmark.

Ultralytics emitted a repeated deprecation warning for the explicit
`half=False` argument. The run completed normally and the warning did not
change inference settings, but future code should replace the deprecated API
without altering this frozen result.

## 8. Decision

L0 remains the stronger of the tested tracking configurations for this
diagnostic. TrackTrack should remain an optional variant and its lower ID-switch
count should be reported together with its much lower coverage. More tracker
tuning on LMOT is not justified because it would consume the diagnostic set as
development data.

The practical next step is not another tracker search. If low-light robustness
must be improved, it should be a separately frozen detector or image-processing
experiment with development data distinct from these four LMOT validation
sequences. The main parking workflow and its slot-level conclusions remain
unchanged.
