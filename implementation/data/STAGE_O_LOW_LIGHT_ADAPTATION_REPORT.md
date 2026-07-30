# Stage O Low-Light Detector Adaptation Report

Date: 29 July 2026

Protocol: `STAGE-O-LOW-LIGHT-DETECTOR-ADAPTATION-20260729-01`

## 1. Outcome and claim boundary

Stage O adds an independent raw detector-only low-light diagnostic and one
ordinary supervised D1-LL fine-tune. It does not change B1, E1b, F2, E4,
ByteTrack or TrackTrack, and it does not rewrite Stage L, M, N-v2 or N-v3.

The detector-only adapter calls `ultralytics.YOLO.predict(...)` at the frozen
`imgsz=640`, confidence 0.30, NMS IoU 0.70, class-agnostic NMS and
`max_det=300`. Runtime metadata records `model_track_called=false` and
`tracker_loaded=false`. These measurements are therefore not the
tracker-emitted-box measurements corrected in Stage N-v3.

LMOT supplies paired light/dark vehicle images and motor-vehicle boxes, but no
parking-slot polygons or occupied/vacant truth. Every Stage O number below is
a low-light vehicle-detection diagnostic on already consumed LMOT validation.
It is not a parking-occupancy Macro F1 result and is not an untouched final
test.

## 2. Audit and frozen protocol

The audit confirmed the user-supplied official LMOT annotation transport and
the light and dark split archives. The non-contiguous duplicate
`LMOT_dark_rgb_trainval(1).tarab` is recorded but excluded. Original download
files were read only. Stage N-v2's extracted validation tree, class map,
active-mark rule and person/bicycle suppression policy were reused.

The official train sequences are 02, 04, 06, 08, 09, 12, 16, 18, 20, 21 and
26. Seed 20260729 assigns 06 and 26 to internal development and the other
nine to O3 training. Sampling uses frames 1, 11, ..., 1201. The grouping key
is `(sequence, frame)`, so paired light and dark images never cross a split.
The resulting O3 data contain:

| Partition | LMOT images | LMOT boxes | NDISPark daylight images | NDISPark boxes |
|---|---:|---:|---:|---:|
| training | 2,178 | 33,778 | 112 | 2,577 |
| internal development | 484 | 2,302 | 0 | 0 |

There are 1,331 paired LMOT frame groups and zero cross-split pairs. The
training manifest's canonical logical-data hash is
`61f9bf93ed58548773d5f287c723c81c11a4857f5e4425c043f483386f79c14d`;
the manifest itself has SHA-256
`9c0f7246e7f9167cf5d09698d5f16e9a5ebc1579f7602caecd7b473a7c086b76`.
The successful v4 preparation is additive. Earlier partial and failed
attempts are preserved. A path recorded with the pre-rename
`.partial` root was not overwritten; additive runtime-path correction YAML
and text lists were used.

The protocol froze all inference settings, split IDs, O1 grid, O3 training
settings, early-stopping rule and selection rule before formal LMOT
validation. LMOT validation and NDISPark night remain consumed-development
diagnostics.

The YAML's `created_at=18:00` was a nominal same-day label rather than its
actual wall-clock freeze time. An additive clarification preserves that file
and records the observed chronology: the final config write was 14:49:18,
before O0/O1/O3 formal metrics at 15:18:28/15:29:57/15:37:56. All four formal
config snapshots are semantically equal to the frozen YAML.

## 3. O1 engineering baseline

O1 is a brightness-gated OpenCV Gamma/CLAHE engineering baseline, not a paper
algorithm. Brightness is decided once per sequence from 32 calibration frames
and then held fixed, preventing per-frame parameter changes and video
flicker.

All eight predeclared combinations produced zero IoU-0.50 true positives over
1,151 internally evaluated motor-vehicle GT boxes. The frozen tie-break chose
brightness threshold 45, Gamma 0.50, CLAHE clip limit 3.0 and 8x8 tiles. The
grid was not expanded after this negative result.

On formal dark validation O1 raised recall from 0.034694 to 0.043216 but
reduced precision from 0.722928 to 0.454643, AP50 from 0.034954 to 0.034143
and AP50-95 from 0.018189 to 0.017125. It fails the frozen eligibility rule.
Making an image brighter was not treated as evidence of detection improvement.

## 4. O2 preprocessing-only diagnostic

GLARE was evaluated first at the feasibility gate. Its official route expects
an isolated legacy PyTorch/CUDA stack plus a compiled deformable-convolution
CUDA extension. The local machine had no Python 3.8, conda, `nvcc` or MSVC
`cl`, so GLARE was blocked before download, build or formal inference.

Per the protocol, the only fallback was the already declared Retinexformer.
The official repository was fixed at commit
`1e9a0efce4b306b6701b824768370ff26066c32a`; its MIT license SHA-256 is
`7d0d24f7ca79ea8b2ea9490111df8e134d71360a2c4c643f687ebddab3e36063`.
The official LOL-v2-real checkpoint is 6,478,393 bytes with SHA-256
`539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b`.
It strict-loaded all 122 state-dict keys in a separate Stage O virtual
environment. That environment exposes the existing runtime packages through a
`.pth` file and keeps its Retinexformer-specific additions local; the original
implementation environment was not upgraded or rewritten.

A clean full-resolution 1800x1000 FP32 preflight completed in 1.979 s with
4,399,764,480 peak allocated and 5,905,580,032 peak reserved CUDA bytes. O2
keeps light images unchanged and applies the official full-resolution
enhancer only to the dark stream. It does not use tiling, half precision,
per-frame parameter adaptation or another enhancer.

The completed v7 formal diagnostic processed all 9,680 light/dark
method-frames. On dark images, pooled precision, recall, AP50 and AP50-95
were 0.581802, 0.019307, 0.017284 and 0.011517. This is worse than O0 on
recall, AP50 and AP50-95, so Retinexformer is a negative preprocessing result
under the frozen D1 settings. The enhancer ran 4,840 times, taking
7,434.234 s (1,535.999 ms per enhanced frame); the whole two-stream
evaluation took 8,080.351 s at 1.198 frames/s. Peak allocated/reserved CUDA
memory was 4,436,940,288/5,976,883,200 bytes.

Failed O2 harness attempts remain visible rather than being deleted:
Unicode-unsafe OpenCV decoding, an evidence-writer license filename mismatch,
an Ultralytics configuration-directory permission failure, and an in-place
operation on an inference tensor. The last issue was corrected by a tested
non-mutating tensor conversion before v7. None of those pre-metric attempts
is presented as a formal result.

One repository-wide CPU test run (42.70 s, CUDA hidden) overlapped the long
v7 execution. It cannot change images, boxes or accuracy metrics, but it makes
O2 whole-run wall FPS descriptive rather than a strict isolated speed
benchmark. FPS is only the final tie-break in the frozen selection order, so
this does not decide the accuracy-separated comparison.

## 5. O3 D1-LL supervised fine-tune

O3 starts from the frozen formal D1 checkpoint, SHA-256
`0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`.
It mixes paired LMOT train light/dark sRGB boxes with the existing NDISPark
daylight parking-training split. It uses ordinary YOLO supervision only:
there is no consistency loss, validation threshold tuning or hyperparameter
search.

The one-epoch smoke completed in 77.080 s. It proved that images, labels,
mixed sources, gradients and checkpoints were valid. Its low internal metric
is retained as smoke evidence, not a formal comparison.

The single formal run used the frozen 20-epoch maximum, patience five,
physical batch four, AMP, AdamW, seed 20260729 and the local RTX 3060 Laptop
GPU. The frozen early-stopping rule stopped after epoch 9, five epochs after
the best epoch 4, in 862.910 s. Peak allocated/reserved CUDA memory was
626,047,488/780,140,544 bytes. The final best-checkpoint validation reported
internal-development precision, recall, AP50 and AP50-95 of 0.281878,
0.164639, 0.135992 and 0.071893.

The best checkpoint is 6,240,049 bytes with SHA-256
`99b658bba0ef117d3206b85fc982c81cc0b94839932bfb9e99780027bab1c5da`;
the last checkpoint SHA-256 is
`9a898a0cb9bc2600a67731e26a5aee371629ca13bd7c813078b603861f6c584a`.
Training curves, labels, confusion matrices, arguments, runtime metadata and
both checkpoints are retained.

## 6. Raw detector-only formal results

The primary aggregate pools every prediction under one confidence ordering
while isolating sequence/frame keys. GT, predictions, TP, FP and FN are
summed. The secondary macro is the unweighted mean of per-sequence rates.

### 6.1 Pooled/micro rates

| Method | Light P | Light R | Light AP50 | Light AP50-95 | Dark P | Dark R | Dark AP50 | Dark AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| O0 D1, original | 0.746017 | 0.347380 | 0.319228 | 0.145256 | 0.722928 | 0.034694 | 0.034954 | 0.018189 |
| O1 Gamma/CLAHE | 0.746017 | 0.347380 | 0.319228 | 0.145256 | 0.454643 | 0.043216 | 0.034143 | 0.017125 |
| O2 Retinexformer | 0.746017 | 0.347380 | 0.319228 | 0.145256 | 0.581802 | 0.019307 | 0.017284 | 0.011517 |
| O3 D1-LL | 0.761299 | 0.547709 | 0.514494 | 0.298966 | 0.725393 | 0.259222 | 0.230516 | 0.116695 |

### 6.2 Dark pooled/micro counts

| Method | GT | Pred | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| O0 | 68,887 | 3,306 | 2,390 | 916 | 66,497 |
| O1 | 68,887 | 6,548 | 2,977 | 3,571 | 65,910 |
| O2 | 68,887 | 2,286 | 1,330 | 956 | 67,557 |
| O3 | 68,887 | 24,617 | 17,857 | 6,760 | 51,030 |

### 6.3 Dark per-sequence macro and retention

| Method | Macro P | Macro R | Macro AP50 | Macro AP50-95 | Dark/light recall retention | Dark/light AP50 retention |
|---|---:|---:|---:|---:|---:|---:|
| O0 | 0.693035 | 0.046131 | 0.041674 | 0.021523 | 0.099875 | 0.109494 |
| O1 | 0.417473 | 0.050545 | 0.037195 | 0.018690 | 0.124405 | 0.106956 |
| O2 | 0.551243 | 0.023778 | 0.022065 | 0.012497 | 0.055579 | 0.054143 |
| O3 | 0.662173 | 0.285201 | 0.248940 | 0.127282 | 0.473284 | 0.448044 |

O3 raises dark pooled recall by 0.224527 and AP50 by 0.195563 over O0,
while precision changes by +0.002465 and AP50-95 rises by 0.098506. It also
improves the light stream, which argues against interpreting the gain as a
brightness-only cosmetic effect. O0, O1, O2 and O3 wall throughput was
17.999, 15.018, 1.198 and 24.249 evaluated frames/s respectively; these whole-run values
include decoding, truth processing, rendering bookkeeping and Python
overheads and are not pure model-kernel benchmarks.

## 7. Selection and P3 reintegration

`STAGE_O_SELECTION_20260729.json` applies the rule frozen before formal
comparison. It requires dark recall gain at least 0.02, AP50 gain at least
0.01, non-decreasing AP50-95 and precision loss no worse than 0.05, then
orders eligible candidates by dark AP50, recall, AP50-95, precision and FPS.
No threshold, sequence or failure frame is changed after results.

O1 and O2 fail all four eligibility checks. O3 passes all four: relative to
O0 its dark recall is +0.224527, AP50 is +0.195563, AP50-95 is +0.098506
and precision is +0.002465. The frozen selection therefore names O3 as
`D1-LL`; there is no fallback to O0.

Only the selected detector-side candidate is allowed into a P3 interface
smoke. That smoke uses the existing integrated entry point with frozen B1,
E1b, F2 and E4 and no truth input. Its repeated consumed-development image is
an interface/qualitative artifact only. It does not estimate or imply
nighttime parking-occupancy improvement, and VIRAT 0502 is not reused. The
successful additive v2 smoke processed four frames and five slots, used
temporal stabilization with no tracker, did not select parameters, and wrote
`occupancy.csv`, `events.csv`, `detections.jsonl`, `annotated.mp4`,
`metrics.json`, `summary.json` and `runtime_metadata.json`. Its metrics status
is `not_computed_no_truth`. The retained v1 launch exposed only
Unicode/config-directory harness compatibility and produced no claimed
result.

## 8. Provenance and interpretation

Part I's LTrack/LMOT material motivates testing paired light/dark tracking and
detection robustness. This project does not use LTrack training code or claim
to reproduce LTrack. Stage N-v2/v3 remains a complete-tracker diagnostic;
Stage O is the deliberately separate raw detector diagnostic.

GLARE and Retinexformer are pretrained preprocessing candidates. O2 does not
reproduce GLARE's ExDark/YOLOv3 paper experiment and does not fine-tune either
enhancer. D1-LL is instead a project-specific supervised YOLOv8n adaptation
from D1 using LMOT/NDISPark data. These method classes are not interchangeable.

Historical parking results retain their original boundaries: Stage K is the
frozen static test; Stage L's P3 extension on Stage K is retrospective;
E1b-only Macro F1 is higher than P3; P3's main advantage is lower false-free;
the continuous VIRAT case is a negative result caused by geometric
association; and LMOT supports only low-light detection/tracking robustness,
not parking-slot occupancy.

## 9. Verification and remaining limitations

Stage J (27/27), Stage K (43/43), Stage L (30/30), Stage M (52/52) and the
selected frozen Stage N-v2 control/output records (64/64) pass read-only size
and SHA-256 verification. The Stage N-v2 structural verifier also confirms
19,360 processed frames and 68,887 motor-vehicle GT boxes. The failed first
generic preservation audit is retained; it correctly detected that README,
PLAN and METHOD_PROVENANCE are living documentation updated after Stage M.
The additive v2 audit excludes those three living documents and verifies the
frozen Stage M controls, code, result artifacts and Stage L references.

All 16 Stage O targeted tests pass. The complete repository run passes
270 tests, above the previous 243-test baseline. `compileall` succeeds for
the implementation source, scripts and `literature_core` source;
`parking-run-integrated --help` succeeds; and `git diff --check` reports no
whitespace errors. The detector-only output audit verifies O0/O1/O2/O3
settings, count identities, six-file contracts and no-tracker runtime flags.
The final artifact registry records input GT, model checkpoints, source,
protocol, logs and outputs with byte counts and SHA-256 values.

This stage still lacks a new nighttime fixed-camera parking dataset with slot
polygons and per-slot occupied/vacant truth. Consequently D1-LL cannot yet be
claimed to improve P3 occupancy, event accuracy or general deployment
performance. LMOT validation was previously viewed and must not be reused as
an untouched final test. The NDISPark daylight mix was intended to reduce
parking-domain forgetting, but Stage O has no new independent parking box
test that proves forgetting was eliminated. A genuinely independent night
parking evaluation is the minimum next evidence step.
