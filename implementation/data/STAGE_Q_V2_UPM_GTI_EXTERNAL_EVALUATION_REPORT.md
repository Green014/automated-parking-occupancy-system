# Stage Q-v2 — UPM-GTI External Night Parking Occupancy Evaluation

Date: 2026-07-29  
Protocol:
`STAGE-Q-V2-UPM-GTI-EXTERNAL-NIGHT-OCCUPANCY-20260729-01`  
Final status: **FORMAL_RUNS_COMPLETE**  
Completed output:
`implementation/outputs/stage_q_v2_upm_gti_external_20260729_v2`

## Executive result

The official ETSIT/UPM-GTI `test.zip` was downloaded, verified, safely
extracted and audited before any model was loaded. The archive contains a
real fixed-camera parking scene, 3,180 800x600 JPEG images, 26 sequence
directories and 26 `groundtruth.txt` files. The frozen low-light selection
contains 376 previously unused official-Test images from 17 sequences and
7,896 known slot states. Its 21-value vectors are complete and contain 90
slot-state changes.

After the user explicitly confirmed the pre-model polygon overlay, the two
frozen P3 methods were run once each with identical settings and no tracker.
P3-D1 produced the stronger overall external occupancy result:

- Macro F1: **0.664318** versus 0.617484 for P3-D1-LL;
- occupied recall: 0.446115 versus **0.457393**;
- false-free rate: 0.553885 versus **0.542607**;
- occupied precision: **0.368530** versus 0.269572;
- vacant recall: **0.914060** versus 0.860665;
- false-occupied rate: **0.085940** versus 0.139335;
- count MAE: **1.329787** versus 1.909574.

D1-LL therefore recovered nine additional occupied slot labels and reduced
false-free by 1.13 percentage points, but added 379 false-occupied labels.
The precision, vacant recall, Macro F1, accuracy and count-error regressions
are material. D1 remains the P3 default. D1-LL remains a secondary frozen
comparison, and Stage P2 remains `FAIL`.

This is useful external low-light slot-occupancy evidence from one shared
camera geometry. It does not prove universal generalization and does not
convert the earlier LMOT detector result into a parking-occupancy claim.

## Source acquisition and legal boundary

Official sources:

- dataset page:
  <https://gti.ssr.upm.es/data/parking-lot-database>;
- official public storage:
  <https://drive.upm.es/index.php/s/TdqfDr25NAsGIea>;
- associated Sensors paper:
  <https://www.mdpi.com/1424-8220/23/6/3329>.

The public share required no login and presented no additional agreement
prompt. The exact downloaded archive is:

| Field | Value |
|---|---|
| Official object | `test.zip` |
| Exact bytes | 250,698,837 |
| SHA-256 | `92d61d8f87fe3e7068d8c42ce8dc2c415c08071c92eeddfd4d47260e8922efdc` |
| Server SHA-1 | `9e462c0720eddf92bb11b4eed7d5e0e597112a5f` |
| Server MD5 | `3157555f948f225621bc618656b60e75` |
| ZIP CRC | all members passed |
| Archive root | `test/` |
| Local policy | ignored raw/extracted data; no redistribution |

The archive is used only for local, non-commercial course research. The
official site calls the resource public, but no standalone image-dataset
license text was found. The record therefore remains:

```yaml
official_public_download: true
explicit_dataset_license_found: false
use_scope: local_noncommercial_course_research
redistribution: prohibited_by_project_policy
attribution_required: true
legal_interpretation_not_claimed: true
```

The article's CC BY 4.0 status is not treated as an automatic license for the
image archive.

## Archive and truth audit

The ZIP has 3,259 entries: 3,206 files and 53 directories, with
250,835,482 uncompressed member bytes. The extractor rejected traversal,
absolute paths, backslash paths, symbolic links, encryption and
case-insensitive duplicates before extraction.

Extracted structure:

| Item | Count / result |
|---|---:|
| `gopro*` sequence directories | 26 |
| JPEG images | 3,180 |
| `groundtruth.txt` files | 26 |
| Parsed truth records | 3,148 |
| Binary slot labels | 66,108 |
| Source `0` / occupied labels | 28,858 |
| Source `1` / vacant labels | 37,250 |
| Resolution | 800x600 throughout |
| Reliable source FPS/timestamps | unavailable |

Twenty-five sequences have exact image/truth bijection. `gopro10` has 67
images but only 35 truth records, so the entire sequence was excluded under
the predeclared rule. No missing image was silently dropped from an included
sequence.

## Frozen night gate and annotations

Night membership was determined before model output from the official Test
split, the fixed pre-model contact sheet and image content. Mean grayscale
luminance at or below 70 was frozen as an auxiliary per-image rule. Model
predictions were not used to select sequences or images.

The selected manifest contains:

- 17 sequences:
  `gopro1`, `gopro4`, `gopro5`, `gopro8`, `gopro9`, `gopro11`, `gopro12`,
  `gopro19`, `gopro23`, `gopro24`, `gopro25`, `gopro26`, `gopro30`,
  `gopro33`, `gopro34`, `gopro36`, and `gopro46`;
- 376 images;
- 7,896 known slot labels, with zero unknown labels;
- 50 selected transition frames and 90 slot-state changes;
- manifest CSV SHA-256
  `8929e6a38b36b578ae2658127625576e632904437d0ba5d2f37470fc0b0746ba`;
- canonical logical-manifest SHA-256
  `d4d391fd0ad11f5c03f1f44edb268df9e5989da5ad413b595d15c98f12f9791e`.

The 21 polygons were drawn before model output using the official paper's
Figure 4(a) numbering and an empty official Test frame. Slot IDs are
`slot_00` through `slot_20`, exactly aligned with vector positions 0 through
20. The user then supplied the explicit confirmation text
`确认 polygon`. The earlier blocked annotation file remains unchanged; an
additive confirmation record binds:

- polygon SHA-256
  `9547f75aed308b5958d475cd225769f0d1d0939d1596cefbee5faf9a3ef6dd66`;
- truth SHA-256
  `ecce202c09182078d60c1e98b4c6f3ad1512b6ff6c4332f7ee4e68c694d636e6`;
- validation-overlay SHA-256
  `4f03e960937bc2848bc3f670d4e4671afbd93dc2b1f8b49e52a9201a3df36ae4`.

## Frozen P3 comparison

The methods differ only in detector weights:

1. `QV2-0`: P3-D1, the current primary/default detector;
2. `QV2-1`: P3-D1-LL, the Stage O secondary frozen comparison.

Shared parameters:

| Component | Frozen setting |
|---|---|
| detector | `YOLO.predict`, imgsz 640, confidence 0.30 |
| NMS | IoU 0.70, class-agnostic, `max_det=300` |
| classes | project vehicle class 0 |
| B1 | polygon coverage 0.40, greedy one-to-one assignment |
| E1b/F2 | threshold 0.76, detector-negative slots only |
| E4 | enabled; rise 0.60, fall 0.15 |
| E4 hysteresis | occupied 0.58, vacant 0.42 |
| tracker | none |

Weight bindings:

| Model | Bytes | SHA-256 |
|---|---:|---|
| D1 | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| D1-LL | 6,240,049 | `99b658bba0ef117d3206b85fc982c81cc0b94839932bfb9e99780027bab1c5da` |
| E1b | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

Each of the 17 source switches recreated detector session state, temporal
filter state and event state. Runtime metadata records 17 detector
generations per method, backend `ultralytics_model_predict`, tracker type
`null`, and `model_track_called=false`.

## Formal occupancy results

All metrics use the same 7,896 known slot labels.

| Metric | QV2-0 P3-D1 | QV2-1 P3-D1-LL | D1-LL minus D1 |
|---|---:|---:|---:|
| Macro F1 | **0.664318** | 0.617484 | -0.046834 |
| Occupied precision | **0.368530** | 0.269572 | -0.098958 |
| Occupied recall | 0.446115 | **0.457393** | +0.011278 |
| Vacant recall | **0.914060** | 0.860665 | -0.053395 |
| False-free rate | 0.553885 | **0.542607** | -0.011278 |
| False-occupied rate | **0.085940** | 0.139335 | +0.053395 |
| Accuracy | **0.866768** | 0.819909 | -0.046859 |
| Count MAE | **1.329787** | 1.909574 | +0.579787 |
| Count RMSE | **1.736651** | 2.337256 | +0.600605 |
| TP | 356 | **365** | +9 |
| TN | **6,488** | 6,109 | -379 |
| FP | **610** | 989 | +379 |
| FN | 442 | **433** | -9 |

This is slot-occupancy evaluation, not raw detector AP. The result reflects
the entire frozen D1/D1-LL → B1 → E1b/F2 → E4 path.

### Frame-only transition evidence

The source provides neither reliable timestamps nor FPS. No seconds-level
transition latency was calculated. Event matching uses adjacent
ground-truth transition windows and three selected ordered frames of target
state stability.

| Metric | P3-D1 | P3-D1-LL |
|---|---:|---:|
| Ground-truth transitions | 90 | 90 |
| Matched | 39 | 41 |
| Missed | 51 | 49 |
| Early / on-time / delayed | 0 / 8 / 31 | 0 / 8 / 33 |
| State-change agreement | 0.433333 | 0.455556 |
| Median steady-state latency | 4 selected frames | 4 selected frames |
| P90 steady-state latency | 8 selected frames | 9 selected frames |

The selected low-light manifest can contain gaps in the original sequence
index. For that reason, selected-frame differences and source-frame-index
differences are both retained in `metrics.json`; neither is converted to
seconds. `annotated.mp4` is a 2 FPS visualization reconstruction and is not
presented as the original capture rate.

## Qualitative and failure analysis

The fixed diagnostic categories are descriptive labels rather than proof of
component causality:

| Category | P3-D1 | P3-D1-LL |
|---|---:|---:|
| detector-negative not recovered by E1b/F2 | 432 | 424 |
| detector or B1 geometry false-occupied | 152 | 396 |
| E1b/F2 classifier override | 35 | 33 |
| E4 temporal lag/carryover | 433 | 569 |

The contact sheets show the principal trade-off directly: D1-LL produces
more low-confidence vehicle boxes around bright structures and boundary
regions. Those boxes sometimes recover a missed occupied slot, but they also
enter B1 and create substantially more false-occupied slot evidence. E4 can
then carry those errors for later selected frames. Some false-free errors
remain detector-negative after E1b/F2 review in both methods.

These categories cannot fully separate detector error, B1 geometry,
polygon ambiguity and source-truth ambiguity without independent
per-object-box truth for this dataset. No category was used to retune any
component.

## Preserved failed attempt

The first formal process stopped before detector construction or prediction
because Ultralytics attempted to create its settings directory outside the
writable workspace. It produced zero detector records and zero occupancy
rows. The partial v1 output was not deleted, rewritten or reused.

An additive v2 runtime protocol changed only:

1. the output root, from the preserved partial `_v1` directory to `_v2`;
2. `YOLO_CONFIG_DIR`, placing Ultralytics settings inside the new output.

Manifest, truth, polygons, weights, P3 parameters, roles and method order
were byte-identical. The successful v2 run is therefore the only completed
run of either method.

## Output contract and runtime

Each method produced 376 annotated ordered frames, a reconstructed
`annotated.mp4`, 7,896 occupancy rows, 376 detection records, events,
metrics, summary, runtime metadata, a confusion matrix, a failure-case JSON
and a qualitative contact sheet.

Descriptive whole-method throughput on the local RTX 3060 Laptop GPU was
7.111 FPS for P3-D1 and 6.801 FPS for P3-D1-LL. Runtime was not a selection
criterion. The two completed method directories plus root audit/runtime
artifacts contain 776 files and 108,989,616 bytes before adding the registry.

## Supported and unsupported claims

Supported:

- the official archive was valid and contained qualifying fixed-camera
  external low-light parking data with per-image slot truth;
- P3-D1 had higher overall slot Macro F1, precision, vacant recall, accuracy
  and lower count error than P3-D1-LL on this frozen external evaluation;
- P3-D1-LL had a small occupied-recall/false-free advantage coupled to a
  much larger false-occupied regression;
- both methods used the same frozen P3 path with no tracker or retuning.

Not supported:

- D1-LL as the new default parking detector;
- universal low-light generalization from one camera geometry;
- a claim that LMOT detector AP directly improved parking occupancy;
- detector AP, tracking robustness or real-time video performance from these
  slot labels and low-frame-rate image sequences;
- seconds-level transition latency.

Stage P2 remains `FAIL`; D1 remains the system default; D1-LL remains a
selected low-light detector candidate and secondary comparison.

