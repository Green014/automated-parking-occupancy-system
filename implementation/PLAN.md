# Part II Implementation Plan

Project: **Automated Parking Lot Occupancy and Tracking System**

Updated: 28 July 2026

## 1. Current objective

The project is in final closure, not open-ended model exploration. Dataset
alignment, detector fine-tuning, detector comparison, polygon-based
slot-occupancy integration and the untouched Stage K evaluation are complete.
The remaining work is report/demo packaging; frozen test results must not be
used for additional tuning or model reselection.

The completed main experimental sequence is:

```text
Part I dataset audit
  -> dataset preprocessing and frozen splits
  -> D0 pretrained YOLOv8n baseline
  -> D1 parking-domain YOLOv8n fine-tuning
  -> D0/D1/D2 detector comparison
  -> B1 polygon slot assignment
  -> P0/P1/P2 slot-occupancy evaluation
  -> P3 E1b asymmetric fusion and optional temporal/tracking support
```

YOLO-World is D2, a detector-replacement comparison rather than a default
parallel branch. P3 is complete and has a separate
`parking-run-integrated` entry; ByteTrack and TrackTrack remain optional.

## 2. Immutable evidence and data boundaries

- Existing frozen experiment outputs must not be overwritten, renamed, or
  edited.
- CNR-EXT is a consumed once-only static external evaluation and cannot be
  used for new parameter selection.
- VIRAT 0503 has been viewed and consumed and cannot be a new method's
  untouched holdout.
- Every new method or rerun uses a new experiment ID, tracked configuration,
  and new output directory.
- Development results, including negative results, are retained.
- Local implementations are described as local adaptations/integrations, not
  as forks of projects whose code was not used.
- No large-data download, detector training, or remote GPU use is planned
  without a separately justified decision.
- NDISPark validation is development validation because it has already been
  used for detector comparison. Its test split is count-only.
- PKLot/CNRPark slot polygons cannot be represented as vehicle boxes.
- Machine-generated Grand Bassin boxes remain preannotations, not human truth.

## 3. Closed baseline definitions

The canonical method/configuration registry is
`configs/baseline_methods.yaml`; the human-readable audit is
`BASELINE_CLOSURE.md`.

| ID | Definition | Status |
|---|---|---|
| B0 | YOLOv8 + bounding-box centre mapping | Closed, runnable |
| B1 | YOLOv8 + polygon coverage | Closed, runnable |
| E0 | Historical static CNR-EXT YOLOv8 coverage baseline | Frozen historical result only |
| T0 | Raw YOLOv8 temporal comparator | Closed, runnable on future protocol-frozen data |

The old `proposed` baseline remains a compatibility label for its historical
ByteTrack/hysteresis engineering path. It is not the final main method.

## 4. Corrected evaluation protocol

Slot classification still reports confusion matrix, accuracy, occupied and
vacant recall, Macro F1, false-free rate, and false-occupied rate.

For each real state transition, temporal evaluation now matches the nearest
observed prediction change into the target state only when it persists for
the configured `stable_frames`. The event records:

- signed prediction-frame minus truth-frame error, in frames and seconds;
- direction (`entry` or `exit`);
- outcome (`early`, `on_time`, `delayed`, or `missed`);
- the truth and matched prediction frames.

Early events are not inserted as zero-latency events. Non-negative
post-truth latency remains available for compatibility, while signed timing
is the primary early/late measure. Unsupported changes remain flicker;
additional changes inside a matched transition window remain transition
instability.

## 5. Stage status and exit evidence

| Stage | Status | Exit evidence |
|---|---|---|
| A. Evaluation and baseline closure | Complete | Signed transition timing tests; B0/B1/E0/T0 registry; canonical four-artifact pipeline; both suites green; frozen checksums unchanged |
| B. Part I dataset alignment | Complete with acquisition gate | `data/PART1_DATASET_ALIGNMENT.md` and `data/part1_dataset_alignment.yaml` |
| C. Frozen data protocol | Complete | `DPROTO-NDISPARK-ONLY-20260727-01`; 112/30/117 official memberships; source/image/artifact hashes; class mapping; leakage audit; dataset card |
| D. Reproducible preprocessing | Complete | `DPREP-NDISPARK-ONLY-20260727-01`; one-class COCO-to-YOLO; 259 image hashes; 3,302 boxes; action log; count MAE/RMSE; Ultralytics load check |
| E. D0/D1/D2 comparison protocol | Complete | `D-COMP-NDISPARK-DEV-20260727-01`; canonical class adapter; frozen conditions/metrics; real preflight blocked before inference on missing D1 |
| F. Local D1 smoke run | Complete | `D1-NDISPARK-SMOKE-20260727-01`; 3 pretrained epochs, batch 4, finite changing losses, 30-image validation, 0.715 GiB peak reserved VRAM |
| G. GPU decision | Complete | `GPU-GATE-NDISPARK-D1-20260727-01`; local 640/batch-4 selected; paid GPU and A100 rejected |
| H. Formal D1 training | Complete | `D1-NDISPARK-FT-20260727-01`; 47 epochs, early stop, epoch-37 best checkpoint frozen |
| I. Detector evaluation | Complete with corrected v2 | Historical v1 retained; class-agnostic NMS, per-model development calibration and max-det sensitivity completed; D1 retained |
| J. Parking pipeline integration | Complete on consumed development | P0/P1/P2 use identical B1 geometry; full seven-file output contract; 27/27 artifacts verified |
| K. Final slot evaluation | Complete | Data-gate v2 passed before predictions; 90 zero-overlap PKLot images, 5,034 known slot labels; P0/P1/P2 frozen once; 43/43 result artifacts verified |
| L. Integrated literature workflow | Complete with mixed result | P3 connected D1+B1+E1b+E4+optional ByteTrack; static retrospective Macro F1 0.987061; E1b-only 0.992226; VIRAT departure remained a documented geometry-driven miss |
| M. Report/demo evidence | Complete with declared limitations | Final Stage K report, date/weather tables, failure cases, continuous P1+B1 negative case, exact checksums |

### Stage B decision gate

The audit's scientific recommendation remains CARPK train plus a balanced
NDISPark supplement. On 2026-07-27 the user chose the local NDISPark-only
backup route instead. The Zenodo API confirms NDISPark as `odc-by` with open
access. The incomplete resumable PKLot archive was preserved; later, 90
complete and individually hashed JPG/XML pairs before its truncated boundary
were admitted by a separate Stage K data gate. CARPK remains deferred.

Stage C froze NDISPark's 112-image train split, consumed 30-image development
validation, and 117-image count-only test as
`DPROTO-NDISPARK-ONLY-20260727-01`. This route cannot produce a new
detector-mAP test or untouched slot-occupancy test; those limitations remain
explicit. Stage D preprocessing, the Stage E comparison protocol, the Stage F
smoke run, the Stage G GPU decision, the Stage H formal D1 run, the corrected
Stage I-v2 evaluation and Stage J integration are complete. The original
Stage K local-inventory blocker remains historical; data-gate v2 later passed
before prediction using 90 previously unused PKLot images with zero hash
overlap against Stage J.

Stage E freezes D0 as COCO-pretrained YOLOv8n, D1 as
NDISPark-fine-tuned YOLOv8n, and D2 as zero-shot YOLO-World under identical
640-pixel, one-class conditions. Source detector classes are filtered during
prediction and then mapped to project class 0; dataset truth is never filtered
by COCO class IDs. The actual preflight loaded no model and ran no prediction:
D0/D2 hashes passed and missing D1 correctly blocked comparison execution.
See `data/DETECTOR_COMPARISON_PROTOCOL.md`.

Stage F completed locally as `d1_ndispark_smoke_20260727_v3` after retaining
two zero-epoch engineering failures in separate v1/v2 directories. The
successful three-epoch run used the exact frozen settings, processed all 30
development-validation images, changed all three training losses, updated the
checkpoint hash, and reported no NaN, OOM, batch reduction, or material
dataloader wait. Peak Torch reserved memory was 767,557,632 bytes on the
6,441,926,656-byte RTX 3060. This smoke checkpoint is not formal D1. Stage G
GPU/local-training feasibility is the next gate; see
`data/D1_SMOKE_REPORT.md`.

Stage G used only the frozen Stage F summary and ran no training or prediction.
At 640/batch 4, the measured peak was 0.715 GiB. The 50-epoch central,
conservative, and stress-bound estimates are respectively 2.43, 4.35, and
8.52 minutes. The formal run remains at the only execution-validated
configuration, physical batch 4; Ultralytics nominal batch 64 yields 16
post-warm-up accumulation steps. Analytical 960/1280 projections are retained
as estimates only and are not selected. Six GiB is the recommended minimum,
local execution passes, rental duration is zero, and an A100 is unnecessary.
Stage H is authorized only as `D1-NDISPARK-FT-20260727-01`, in a new output
directory, freshly initialized from frozen COCO-pretrained `yolov8n.pt`. See
`data/GPU_DECISION_REPORT.md` and
`configs/d1_ndispark_formal_frozen_20260727.yaml`.

Stage H executed the one frozen seed locally from the original D0 weights.
Early stopping ended training at epoch 47; epoch 37 was selected with
development Precision 0.93708, Recall 0.88339, mAP@0.5 0.94478, and
mAP@0.5:0.95 0.67556. The best checkpoint SHA-256 is
`0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`.
These are consumed-development diagnostics and do not yet compare D1 with D0
or D2.

The initial runner retained a post-run resource callback audit failure after
training and final best-checkpoint validation had completed. Existing
artifacts were finalized offline without loading a model or rerunning
training. The failure remains recorded, the callback is fixed and tested, and
all 14 selected artifacts pass their frozen hashes. See
`data/D1_FORMAL_TRAINING_REPORT.md`.

Stage I ran the frozen canonical comparison on the consumed development
validation. D1 ranked first by mAP@0.5:0.95 (0.64969) and recall (0.84160);
D2 ranked second and D0 third. D1 was selected before test prediction. A
single threshold of 0.10 was then selected from the fixed
0.05-to-0.95 development grid by aggregate D0/D1/D2 count MAE and frozen as
`D-COUNT-NDISPARK-TEST-20260727-01`.

The 117-image count-only test produced a scientifically useful negative
result: D2 MAE 2.58974, D0 MAE 2.99145, and D1 MAE 3.46154. No per-model
thresholds, post-test reselection, detector mAP, or box precision/recall were
introduced. Development FP/FN montages and night dense-overlap evidence were
exported and visually checked; no official occlusion tag was invented. All 24
frozen Stage I evidence artifacts passed hashes. See
`data/DETECTOR_EVALUATION_REPORT.md`.

Stage J froze and executed P0/P1/P2 with the same B1 0.40 polygon coverage,
one-to-one assignment, class-agnostic NMS, `max_det=300` and no temporal
filtering. Consumed-development Macro F1 was 0.768040, 0.825723 and 0.735168.
P1 led but retained a 0.328930 false-free rate. All methods wrote the same
video/CSV/JSON contract and all 27 registered artifacts passed hashes.

Stage J's read-only grouped analysis qualified that pooled lead: P1 versus P0
was 6 wins, 2 ties and 19 losses by image, with a paired 95% interval that
included zero.

Stage K then froze and executed the same P0/P1/P2 configurations once on 90
previously unpredicted PKLot images and 5,034 known slot labels. Pooled Macro
F1 was 0.785612, 0.808398 and 0.796548. P1's pooled lead was not
camera-stable: camera-macro F1 was 0.841099, 0.824835 and 0.849168, and P1
versus P0 was 17 wins, 11 ties and 62 losses by image. No test result changed
the preselected D1 provenance, threshold, checkpoint or mapping rule.
`data/STAGE_K_FINAL_REPORT.md` records the result and limitations.

## 6. Optional F2 guardrails

F2 will compare centre evidence, core-region coverage, and one-to-one
geometric assignment. YOLOv8 is primary evidence. E1b may participate only
inside an uncertainty interval selected from development data. Low-confidence
classification cannot override clear detector evidence. Temporal hysteresis
may suppress short fluctuations but cannot be credited with repairing a
long-term geometry error. Every branch input, gate decision, and output must
be logged.

F2 will not begin before D1 training, D0/D1/D2 comparison, and P0/P1/P2
occupancy comparison are complete. It will not run on an external test set
during development. If the development ablation does not support it, the
negative result is frozen without repeated tuning to manufacture an
improvement.

## 7. Run and artifact contract

Canonical geometry baselines use one entry:

```powershell
.\.venv\Scripts\parking-run.exe `
  --input <video> `
  --slots <slot-map.json> `
  --method B0 `
  --output-dir <new-output-directory>
```

`B0`, `B1`, and `T0` are valid. Canonical detector and mapping parameters come
from the registry. Normal runs produce `annotated.mp4`, `occupancy.csv`,
`events.csv`, `detections.jsonl`, `summary.json`, `metrics.json`, and
`runtime_metadata.json`.

New output directories are mandatory for new experiments. Evaluation
recomputations use a new directory or filename and never replace historical
metrics.

## 8. Verification gates

- Run the baseline package tests after changes under `implementation/`.
- Run the `literature_core` tests after changes under that module.
- Run source compilation for both Python trees.
- Check `git diff --check`.
- Confirm frozen artifacts are untouched with `git status` and, when needed,
  the existing frozen-artifact verifier.
- Update README/results/provenance documents only with executed evidence.

## 9. Submission evidence

Completed evidence:

- frozen detector and slot datasets with source hashes and leakage audits;
- pretrained D0 versus fine-tuned D1 versus YOLO-World D2 comparison;
- P0/P1/P2 under identical B1 polygons and mapping;
- a 90-image untouched Stage K slot-occupancy evaluation;
- grouped camera/date/weather reporting and image-level bootstrap intervals;
- a real continuous P1+B1 video case with the missed transition retained;
- final method, timing, failure-case and checksum records.

Stage L subsequently executed one controlled P3 extension without changing
P0/P1/P2 or Stage K. P3 uses D1+B1 as primary evidence and invokes E1b only
for detector-negative slots before optional E4/E5 processing. On the
retrospective Stage K images it increased Macro F1 from 0.808398 to 0.987061,
but E1b alone remained higher at 0.992226. On VIRAT 0502 all variants missed
the departure because the oblique slot polygon continued to overlap an
adjacent stationary vehicle. No post-result polygon or threshold tuning was
performed. See `data/STAGE_L_INTEGRATED_WORKFLOW_REPORT.md`.

No YOLO-World fine-tuning is required for closure. Submission packaging
should clearly state that the sampled PKLot montages are not temporal
evidence, Stage L's Stage K extension is retrospective rather than untouched,
and the VIRAT continuous case is a negative consumed-development result.

## 10. Stage M open-source tracking robustness closeout

The earlier stage table's “M. Report/demo evidence” label predates this
additive extension. The current Stage M protocol is
`STAGE-M-OPEN-SOURCE-TRACKING-ROBUSTNESS-20260728-01`; it does not replace or
rewrite the earlier evidence.

| Work item | Status | Exit evidence |
|---|---|---|
| M0 source/licence freeze | Complete | Ultralytics 8.4.104, ParkingManagement, TrackTrack registration/implementation, tracker YAMLs, package versions and licence files are hash-checked |
| M1 OS0-Controlled | Smoke complete; formal gated | Official `ParkingManagement` with D1/shared settings/TrackTrack; local per-slot audit only; static reset verified |
| M2 T0--T3 | Smoke complete; formal gated | T0 raw gate, T1 + E4, T2 + ByteTrack, T3 + TrackTrack; shared output schema and reset/ID/no-detection tests |
| M3 continuous video | Blocked | No new licensed two-scene fixed-camera bundle with distinct human-reviewed polygon and transition truth |
| M4 adverse conditions | Partially eligible by task | NDISPark detector/count only; end-to-end low-light occupancy and AODRaw formal use blocked |
| M5 LMOT | Deferred | Validation tracking diagnostic only if acquired; no slot-occupancy claim |

The registered smoke is `outputs/stage_m_smoke_20260728_v2/`. It contains no
truth by design, and all accuracy/transition metrics remain uncomputed. The
first `v1` directory is retained as a dependency-discovery run: Ultralytics
installed Shapely 2.1.2 during its first ParkingManagement call. Shapely was
then included in the runtime freeze before `v2`; no model, tracker or
threshold parameter changed.

Before any future formal run, the data gate must verify two physically
distinct scenes, different hashed video/polygon/truth bundles, fixed cameras,
human review, relevant events, licence status, configuration freeze and zero
prior executions of the test scene. The runner also requires the truth and
gate files and refuses to overwrite an output directory.

No A100 is authorized. The existing RTX 3060 is sufficient for the currently
allowed OS0/TrackTrack, AODRaw sRGB inference or LMOT validation diagnostic.
RAW pretraining, distillation or LTrack retraining requires a separate
compute/storage decision.

## 11. Stage N LMOT diagnostic and parking data gate

| Work item | Status | Exit evidence |
|---|---|---|
| N1 limited LMOT acquisition | Blocked before download | Official source/licence/release checked; package size, val/sRGB selection and client/login requirements unresolved; 0 downloaded bytes |
| N2 format and truth audit | Implemented; real data gated | Strict nine-field parser, paired-frame/GT/ID/image audit, no inferred numeric class IDs, unified motor-vehicle and excluded-object suppression tests |
| N3 TrackEval | Complete | Official commit `12c8791...`, MIT licence and source-tree hash frozen; official HOTA/CLEAR/Identity synthetic tests pass |
| N4 frozen L0--L3 design | Complete | D1 and inference settings unchanged; ByteTrack/TrackTrack both use full Ultralytics tracking path |
| N5 LMOT metrics | Synthetic plumbing only | Required output schema exists; no LMOT metric or low-light conclusion |
| N6 formal parking data | Blocked | Local and public candidates fail the complete two-scene/event/truth gate; no OS0/T0--T3 run |

Formal LMOT execution needs an approved local RGB validation archive plus
officially evidenced numeric class IDs and ignore semantics. Formal parking
execution still requires two physically distinct licensed continuous scenes,
frozen development/test roles, polygons and human-reviewed interval or
transition truth before prediction.

## 12. Stage N-v2 LMOT completion

Stage N-v2 preserves the original blocked Stage N record and adds the actual
validation experiment after manual acquisition.

| Work item | Status | Exit evidence |
|---|---|---|
| N2.1 split archive audit | Complete | 13 RGB parts and annotation tar hashed; duplicate dark `tarab` identified and excluded |
| N2.2 validation extraction | Complete | 4,840 light JPG + 4,840 dark PNG frames; train/RAW/test/real not extracted |
| N2.3 class/mark evidence | Complete | IDs 1--6 visually checked; 131,781/131,781 rows carry active mark 1 |
| N2.4 L0--L3 execution | Complete | 19,360 method-frames, four methods, four sequences, frozen D1 and tracker settings |
| N2.5 official metrics | Complete | Official TrackEval HOTA/CLEAR/Identity; output verifier closes 68,887 motor-vehicle GT |
| N2.6 closeout | Complete | Report, qualitative frames, per-sequence metrics and artifact registry frozen |

The experiment supports a low-light robustness limitation and a tracker
coverage/identity trade-off. It does not open the formal parking-occupancy
data gate and does not justify LMOT-driven parameter tuning.

## 13. Stage N-v3 emitted-box correction

Stage N-v3 is an additive offline correction. It fixes the unused-GT fallback
in the local emitted-box matcher and replaces ambiguous count means with
summed counts plus a clearly separated all-data pooled/micro primary result
and per-sequence macro diagnostic. It consumes only the 16 saved v2 detection
JSONL files and four LMOT GT files. No model, tracker, inference, training, or
TrackEval call is made. Official HOTA, DetA, AssA, IDF1, MOTA and ID-switch
results remain unchanged because they do not use the corrected local matcher.
