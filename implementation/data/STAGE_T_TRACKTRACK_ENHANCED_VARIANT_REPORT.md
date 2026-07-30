# Stage T — Optional TrackTrack-enhanced Variant

Date: 2026-07-29  
Protocol: `STAGE-T-OPTIONAL-TRACKTRACK-VARIANT-20260729-01`  
Status: consumed-development diagnostic complete; formal occupancy improvement blocked

## Scope

Stage T adds a separately invoked P3-TT variant:

`D1 detection -> TrackTrack identity association -> B1 one-to-one polygon mapping -> E1b/F2 fusion -> occupancy + track IDs`

It does not replace the Stage S default. E4 is disabled in both TT0 and TT1 so
the controlled comparison isolates the tracker path. No detector, classifier,
fusion, mapping, temporal, or tracker threshold was changed. TrackTrack uses the
unchanged Stage M freeze
`configs/tracktrack_stage_m_frozen_20260728.yaml`, SHA-256
`54a158728a3dd41b523c7d9054fa0e187548075f563adb71c5db72e797328f37`.

The explicit configuration is `configs/p3_tt_tracktrack_optional_20260729.yaml`.
It can be invoked without changing the default release:

```powershell
python -m parking_occupancy.stage_t_cli `
  --input <continuous-video> `
  --slots <slot-polygons.json> `
  --d1-weights <d1-best.pt> `
  --e1b-checkpoint <e1b-best.pt> `
  --source-id <video-id> `
  --output-dir <new-output-directory>
```

The Stage S `parking-run-final` command remains D1 + B1 + F2 with E4 off and
tracker `none`.

## Reused and added implementation

Reused unchanged:

- `stage_m_tracking.UltralyticsSequenceAdapter` and its source-reset semantics
- Stage M frozen TrackTrack YAML and Ultralytics TrackTrack registration
- `integrated_runner.run_integrated_video`
- D1 detector settings, B1 one-to-one mapping, E1b/F2 fusion, event schema, and
  visualization

Added:

- explicit P3-TT configuration and CLI module
- a post-run `tracks.jsonl` contract with per-frame track IDs, boxes, optional
  one-to-one slot reference, observation index, gap, short-gap reacquisition,
  and expiry audit fields
- consumed-development input conversion and TT0/TT1 comparison scripts
- optional TrackTrack demo renderer
- Stage T tests and independent artifact registry

The ordinary P3 and P3-TT paths do not share live state. Source changes reset
the adapter, frame counter, event state, and tracker model. Track IDs are
metadata carried through B1; the unit tests prove that adding an ID to an
otherwise identical detection does not itself change the slot state.

## Controlled experiment

Input: `VIRAT_S_050202_10_002159_002233`, 1,974 continuous source frames, one
project-reviewed departure slot, one truth transition. This sequence was
already consumed for development and is not an untouched test. TT0 and TT1 use
the same D1 weights, E1b checkpoint, polygon, confidence/NMS, B1 coverage, and
F2 threshold.

| Metric | TT0: D1+B1+F2 | TT1: D1+TrackTrack+B1+F2 | TT1-TT0 |
|---|---:|---:|---:|
| Macro F1 | 0.456797 | 0.456797 | 0.000000 |
| Occupied recall | 1.000000 | 1.000000 | 0.000000 |
| Vacant recall | 0.000000 | 0.000000 | 0.000000 |
| False-free rate | 0.000000 | 0.000000 | 0.000000 |
| False-occupied rate | 1.000000 | 1.000000 | 0.000000 |
| Accuracy | 0.840932 | 0.840932 | 0.000000 |
| Balanced accuracy | 0.500000 | 0.500000 | 0.000000 |
| Predicted state changes | 0 | 0 | 0 |
| Event rows | 1 | 1 | 0 |
| Median-frame FPS proxy | 33.521 | 14.215 | -19.306 |

Confusion counts for both variants are TP=1,660, TN=0, FP=314, FN=0. The
occupied-heavy development clip makes accuracy 0.840932 look acceptable, but
the class-aware metrics expose complete failure on the vacant portion. Both
variants remain occupied after the vehicle leaves. Their only event is the
initial vacant-to-occupied state at frame 0; neither emits the truth departure
at frame 1660. Zero state changes therefore indicates a stuck prediction, not
useful temporal stability.

TrackTrack changes the detection stream into 23,426 tracked observations across
18 source-track IDs. The descriptive audit finds 17 short-gap reacquisitions,
no same-ID reappearance beyond the 30-frame buffer, and a maximum observation
gap of 11 frames. VIRAT 0502 has no identity truth, so these counts demonstrate
logging and continuity mechanics only; they do not measure ID correctness.

Runtime is descriptive for this local RTX 3060 Laptop GPU run. The reported
“steady-state FPS” is explicitly an inverse-median-frame-time proxy, not a
separately timed warmup-stripped benchmark. TrackTrack increases median
end-to-end frame time from 29.832 ms to 70.348 ms and reduces that proxy from
33.521 to 14.215 FPS. The mean-frame FPS values are 31.287 and 12.022.

## Existing LMOT evidence boundary

No LMOT inference or metric recomputation is performed in Stage T. Existing
Stage N-v2 official TrackEval results are cited only as an MOT diagnostic:

| LMOT condition | Backend | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---:|---:|---:|---:|---:|---:|
| well-lit | ByteTrack L0 | 26.613 | 23.422 | 30.582 | 34.147 | 22.039 | 481 |
| well-lit | TrackTrack L1 | 22.940 | 17.524 | 30.207 | 29.840 | 20.635 | 147 |
| low-light | ByteTrack L2 | 6.028 | 2.053 | 17.894 | 4.442 | 2.114 | 39 |
| low-light | TrackTrack L3 | 3.454 | 0.650 | 18.553 | 1.552 | 0.814 | 7 |

The frozen TrackTrack setting reduced ID switches but emitted a more
conservative detection set and had lower HOTA/DetA. LMOT contains no parking
slot polygons or occupancy truth, so it cannot establish parking-occupancy
improvement.

## Optional demo

`data/stage_t/demo/demo_tracktrack_optional.mp4` is a separate 15.015-second
clip from TT1 frames 1450–1899. It decodes to all 450 expected 1920 x 1080
frames at 29.97 FPS using FMP4 decoding. The title is exactly
“Optional TrackTrack-enhanced variant.” It shows the existing TrackTrack ID
labels and the B1 slot polygon while keeping an explicit
`consumed-development` / `E4 OFF` / `not Stage S default` banner.

The clip is not appended to or presented as the Stage S default-system demo.
Its rendering reused the completed TT1 annotated output and ran no additional
model inference.

## Claim classification

- Formal: no new formal test is created in Stage T.
- Post-hoc: Stage S rendering and Stage R component attribution remain post-hoc
  uses of frozen Stage Q-v2 outputs.
- Development diagnostic: TT0/TT1 on consumed VIRAT 0502 and the Stage T demo.
- Existing tracking diagnostic: Stage N-v2 LMOT TrackEval metrics; not an
  occupancy evaluation.
- Blocked: any formal claim that TrackTrack improves slot-level occupancy.

The blocked conclusion can only be revisited with a new genuinely continuous
parking video and independent frame-aligned per-slot truth. No such data are
currently available, and Stage T does not download any.

## Validation record

Successful checks:

```powershell
python -m pytest tests/test_stage_t_tracktrack.py `
  tests/test_stage_t_demo.py tests/test_stage_m_tracking.py `
  tests/test_integrated_runner.py -q
# 22 passed; 15 are Stage T-specific

$env:PYTHONPATH='<pinned TrackEval source>;literature_core/src'
python -m pytest tests -q
# 279 passed

python -m pytest literature_core/tests -q
# 83 passed

python -m pytest tests/test_stage_n_lmot.py -k official_trackeval -q
# 3 passed

python -m compileall -q implementation/src implementation/scripts `
  implementation/tests implementation/literature_core/src `
  implementation/literature_core/scripts implementation/literature_core/tests

git diff --check
```

The final in-memory submission audit contains 574 candidate files totaling
62,555,325 bytes. It reports no model weights, datasets, virtual environments,
or runtime `outputs`. The intentional Stage S and Stage T MP4/PNG presentation
artifacts account for 51,708,902 bytes. `git diff --check` exits successfully
and reports only Git LF-to-CRLF conversion warnings.

The Stage S artifact registry is re-verified after Stage T, so the optional
variant has not changed the frozen Stage S default package.

## Conclusion

TrackTrack is useful here as an optional identity-output mechanism: P3-TT emits
track IDs and auditable short-gap continuity. On the only permitted continuous
slot-truth diagnostic it provides zero occupancy benefit and a substantial
runtime cost. It therefore remains separate from the final default
D1 + B1 + F2 occupancy pipeline.
