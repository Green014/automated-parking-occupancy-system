# Stage V Multi-Backend Occupancy Report

Date: 2026-07-30  
Status: implementation and local smoke complete; public release gated

## Delivered system

All modes implement:

```text
OccupancyBackend.process_frame(frame, slots, frame_index, timestamp_s)
    -> FrameOccupancyResult
```

`FrameOccupancyResult` contains the frame index, timestamp, one
`SlotOccupancyState` per configured slot, evidence score/source, vehicle
detections, optional track IDs, stage timing, and warnings. The runner rejects
duplicate, missing, or extra slot decisions. Visualization independently
checks that rendered slot count equals configured slot count.

| Mode | ID | Input evidence | Output decision |
|---|---|---|---|
| Classic | C0 | BGR frame + polygon ROIs | Clean-room OpenCV foreground ratio versus an explicitly uncalibrated threshold |
| Detection | C1 | Frozen D1 detections + B1 | One-to-one polygon coverage; all unmatched slots are explicitly vacant |
| Fusion | C2 | D1+B1 plus E1b review of detector-negative slots | Frozen F2 asymmetric decision; clear detector evidence cannot be overwritten |

Rectangle slot definitions are converted once to the canonical polygon
representation. The reference repository's `frame_width`/`frame_height` and
`spaces` polygon schema is also accepted without creating a second runtime slot
model.

## Runtime and outputs

The entry point is:

```powershell
python scripts\run_stage_v_multimode_demo.py `
  --input <video-or-image-directory> `
  --slots <slots.json> `
  --mode classic|detection|fusion|compare `
  --output-dir <new-output-directory>
```

Detection/Fusion require the exact frozen artifacts. `compare` processes the
same frames and slot map through C0/C1/C2 and shares one D1 result per frame.
Every output root must be new.

Per mode:

- `annotated.mp4` or `annotated_images/`;
- `occupancy.csv`;
- `events.csv`;
- `detections.jsonl`;
- `summary.json`;
- `metrics.json`;
- `runtime_metadata.json`;
- `configuration_snapshot.yaml`.

Comparison additionally writes `comparison.json`, `method_metrics.csv`, and
`runtime_comparison.csv`. A side-by-side video is optional and was not
generated.

## Optional modules

- E4 remains default-off and requires explicit `--temporal` on a single
  continuous fusion video. It is rejected for static/image-directory input.
- ByteTrack/TrackTrack remain default `none`. They can be explicitly selected
  for a single Detection/Fusion run. The controlled C0/C1/C2 comparison
  rejects tracking so component definitions stay unchanged.
- The real Stage V smoke used neither E4 nor tracking.

## Real local smoke

Run: `outputs/stage_v_multimode_smoke_20260730_v3`  
Input role: already-consumed Stage O interface video, demonstration only  
Device: local CPU  
Input: 4 continuous frames, 1280×720, 5 configured slots  
Truth: none  
Models: exact frozen D1 and E1b hashes  
Parameter changes: none  
Output coverage: 20/20 unique frame-slot rows in every mode  
Video decode: 4/4 frames in every mode

| Mode | Occupied row count | Mean attributed frame latency | Steady-state FPS proxy |
|---|---:|---:|---:|
| C0 Classic | 8/20 | 31.576 ms | 33.379 |
| C1 Detection | 4/20 | 27.328 ms | 34.508 |
| C2 Fusion | 8/20 | 55.824 ms | 16.543 |

Component mean timings from this four-frame CPU smoke:

- Classic preprocessing: 12.492 ms.
- Shared D1 detection attribution: 15.351 ms for both C1 and C2.
- B1 mapping: 0.471 ms (C1) and 0.421 ms (C2).
- E1b classification: 28.214 ms in C2.
- F2 fusion: 0.026 ms in C2.
- Render/write: 9.708–11.110 ms.

These values are a short local runtime smoke, not a stable hardware benchmark.
Model construction and one explicit inference warm-up are outside reported
steady-state timing. C1 and C2 share cached D1 detections, while runtime
comparison attributes the same D1 cost to both methods.

No Macro F1, recall, false-free, false-occupied, transition latency, or tracking
improvement is reported because the smoke has no key-aligned reliable truth.
The different occupied counts are observations, not accuracy rankings.

## Method recommendation

- Default research occupancy mode: **Fusion (C2)**, consistent with the frozen
  Stage R/S decision, because F2 improves the frozen Stage Q-v2 component
  attribution while preserving detector authority.
- Engineering/teaching baseline: **Classic (C0)**. It is interpretable and
  inexpensive but its new threshold has not been calibrated, and fixed image
  statistics are expected to be sensitive to lighting, shadows, viewpoint,
  paint, and surface texture.
- Detection (C1) is the controlled ablation that exposes the contribution of
  E1b/F2.
- Tracking was not used in the Stage V comparison and no occupancy improvement
  is attributed to it.

## Evidence classification

Quantitative conclusions retained from frozen work:

- Stage R post-hoc component attribution supports D1+B1+F2 over B1 alone on
  the already frozen Stage Q-v2 outputs.
- Stage T consumed-development evidence found no TrackTrack slot-occupancy
  benefit and a runtime cost.

Stage V observations:

- Interface, schema, coverage, cache, and rendering invariants are test-backed.
- The real four-frame run demonstrates execution and output production only.
- No new accuracy, robustness, temporal, or tracking claim is created.

## Validation status

- `tests/test_stage_v_multimode.py`: 12 passed.
- `literature_core/tests`: 83 passed.
- Main suite excluding the three frozen Stage U exact-candidate-set snapshot
  assertions: 302 passed, 3 optional tests skipped, 3 deselected.
- Unfiltered main suite: 302 passed, 3 optional tests skipped, 3 failed. All
  three failures are frozen Stage U portable-release snapshot guards that
  correctly detect the nine newly added Stage V files; they are not execution,
  schema, model, or result failures. Stage U artifacts were not regenerated.
- Full `compileall` and `git diff --check`: passed. The latter emitted only
  pre-existing Windows line-ending conversion warnings.
- Frozen Stage R and Stage S registries still verify exactly.

## Release recommendation

Do not publish Stage V yet. The code is suitable for local review and eventual
merge, but the public repository has no top-level licence and the Ultralytics
AGPL/enterprise boundary has not been resolved in repository licensing
materials. The member reference also remains unlicensed, although Stage V
avoids direct reuse. Add the chosen project licence and third-party notices,
then rerun the release audit before public GitHub publication.
