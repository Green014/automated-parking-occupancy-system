# Stage V Reproduction Guide

Date: 2026-07-30

## Preconditions

1. Install the project with the `integrated` and `dev` extras and install the
   sibling `literature_core` package.
2. Supply D1 and E1b locally; Stage V never downloads them.
3. Verify the exact identities:

   - D1: 6,255,409 bytes,
     `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`
   - E1b: 8,045,704 bytes,
     `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3`

4. Use a new output directory. Existing output roots are rejected.
5. Keep E4 and tracking off unless the run is explicitly designed for those
   optional components.

If Ultralytics cannot write its user settings directory, set
`YOLO_CONFIG_DIR` to a writable local runtime directory outside the release
candidate set.

## Single modes

Classic requires no model:

```powershell
python scripts\run_stage_v_multimode_demo.py `
  --input <video-or-image-directory> `
  --slots <slots.json> `
  --mode classic `
  --output-dir <new-output-directory>
```

Detection:

```powershell
python scripts\run_stage_v_multimode_demo.py `
  --input <video-or-image-directory> `
  --slots <slots.json> `
  --mode detection `
  --d1-weights <frozen-D1-best.pt> `
  --output-dir <new-output-directory>
```

Fusion:

```powershell
python scripts\run_stage_v_multimode_demo.py `
  --input <video-or-image-directory> `
  --slots <slots.json> `
  --mode fusion `
  --d1-weights <frozen-D1-best.pt> `
  --e1b-checkpoint <frozen-E1b-best.pt> `
  --output-dir <new-output-directory>
```

Use `--temporal` only for a genuinely continuous fusion video. Use
`--tracker bytetrack|tracktrack` only when track IDs are part of the declared
run. Both defaults are off.

## Controlled comparison

```powershell
python scripts\run_stage_v_multimode_demo.py `
  --input <video-or-image-directory> `
  --slots <slots.json> `
  --mode compare `
  --d1-weights <frozen-D1-best.pt> `
  --e1b-checkpoint <frozen-E1b-best.pt> `
  --output-dir <new-output-directory>
```

`compare` locks C0/C1/C2 to the same input frames and slot polygons. C1 and C2
share D1 cache entries. Tracking and E4 are rejected in this comparison.

## Truth and metric boundary

Add `--truth <slot-state.csv> --truth-role development|test|consumed-demonstration`
only when truth has exactly one binary state for every
`video_id + frame_index + slot_id` prediction key. A mismatch fails.

Stage V reports static slot metrics only. It does not report transition latency
or tracking improvement from image directories, sparse PKLot montages, or
short demonstration material.

## Tests and audit

```powershell
python -m pytest tests/test_stage_v_multimode.py -q
python -m pytest tests -q
python -m pytest literature_core/tests -q
python -m compileall -q src scripts tests literature_core/src `
  literature_core/scripts literature_core/tests
git diff --check
```

Before public release:

1. verify the Stage V artifact registry;
2. scan tracked/unignored candidates for absolute machine paths, credentials,
   datasets, weights, virtual environments, and runtime outputs;
3. add a top-level project licence and required third-party notices;
4. confirm no reference-repository code or checkpoint entered the candidate
   set;
5. obtain user approval before pushing.
