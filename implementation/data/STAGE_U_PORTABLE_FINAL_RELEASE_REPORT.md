# Stage U — Portable Final Release and Generic P3-TT Runtime

Date: 2026-07-30  
Protocol: `STAGE-U-PORTABLE-FINAL-RELEASE-20260730-01`

## Outcome

Stage U separates the generic P3-TT runtime from the frozen Stage T experiment,
adds a repository-level submission entry, classifies historical registries,
and constructs a clean portable package without local outputs, datasets,
weights, or virtual environments.

No model inference, training, fine-tuning, threshold selection, or demo
rendering is performed.

## Runtime separation

The frozen `stage_t_tracktrack.run_stage_t_variant` and Stage T CLI remain
unchanged. They continue to enforce the exact VIRAT, D1, and E1b hashes used by
the consumed-development TT0/TT1 diagnostic.

The additive generic runtime is implemented in:

- `src/parking_occupancy/p3_tt_runtime.py`
- `scripts/run_p3_tt.py`

It accepts any local video and polygon map, forces E4 off and TrackTrack on,
creates fresh per-run state, reuses the frozen P3-TT configuration and existing
TrackTrack adapter, and writes the standard occupancy, event, detection,
trajectory, video, summary, and runtime records. It does not reproduce or
reimplement TrackTrack.

Custom D1 or E1b files are allowed, but metadata records
`custom_weights=true` and
`stage_t_result_comparison_applicable=false`.

## Default-system preservation

The Stage S default remains:

`D1 -> B1 -> F2`, E4 off, tracker none.

Stage S registry and demo hashes are rechecked rather than rewritten.

## Registry model

Historical artifact registries are classified from their actual path bindings:

- portable historical registries contain repository-relative bindings whose
  artifacts are included in the submission candidates;
- local-only historical registries bind local outputs, datasets, weights,
  machine-specific absolute paths, or other non-submission artifacts.

The original registry contents are not altered. The Stage T full registry is
classified as `local_full_artifact_registry` / local-only and remains on the
local machine. An additive nested `.gitignore` excludes it and the other
local-only historical registries from the portable candidate set.

The Stage U submission registry covers every final submission candidate except
its own self-hash. It uses repository-relative paths and is verified again
inside the clean package.

## Submission and clean-package audit

Final counts, byte totals, registry hash, clean-package commands, pass/skip
counts, and demo codec checks are recorded in:

- `data/stage_u/STAGE_U_SUBMISSION_AUDIT.json`
- `data/stage_u/STAGE_U_HISTORICAL_REGISTRY_CLASSIFICATION.json`
- `data/stage_u/STAGE_U_CLEAN_PACKAGE_VERIFICATION.json`
- `data/stage_u/STAGE_U_DEMO_COMPATIBILITY_AUDIT.json`
- `data/stage_u/STAGE_U_SUBMISSION_ARTIFACT_REGISTRY_20260730.yaml`

The clean package is built only from the computed candidate list. Optional
TrackEval tests are run separately against the pinned external source; tests
that require intentionally omitted local-only historical registries are
reported explicitly rather than hidden.

The verified clean-package run copied the complete 586-file candidate set; its
final byte total is recorded in the submission audit and clean-verification
JSON. It contained no outputs, datasets, model weights, or virtual
environments. Its results were:

| Check | Result |
|---|---:|
| Stage U targeted mock/stub tests | 11 passed |
| Selected implementation tests in clean package | 278 passed, 1 skipped |
| `literature_core` tests | 83 passed |
| `compileall` | passed |
| Portable registry, complete coverage | passed |

Eight tests that explicitly open local-only historical registries, frozen
runtime outputs, weights, or datasets are listed by node ID in the clean
verification JSON and deselected only in the intentionally stripped package.
They remain present and runnable in the full local worktree. The three official
TrackEval tests are not part of that package invocation.

The fixed external TrackEval checkout was independently rechecked at commit
`12c8791b303e0a0b50f753af204249e622d0281a` and tree
`ce583e59d96b33aad5f8f62149d96bc6bc4d8f96`; all three official TrackEval
tests passed. The optional BURST import still reports missing `pycocotools`,
which does not affect the MOT HOTA/CLEAR/Identity tests used here.

Historical registry classification is:

- portable: Stage R and Stage S;
- local-only: Stage M, Stage N, Stage N-v2, Stage N-v3, Stage O, Stage P,
  Stage Q, Stage Q-v2, and Stage T.

All 11 registry files remain unmodified. The local-only files stay on this
machine and are excluded only from the submission candidate set.

## Demo compatibility

The Stage S and Stage T frozen videos and images are not modified. Their
existing FMP4 encoding is retained. FMP4 can require an additional H.264
presentation copy for some PowerPoint environments, but no new dependency is
downloaded and the frozen originals are not replaced.

| Demo | SHA-256 | Decode result |
|---|---|---|
| Stage S `demo_main.mp4` | `f4e9e59b5bcef1b51f2e94b8443c5f22a69ca850bfc77f5c9b94a1bf947ac608` | 500/500 frames, 1280x720, 10 FPS, 50.0 s, FMP4 |
| Stage T `demo_tracktrack_optional.mp4` | `b5dfdeb850acdd0a87072a9c48fda44dd5e13725fb7f0e428cfc6164b4d24c1f` | 450/450 frames, 1920x1080, 29.97 FPS, 15.015015 s, FMP4 |

Here the duration values are container/video durations; no experiment timing
or inference-speed claim is derived from them. Stage S metadata confirms
`raw_state`, E4 off, and tracker none. Stage T metadata confirms the optional
title, consumed-development role, E4 off, and no occupancy-improvement claim.
No existing `ffmpeg` executable was available, so no H.264 presentation copy
was generated.

## Claim boundary

- Stage S default evidence: unchanged.
- Stage T TT0/TT1: consumed-development diagnostic.
- Generic P3-TT output: user-supplied local run, not a Stage T comparison.
- TrackTrack occupancy improvement: not claimed.
- Deployment readiness: not claimed.
