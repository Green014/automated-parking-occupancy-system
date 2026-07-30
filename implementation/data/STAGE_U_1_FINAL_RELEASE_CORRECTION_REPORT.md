# Stage U.1 — Final Release Correction

Date: 2026-07-30  
Protocol: `STAGE-U.1-FINAL-RELEASE-CORRECTION-20260730-01`

## Scope

Stage U.1 corrects publication metadata and packaging only. It runs no model
inference, training, fine-tuning, threshold selection, or new experiment.
Stage S/T frozen results and original demonstrations remain unchanged.

## Stable evidence chain

The release uses a one-way, two-phase publication protocol:

1. generate the clean-package verification report from the current candidate
   snapshot;
2. write the final audit JSON/CSV with
   `SELF_REFERENCE_EXCLUDED` for the audit JSON, candidate CSV, and portable
   registry hashes;
3. write the portable registry, which records the actual audit JSON/CSV hashes
   but omits its own hash;
4. converge byte sizes and verify saved JSON, CSV, and registry against the
   current files;
5. rerun the clean-package verifier with `--no-record`, so the final candidate
   tree is tested without changing any source artifact.

The phase-1 clean report labels its byte count as a pre-finalization snapshot.
The authoritative final candidate count and byte total are the values from the
final audit and phase-2 `--no-record` run.

## Optional TrackEval boundary

The three official TrackEval tests receive the `trackeval` pytest marker at
collection time. A standard `.[integrated,dev]` environment without TrackEval
skips them with an explicit reason. Installing `.[trackeval]` enables the same
three tests; their metric implementation and frozen Stage N results are not
modified.

Tests that require deliberately omitted local historical registries, outputs,
weights, or datasets are skipped only when
`STAGE_U_PORTABLE_PACKAGE=1`. They remain active in the complete local
worktree.

## TrackTrack presentation clarification

The original
`implementation/data/stage_t/demo/demo_tracktrack_optional.mp4` is unchanged.
The additive Stage U.1 copy is a post-hoc explanatory rendering of that frozen
video, not a new experiment:

- yellow boxes are frozen vehicle detections with TrackTrack IDs;
- the red/green polygon is the predicted state of one evaluated parking slot;
- only one slot is evaluated;
- other visible parking positions are not evaluated;
- from source frame 1660, it labels the known
  `truth=vacant, prediction=occupied` false-occupied failure.

No local H.264 encoder was usable. The presentation copy therefore remains
FMP4; no dependency was downloaded.

## Model assets

D1 and E1b filenames, sizes, SHA-256 values, and acquisition boundaries are
recorded in `STAGE_U_1_MODEL_ASSETS.md`. The weights are not present in the
portable package.

## Validation classification

| Validation | Result |
|---|---:|
| Stage U/U.1 targeted tests | 17 passed |
| Standard implementation, `.[integrated,dev]`, no TrackEval | 293 passed, 3 optional skipped |
| Official TrackEval with pinned source | 3 passed |
| Standard implementation inside portable package | 284 passed, 12 skipped |
| `literature_core` | 83 passed |
| `compileall` | passed |
| Portable registry and saved audit/CSV chain | passed |
| Presentation video decode | 450/450 frames |

The 12 portable-package skips are explicit: three optional TrackEval tests,
eight tests requiring omitted local-only artifacts, and one Stage Q-v2
registry check whose frozen registry is intentionally absent from the portable
package.

- Executed: portable-registry verification, clean-package verification,
  external-ZIP extraction verification, demo decode, and frozen-hash checks.
- Optional and executed separately with the pinned source: three official
  TrackEval tests.
- Not executed: model inference, training, fine-tuning, threshold adjustment,
  new occupancy evaluation, and new TrackTrack experiment.
- Claim retained: no TrackTrack slot-level occupancy improvement is claimed.

Exact commands, pass/skip counts, final candidate totals, ZIP path, and ZIP
SHA-256 are reported by the final Stage U.1 handoff and the machine-readable
release artifacts.
