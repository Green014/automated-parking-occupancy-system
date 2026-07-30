# Stage I-v2 preservation audit

Audit time: `2026-07-27T21:55:11.2781183+08:00`

This preflight records the state that must remain recoverable while Stage I-v2
corrects the detector-comparison methodology. It does not promote the corrected
work to a new untouched test. NDISPark test remains consumed and any new use of
its counting labels is post-hoc sensitivity analysis.

## Git and source state

- Branch: `codex/part1-dataset-alignment`
- HEAD: `2a13f49`
- Detached HEAD: no
- Porcelain entries at audit time: 84 total (19 tracked changes, 65 untracked)
- Action taken: no reset, checkout, staging, commit, or cleanup
- Preservation rule: all pre-existing tracked and untracked work remains in
  place; Stage I-v2 files use new names and protocol identifiers.

The source worktree contains the accumulated Stage A-I implementation and
documentation changes. These are not frozen experimental outputs. They must be
reviewed and committed in coherent batches rather than omitted because they are
currently untracked.

## Ignored and external artifacts

The following items live under the external artifact root rather than the Git
worktree. Paths are expressed relative to that root so no user-specific data
path is introduced into source configuration.

| Role | Relative path | Files | Bytes | Preservation status |
|---|---|---:|---:|---|
| Stage I v1 development comparison | `implementation/outputs/detector_comparison_stage_i_20260727_v1` | 32 | 4,645,256 | frozen, read-only |
| Stage I v1 failed count preflight | `implementation/outputs/detector_count_test_stage_i_20260727_v1` | 1 | 1,198 | frozen negative evidence |
| Stage I v1 count evaluation retry | `implementation/outputs/detector_count_test_stage_i_20260727_v2` | 14 | 762,707 | frozen, read-only |
| Stage I v1 qualitative outputs | `implementation/outputs/detector_qualitative_stage_i_20260727_v1` | 14 | 24,118,308 | frozen, read-only |
| D1 formal training | `implementation/outputs/d1_ndispark_formal_20260727_v1` | 37 | 139,626,147 | frozen, no retraining |

External model weights:

| Method | Relative path | Bytes | SHA-256 |
|---|---|---:|---|
| D0 | `implementation/yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| D1 | `implementation/outputs/d1_ndispark_formal_20260727_v1/weights/best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| D2 | `implementation/literature_core/models/yolov8s-worldv2.pt` | 25,923,032 | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` |

These weights are inputs, not repository content. Runtime locations must continue
to be supplied by CLI arguments or environment variables.

## Frozen verification evidence

The verification tools were rerun into new output filenames. The JSON bytes are
identical to the earlier verification records, which confirms that the expected
and actual artifact hashes did not change between the v1 and v2 audit runs.

| Verification | Checks | Earlier SHA-256 | Audit-rerun SHA-256 | Result |
|---|---:|---|---|---|
| D1 formal training | 14/14 | `9f1ef3f655c454dc1391377634b703bdc996fb287ce1ee704862d3689738fd45` | `9f1ef3f655c454dc1391377634b703bdc996fb287ce1ee704862d3689738fd45` | pass, byte-identical |
| Stage I v1 | 24/24 | `0598fa9844e41b66ae5e8061d57bbba6a2eb43d0420a27d23b514d69cdf60f22` | `0598fa9844e41b66ae5e8061d57bbba6a2eb43d0420a27d23b514d69cdf60f22` | pass, byte-identical |

New audit evidence:

- `implementation/outputs/d1_formal_training_verification_20260727_v2.json`
- `implementation/outputs/stage_i_artifact_verification_20260727_v2.json`
- `implementation/outputs/compileall_stage_i_v2_phase_a_20260727_v1/`

All are ignored generated artifacts and do not replace the v1 evidence.

## Phase A validation

| Check | Result |
|---|---|
| implementation tests | 87 passed |
| literature_core tests | 82 passed |
| compileall | passed with `PYTHONPYCACHEPREFIX` directed to a new ignored output directory |
| `git diff --check` | passed; existing line-ending warnings only |
| D1 artifact verification | 14/14 passed |
| Stage I artifact verification | 24/24 passed |

An initial literature test invocation used the repository root and could not
resolve the literature package's `scripts` import; rerunning from
`implementation/literature_core` passed. An initial compileall invocation could
not write worktree bytecode caches under the sandbox; redirecting bytecode to a
new ignored output directory passed. These environment-level first attempts are
retained rather than hidden.

## Recommended future commit batches

No commit was created during this audit. When the user chooses to commit, the
current changes should be reviewed and separated approximately as follows:

1. Metric correction, baseline registry/closure, unified output contract, and
   their tests and documentation.
2. Part I dataset audit, frozen data protocol, preprocessing tooling, tests, and
   dataset cards.
3. D1 training protocol, smoke/formal-run provenance, artifact verifiers, and
   evaluation tooling.
4. Stage I v1 documentation/provenance records and the Stage I-v2 corrected
   protocol, implementation, tests, and result boundary documentation.

Ignored raw data, weights, checkpoints, videos, and generated output directories
must not be added to any batch.
