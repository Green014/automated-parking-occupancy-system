# Stage W.1 Release Hardening Report

Date: 2026-07-30  
Status: local source-review candidate  
Public release ready: **false**  
Model training or inference performed: **false**

## Scope

Stage W.1 is a source, dependency, documentation and release-verification
correction. It does not retrain, fine-tune or execute D1, E1b, D1-LL,
YOLO-World, ByteTrack or TrackTrack. It does not regenerate or modify any
Stage J–W prediction, metric, model hash, input hash, annotated video or
dashboard demonstration.

## Frozen configuration identity correction

Stage V.1 previously recorded the formal configuration SHA-256 but classified
the config as frozen from critical parameter values alone. W.1 corrects the
contract:

- `classification: frozen` requires both the exact formal SHA-256
  `198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0`
  and all frozen critical values;
- an identical byte copy at another path remains frozen;
- a byte-only change such as an added comment is rejected unless
  `--allow-custom-config` is explicit;
- an allowed hash-only change is custom with `exact_sha256_match: false`,
  `hash_mismatch: true`, `critical_parameters_match: true` and no fabricated
  parameter difference;
- a parameter change remains rejected by default and becomes custom only
  under the same explicit opt-in;
- neither custom case may use the frozen C1/C2 method identity.

`frozen_parameters_changed` remains true for any loss of frozen identity.
`parameter_values_changed` distinguishes an actual value change from a
byte-only hash change.

## Dependency and CLI correction

`pyproject.toml` now declares:

- `dashboard = ["Flask>=3.1,<4"]`;
- `parking-compare = "parking_occupancy.stage_v_runner:main"`;
- `parking-dashboard = "parking_occupancy.stage_w_cli:main"`;
- the Stage W template/static package data.

`stage_w_requirements.txt` retains the identical Flask range as a compatibility
entry. The complete development/test installation is:

```powershell
python -m pip install -e .\literature_core
python -m pip install -e ".[integrated,dashboard,dev]"
```

The dashboard server module can be imported without Flask. A server creation
attempt then raises a clear installation error; the server test module skips
cleanly when the optional dashboard dependency is absent instead of aborting
test collection.

The Stage V README example now uses the implemented `--output-dir` option.
The documented P3, Stage V comparison, Stage W dashboard and P3-TT commands
were checked against their `argparse` schemas. `parking-run-final` retains its
existing algorithm and remains the default research runtime.

## User-facing role separation

| Entry | Role | Method boundary |
|---|---|---|
| `parking-run-final` | default research runtime | D1 → B1 → E1b/F2; E4/tracker default off |
| `parking-compare` | controlled method comparison | Classic / D1+B1 / D1+B1+E1b+F2 |
| `parking-dashboard` | local presentation/demo entry | adapter around the explicitly selected backend |
| `scripts/run_p3_tt.py` | optional tracking experiment | D1 → TrackTrack → B1 → E1b/F2 |

TrackTrack is not claimed to improve slot occupancy. The Stage W four-frame
loop remains a consumed, truth-free interface demonstration and is not an
accuracy or continuous-video validation.

## Registry version boundary

The old registries are preserved byte-for-byte and classified here as
pre-hardening historical snapshots:

| Snapshot | Registry SHA-256 |
|---|---|
| Stage V.1 | `19aec081be8e9707f0025365a136dcd6fc68a005a373ec7a4d6e1e99680bd372` |
| Stage W | `0a1daf77de33d753f5a79609146f80c5a53a382f78975b03b086b036af410bc9` |

Their registry identity and immutable local smoke evidence can still be
verified, but their old source rows are not compared with the post-W.1
worktree. The current source candidate is governed by
`STAGE_W_1_ARTIFACT_REGISTRY.yaml`.

No model output was regenerated. W.1 references and verifies the existing
local demonstration files by their prior bytes and SHA-256 values.

## Permission and public-release gate

No top-level project licence exists, and W.1 does not choose one. The user
still needs to provide:

- granting member name;
- authorization date;
- written authorization record;
- whether modified member-derived HTML/CSS may be publicly redistributed;
- whether related models, data or video may be publicly redistributed.

Ultralytics licensing and the member-derived interface permission are
independent boundaries. Resolving either one does not resolve the other.
Therefore `public_release_ready=false`; no public push, Release, model, data,
local output or member-derived source redistribution is authorized.

## Verification record

All W.1 validation is model-free: unit tests use fakes, CLI checks use
`--help`, registries hash existing files, and compile checks only parse source.

| Check | Result |
|---|---|
| `python -m pytest tests -q` | 338 passed, 3 skipped, 0 failed |
| `python -m pytest literature_core/tests -q` | 83 passed |
| Stage V targeted tests | 24 passed |
| Stage W adapter/server/W.1 targeted tests | 21 passed |
| `compileall` | passed |
| `git diff --check` | passed; line-ending conversion warnings only |
| V.1/W historical + W.1 current registries | passed |
| four CLI `--help` smokes | passed |

The three skips are the existing optional official TrackEval metric tests
because TrackEval is not installed. No test was deselected in the final runs.
