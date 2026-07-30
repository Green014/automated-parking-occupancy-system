# Stage W.2 Cross-Platform Release Packaging Report

Date: 2026-07-30  
Status: current local source candidate  
Public release ready: **false**  
Model training run: **false**  
Model inference run: **false**

## Scope

Stage W.2 changes source packaging, line-ending policy, tests, registry
boundaries and documentation only. It does not train, fine-tune or load a
model; regenerate predictions, metrics, screenshots or video; or alter any
Stage J-W.1 experiment value, model hash, input hash or conclusion.

## Cross-platform line-ending contract

The root `.gitattributes` forces LF for Python, TOML, YAML, JSON/JSONL, CSV,
Markdown, text, HTML, CSS, INI, CFG and SVG source. Model formats, video,
images, PDFs, Office documents and archives are binary.

The formal config
`implementation/configs/p3_stage_r_recommended_default_20260729.yaml` contains
no CRLF and remains:

`198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0`

Frozen Stage V.1 identity is not relaxed. Both the exact byte hash and frozen
critical parameters are still required.

## Source commit manifest

`STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml` is an include-then-exclude contract.
It includes source, portable configs, scripts/tests, compact research
evidence, literature-core materials and existing Stage S/T/U.1 presentation
media. It excludes weights, datasets/downloads, outputs/runs,
environments/caches, machine-local path records, local vendor copies,
unauthorized member model/data/video assets and temporary Git repositories.
Exclusion never deletes a local file.

## Version boundary

Stage W.1 is pinned byte-for-byte as the pre-W.2 historical source snapshot
with registry SHA-256
`ac113b84afad1622f75230d3c34c9578b6c980fdad9adf5d079dc3cc60377a27`.
Stage W.2 is the current local source candidate. Earlier V.1/W/W.1 registries
are not rewritten.

Existing Stage W demonstration evidence is referenced through its saved
hashes only. The four-frame loop is a consumed, truth-free interface
demonstration and is not accuracy evidence or continuous-video validation.

## Temporary Windows checkout method

`scripts/verify_stage_w_2_checkout.py` copies only manifest candidates to the
system temporary directory, initializes and commits only a temporary source
repository, and clones it using `git -c core.autocrlf=true clone`. The clone
then verifies the formal config LF/hash, W.2 registry, source manifest, four
CLI `--help` commands and `compileall`. It requires no models or datasets and
connects no public remote.

## Verification record

All checks were model-free. Unit tests used fakes or saved compact evidence;
CLI checks used `--help`; registries only read and hash existing files.

| Check | Result |
|---|---|
| `python -m pytest tests -q` | 341 passed, 1 skipped, 0 failed |
| `python -m pytest literature_core/tests -q` | 83 passed, 0 failed |
| Stage V targeted | 24 passed |
| Stage W/W.1/W.2 targeted | 21 passed, 1 skipped |
| `compileall` | passed |
| `git diff --check` | passed |
| V.1/W/W.1 historical and W.2 current registries | passed |
| four CLI `--help` smokes | passed |
| temporary `core.autocrlf=true` checkout | passed |

The single skip is the Stage W Flask server test because the already-installed
environment does not contain the optional dashboard dependency. Module
collection and all CLI help remain safe without Flask; running the local
Dashboard server still requires `Flask>=3.1,<4`.

The current manifest resolves 542 candidate files across eight include
categories. Thirty-two existing files that first matched an include rule are
removed by the eight path-exclusion categories plus the machine-local
absolute-path content rule. Ignored weights, datasets, outputs/runs,
environments and caches stay outside enumeration and are not deleted.

The isolated checkout copied the same 542 candidates, committed them only in a
temporary source repository and cloned with `core.autocrlf=true`. The clone:

- retained the exact formal config SHA-256 above with no CRLF;
- verified all 35 required W.2 registry artifacts, with the eight ignored
  Stage W demonstration records correctly reported unavailable;
- passed all four CLI help checks and `compileall`;
- contained no model weights, outputs or runs;
- connected no public remote;
- was removed after success.

## Permission and distribution gate

No top-level project licence is present and W.2 does not choose one. The user
must still provide the granting member name, authorization date, written
record, modified HTML/CSS public redistribution decision, and related
model/data/video redistribution decision. Ultralytics licensing and
member-interface permission remain separate unresolved boundaries.

Therefore `public_release_ready=false`. No real repository staging, commit,
push, public Release, model, dataset, output or unauthorized member asset
distribution is authorized by Stage W.2.
