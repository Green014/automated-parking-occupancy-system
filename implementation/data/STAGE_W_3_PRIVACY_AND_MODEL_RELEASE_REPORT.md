# Stage W.3 Privacy-Safe Public Source and Model Release Report

Date: 2026-07-30

## Scope

Stage W.3 prepares a privacy-safe public source candidate and two independent
future model Release assets. It does not train, fine-tune, load, or execute a
model; regenerate predictions, metrics, videos, or screenshots; alter any
Stage J–U frozen result; operate on the real Git index; connect a remote; or
publish a Release.

## W.2 historical boundary and environment counts

The W.2 manifest, registry, report, and release index remain byte-identical
historical records. W.2 recorded 341 passed and 1 skipped implementation tests
in an environment without Flask, plus 83 passed literature-core tests.

W.3 records the current validation separately. Pass/skip totals can differ
without a research-result change because:

- an installed Flask makes Stage W server tests run instead of skip;
- optional TrackEval availability controls marked tracking tests;
- W.3 adds new regression tests;
- later source-hardening tests increase collection totals.

The exact current totals are filled from the final model-free validation:

- implementation full suite: 355 passed, 3 skipped, 0 failed;
- literature-core full suite: 83 passed, 0 skipped, 0 failed;
- Stage V directed suite: 24 passed, 0 skipped, 0 failed;
- Stage W/W.1/W.2/W.3 directed suite: 38 passed, 0 skipped, 0 failed;
- W.3-only directed suite: 9 passed, 0 skipped, 0 failed.

The canonical W.3 validation interpreter is the repository-relative
`implementation/.venv_stage_o_retinexformer/Scripts/python.exe`, running
Python 3.12.13. Flask is installed, so the dashboard server tests run.
TrackEval is absent, so its three explicitly optional Stage N tests are the
only skips. The W.3 suite also contains nine release regression tests. These
environment and test-collection differences do not change a model or
experiment output, and the historical W.2 total is not rewritten to mimic the
later environment.

## Privacy and permission boundary

`PUBLIC_PERMISSION_AND_PROVENANCE.md` records only anonymous confirmation,
private evidence retention, authorization from the upstream code owner, the
project owner's anonymous attestation, the upstream repository and audited
commit, and the actual code boundary. The older private permission record
remains local and is explicitly excluded from public source.

The source resolver removes text containing a concrete Windows user home,
email address, credential-bearing RTSP URL, private-key header, or recognized
secret-token shape. It also excludes weights, datasets, outputs/runs, vendor
copies, environments, caches, local-only registries, and unconfirmed member
assets.

## License boundary

The repository-root `LICENSE` is the complete AGPL version 3 text and is
declared `text eol=lf`. `THIRD_PARTY_NOTICES.md` records the independent
Ultralytics, PyTorch/torchvision, OpenCV, Flask, other dependency, NDISPark
ODC-By-1.0, PKLot CC-BY-4.0, and adapted-interface boundaries. Unconfirmed
local assets remain excluded rather than assigned a guessed license. Both
Python package metadata files declare `AGPL-3.0-only`.

## Model assets

The exact frozen files were found by byte count and SHA-256, then copied
byte-for-byte to the ignored release-assets directory:

| Asset | Bytes | SHA-256 | Status |
|---|---:|---|---|
| D1 | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` | verified |
| E1b | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` | verified |

Same-sized files with different hashes were not selected. The source manifest
contains no `.pt`. At the time of this frozen validation, the Release URL was
still pending.

## Final validation record

- formal config current SHA-256:
  `198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0`;
- formal config temporary `core.autocrlf=true` clone SHA-256: the same value,
  with no CRLF;
- public source candidate: 554 files, 69,500,502 bytes;
- source selection: 8 include categories, 9 explicit exclude categories,
  34 included files removed by an explicit or privacy exclusion;
- privacy scan: 546 UTF-8 text files scanned, 0 findings;
- W.3 registry: 33 artifacts, all verified locally;
- clean checkout: passed; all four CLI help checks and compileall passed
  without a model or dataset;
- package builds: passed; both wheel metadata files declare
  `License-Expression: AGPL-3.0-only`;
- compileall: passed;
- `git diff --check`: passed;
- all applicable historical/current registry verifiers: passed;
- real Git index unchanged and no remote configured at validation time: passed.

## Release state

- `source_publication_ready=true`
- `model_assets_ready_for_github_release=true`
- `public_release_published=false`
- `model_training_run=false`
- `model_inference_run=false`

After successful validation, remaining manual actions are to configure an
authorized remote, push the verified source commit, create a real GitHub
Release, upload D1/E1b plus `SHA256SUMS.txt` and
`MODEL_RELEASE_METADATA.yaml`, and replace the pending Release URL with the
real URL. W.3 performs none of the remote or publication actions.

## Publication follow-up

Those manual actions were completed on 2026-07-31. The source and verified
D1/E1b assets are published as
[v1.0.0](https://github.com/Green014/automated-parking-occupancy-system/releases/tag/v1.0.0).
The release-time follow-up is additive: the `public_release_published=false`
value above remains the truthful state of the frozen W.3 validation rather
than being rewritten after the fact.
