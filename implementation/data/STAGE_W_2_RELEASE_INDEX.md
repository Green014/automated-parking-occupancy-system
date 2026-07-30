# Stage W.2 Cross-Platform Local Source Candidate Index

Date: 2026-07-30  
Public release ready: **false**

## Current W.2 boundary

- `.gitattributes`: LF source checkout and explicit binary asset rules;
- `STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml`: machine-readable include/exclude
  source boundary;
- `STAGE_W_2_ARTIFACT_REGISTRY.yaml`: current source and release identity;
- `STAGE_W_2_CROSS_PLATFORM_RELEASE_REPORT.md`: model-free verification
  record;
- `../tests/test_stage_w_2_cross_platform_release.py`: regression contract;
- `../scripts/verify_stage_w_2_checkout.py`: current-tree and temporary
  `core.autocrlf=true` checkout verifier.

## Runtime entry points

| Entry | Purpose |
|---|---|
| `parking-run-final` | default P3 research runtime |
| `parking-compare` | Stage V.1 Classic/Detection/Fusion comparison |
| `parking-dashboard` | local Stage W Flask dashboard |
| `python scripts/run_p3_tt.py` | optional P3-TT TrackTrack experiment |

Stage W.2 does not change any algorithm or command. Its verification invokes
only `--help`, compile, Git attributes, manifest, and byte-hash checks.

## Version history

- Stage V.1 and Stage W registries: pre-hardening historical snapshots;
- Stage W.1 registry SHA-256
  `ac113b84afad1622f75230d3c34c9578b6c980fdad9adf5d079dc3cc60377a27`:
  exact pre-W.2 historical source snapshot;
- Stage W.2: current local source candidate.

No historical registry is rewritten. Existing local demonstration evidence is
hash-checked only when present and is never regenerated. The Stage W
four-frame loop remains a truth-free interface demonstration, not an accuracy
measurement or continuous-video validation.

## Source commit boundary

The source manifest includes repository release metadata, implementation
source/configs/scripts/tests, compact research evidence, the required
literature-core source and documents, and the existing Stage S/T/U.1
presentation media. It excludes model weights, datasets/downloads,
outputs/runs, environments/caches, machine-local path records, local vendor
copies, unauthorized member model/data/video assets, and temporary Git
repositories. Excluded local files are not deleted.

The finalized local selection contains 542 candidate files across eight
include categories. Thirty-two existing files matched an exclusion after
initial inclusion; ignored large/local trees are excluded without enumeration.
The `core.autocrlf=true` temporary checkout passed config identity, registry,
four CLI help and compile checks without a model or dataset.

## Release gate

`public_release_ready=false`. No top-level project licence has been selected.
The granting member name, authorization date, written record, modified
HTML/CSS public redistribution permission, and model/data/video redistribution
permission remain pending. Ultralytics licensing and member-interface
permission are independent boundaries. Do not push publicly or create a
public Release.
