# Stage W.3 Release Index

Date: 2026-07-30

Status: privacy-safe public source and model assets prepared locally. This
index belongs to the verified source commit; no remote push, upload, or public
release was asserted by that frozen prepublication snapshot.

## Current release boundary

- Public source manifest:
  `STAGE_W_3_PUBLIC_SOURCE_MANIFEST.yaml`
- Model release manifest:
  `STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml`
- Artifact registry:
  `STAGE_W_3_ARTIFACT_REGISTRY.yaml`
- Release report:
  `STAGE_W_3_PRIVACY_AND_MODEL_RELEASE_REPORT.md`
- Anonymous public permission record:
  `PUBLIC_PERMISSION_AND_PROVENANCE.md`
- Model cards: `MODEL_CARD_D1.md`, `MODEL_CARD_E1B.md`
- Project license: repository-root `LICENSE` (AGPL-3.0-only)
- Third-party attribution: repository-root `THIRD_PARTY_NOTICES.md`

W.2 is the exact pre-W.3 historical source snapshot. Its registry and recorded
test totals are identity-checked rather than compared with the later live
worktree.

## User-facing entry points

| Role | Entry point | Boundary |
|---|---|---|
| Default research system | `parking-run-final` | D1 → B1 → E1b/F2; E4 off, tracker none |
| Controlled method comparison | `parking-compare` | Classic / Detection / Fusion |
| Local interface demonstration | `parking-dashboard` | Stage W adapter; not a new model |
| Optional tracking experiment | `scripts/run_p3_tt.py` | TrackTrack identity variant; no occupancy-improvement claim |

The Stage W four-frame loop remains a truth-free interface demonstration. It
is not an accuracy result or continuous-video validation.

## Model assets

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `D1_NDISPark_best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| `E1b_CBAM_best.pt` | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

The files are stored only under ignored
`implementation/outputs/stage_w_3_model_release_assets/` and remain outside
Git source. They were subsequently published in the
[v1.0.0 GitHub Release](https://github.com/Green014/automated-parking-occupancy-system/releases/tag/v1.0.0).

## Verification entry

From `implementation`:

```powershell
python scripts\verify_stage_w_3_release.py --current-only
python scripts\verify_stage_w_3_release.py
python scripts\verify_stage_v_w_registries.py
```

All checks are metadata, hashing, source scanning, CLI help, compilation, or
tests. They do not load a model or run inference.

## Publication state

- `source_publication_ready=true` after the recorded W.3 checks
- `model_assets_ready_for_github_release=true`
- `public_release_published=false`

Manual steps remain: configure an authorized remote, push the verified source
commit, create the real GitHub Release, upload both weight assets and the
checksum/metadata files, and replace the pending URL with the actual Release
URL.

## Publication follow-up

The manual steps above were completed on 2026-07-31. Release `v1.0.0` contains
the two checksum-verified model weights, `SHA256SUMS.txt`, and
`MODEL_RELEASE_METADATA.yaml`. The frozen `public_release_published=false`
field remains unchanged because it describes the state when W.3 validation
was recorded.
