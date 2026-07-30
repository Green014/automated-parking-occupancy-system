# Local Review Candidate Index

Date: 2026-07-30

## Runtime entry points

- Default research runtime: `parking-run-final`
- Controlled Stage V.1 comparison: `parking-compare`
- Local Stage W dashboard: `parking-dashboard`
- Optional P3-TT experiment: `python implementation/scripts/run_p3_tt.py`
- Frozen Stage T experiment: retained unchanged for historical reproduction

## Role and method boundary

- Default main system: `D1 -> B1 -> E1b/F2`, E4 off, tracker none
- Comparison tool: Classic / D1+B1 / D1+B1+E1b+F2
- Dashboard: presentation adapter around a selected backend, not a new method
- Optional tracking experiment: `D1 -> TrackTrack -> B1 -> E1b/F2`, E4 off
- TrackTrack slot-level occupancy improvement: not established
- Deployment readiness: not claimed

## Primary documents

- `README.md`
- `implementation/data/SYSTEM_RELEASE_INDEX.md`
- `implementation/data/STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md`
- `implementation/data/STAGE_T_TRACKTRACK_ENHANCED_VARIANT_REPORT.md`
- `implementation/data/STAGE_U_PORTABLE_FINAL_RELEASE_REPORT.md`
- `implementation/data/STAGE_U_1_FINAL_RELEASE_CORRECTION_REPORT.md`
- `implementation/data/STAGE_U_1_MODEL_ASSETS.md`
- `implementation/data/STAGE_W_1_RELEASE_HARDENING_REPORT.md`
- `implementation/data/STAGE_W_2_CROSS_PLATFORM_RELEASE_REPORT.md`
- `implementation/data/STAGE_W_3_PRIVACY_AND_MODEL_RELEASE_REPORT.md`
- `implementation/data/STAGE_W_3_RELEASE_INDEX.md`
- `implementation/data/STAGE_W_3_PUBLIC_SOURCE_MANIFEST.yaml`
- `implementation/data/STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml`
- `implementation/data/MODEL_CARD_D1.md`
- `implementation/data/MODEL_CARD_E1B.md`
- `implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md`
- `LICENSE`
- `THIRD_PARTY_NOTICES.md`

## Portable evidence

- Stage R portable historical registry
- Stage S portable historical registry
- Stage U submission artifact registry
- Stage U submission audit and historical-registry classification
- Frozen Stage S and Stage T presentation media
- Additive Stage U.1 TrackTrack identity-diagnostic presentation copy

Historical registries that bind ignored runtime outputs, datasets, weights, or
machine-specific absolute paths are retained locally and intentionally omitted
from the portable package. Their contents are not rewritten.

Stage V.1 and W registries are retained as pre-hardening historical snapshots.
Stage W.1 is the exact pre-W.2 snapshot and W.2 is the exact pre-W.3
cross-platform source snapshot. The current public source candidate is governed
by the W.3 registry and public source manifest. `.gitattributes` enforces LF
for release text, including `LICENSE`, and binary treatment for models, media,
images, documents, and archives.

## Release gate

The project source license is AGPL-3.0-only. Public redistribution of the
adapted interface was authorized by the upstream code owner and is attested
anonymously by the project owner; the underlying evidence remains private and
can be shown to the instructor if required. Ultralytics, datasets, and the
adapted interface remain independent license/provenance boundaries.

`source_publication_ready=true` and
`model_assets_ready_for_github_release=true` are local preparation states,
subject to the W.3 verifier. `public_release_published=false`: an authorized
remote push, real GitHub Release, weight upload, and replacement of
`Release URL pending` remain manual actions.
