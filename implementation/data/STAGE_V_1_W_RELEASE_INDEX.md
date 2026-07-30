# Stage V.1 / W Additive Release Index

Date: 2026-07-30

This is the non-frozen index for the post-Stage-U.1 work. Earlier final
indexes and registries remain historical snapshots and are not rewritten.

## User entry points

- System overview and commands: `../README.md`
- Stage V.1 closure: `STAGE_V_1_CLOSURE_REPORT.md`
- Stage W integration: `STAGE_W_UI_INTEGRATION_REPORT.md`
- Permission/provenance boundary: `STAGE_W_PERMISSION_AND_PROVENANCE.md`
- Reproduction: `STAGE_W_REPRODUCTION_GUIDE.md`
- V.1 registry: `STAGE_V_1_ARTIFACT_REGISTRY.yaml`
- W registry: `STAGE_W_ARTIFACT_REGISTRY.yaml`

## Runtime entry points

- Three core modes and corrected comparison:
  `scripts/run_stage_v_multimode_demo.py`
- Flask dashboard:
  `scripts/run_stage_w_dashboard.py`
- Registry verification:
  `scripts/verify_stage_v_w_registries.py`

Core modes are `classic`, `detection`, and `fusion`. The optional
`member-reference` mode requires an external audited checkout and is not a
fallback.

## Local functional evidence

- `outputs/stage_v_1_multimode_smoke_20260730_v3`
- `outputs/stage_v_1_fusion_smoke_20260730_v3`
- `outputs/stage_w_dashboard_smoke_20260730_v3`
- `outputs/stage_w_dashboard_smoke_20260730_v3/dashboard_ui_demo.mp4`

These ignored outputs are registered as locally available optional evidence.
They are consumed, truth-free demonstrations and make no accuracy claim.

## Public-release status

Local course-project integration is complete. Public merge/release is not
recommended yet because public redistribution of the member-derived interface
has not been confirmed, the granting member/date remain to be recorded, and
the project/Ultralytics licence boundary remains unresolved. No public push is
part of this release.

