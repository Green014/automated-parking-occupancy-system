# Stage W.1 Local Review Candidate Index

Date: 2026-07-30  
Public release ready: **false**

## Entry points

| Entry | Purpose |
|---|---|
| `parking-run-final` | default P3 research runtime |
| `parking-compare` | Stage V.1 Classic/Detection/Fusion comparison |
| `parking-dashboard` | local Stage W Flask dashboard |
| `python scripts/run_p3_tt.py` | optional P3-TT TrackTrack experiment |

## Installation

```powershell
python -m pip install -e .\literature_core
python -m pip install -e ".[integrated,dashboard,dev]"
```

`stage_w_requirements.txt` is retained as a compatibility-only dashboard
dependency entry and uses the same `Flask>=3.1,<4` range.

## Current documents

- `STAGE_W_1_RELEASE_HARDENING_REPORT.md`
- `STAGE_W_1_ARTIFACT_REGISTRY.yaml`
- `STAGE_W_PERMISSION_AND_PROVENANCE.md`
- `STAGE_W_REPRODUCTION_GUIDE.md`
- `../README.md`
- repository-level `README.md`
- repository-level `FINAL_RELEASE_INDEX.md`

## Historical snapshot classification

- `STAGE_V_1_ARTIFACT_REGISTRY.yaml`: pre-hardening historical snapshot,
  registry SHA-256
  `19aec081be8e9707f0025365a136dcd6fc68a005a373ec7a4d6e1e99680bd372`.
- `STAGE_W_ARTIFACT_REGISTRY.yaml`: pre-hardening historical snapshot,
  registry SHA-256
  `0a1daf77de33d753f5a79609146f80c5a53a382f78975b03b086b036af410bc9`.

The old registry files are not rewritten. Their immutable smoke evidence
remains hash-verifiable, while the W.1 registry owns the corrected live
source, tests, dependency metadata, READMEs and release indexes.

## Unchanged local evidence

- `outputs/stage_v_1_multimode_smoke_20260730_v3`
- `outputs/stage_v_1_fusion_smoke_20260730_v3`
- `outputs/stage_w_dashboard_smoke_20260730_v3`
- `outputs/stage_w_dashboard_smoke_20260730_v3/dashboard_ui_demo.mp4`

These ignored files are existing evidence only. They were not regenerated,
uploaded or reinterpreted. The four-frame loop remains a consumed interface
demonstration with no truth and no accuracy or continuous-video claim.

## Release gate

Do not publish or push while the project licence, Ultralytics boundary,
granting member/date/written record, modified HTML/CSS redistribution scope
and model/data/video redistribution scope remain unresolved.

