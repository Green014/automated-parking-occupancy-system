# Automated Parking Lot Occupancy and Tracking System — Release Index

Date: 2026-07-29

This additive index preserves the frozen Stage L–R reports and registries. It is the
starting point for the final default release and the separate optional tracking
variant.

## Final default system (Stage S)

`D1 -> B1 -> E1b/F2 -> Occupancy Output`

- configuration: `configs/p3_stage_r_recommended_default_20260729.yaml`
- command: `parking-run-final`
- E4: disabled by default; conditional continuous-video option
- tracker: `none` by default
- D1-LL: negative low-light fine-tuning experiment
- report: `data/STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md`
- corrected evidence: `data/stage_s/STAGE_S_FINAL_SYSTEM_EVIDENCE.csv`
- demo: `data/stage_s/demo/demo_main.mp4`
- registry: `data/stage_s/STAGE_S_ARTIFACT_REGISTRY_20260729.yaml`

The Stage R final-compatible comparison is D1 R1 Macro F1 0.706681 versus D1-LL
R1 Macro F1 0.666978. D1 R1 occupied recall remains 0.370927; this is not a
deployment-ready result.

## Optional tracking system (Stage T)

Stage T is a separate, explicit variant:

`D1 -> TrackTrack identity association -> B1 -> E1b/F2 -> occupancy + track IDs`

It does not replace the Stage S default. Its consumed-development diagnostic and
claim boundaries are recorded independently in the Stage T report and registry.

## Claim taxonomy

- Formal: results explicitly frozen as formal tests by their original protocols.
- Post-hoc: component attribution or rendering from already frozen outputs.
- Development diagnostic: reused or consumed data used for engineering evidence.
- Blocked: conclusions that require new data or truth not currently available.

TrackTrack is not claimed to improve slot occupancy unless a same-input,
continuous-video, slot-truth comparison supports that statement.

