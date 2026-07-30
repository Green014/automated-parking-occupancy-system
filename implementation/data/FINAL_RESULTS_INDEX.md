# Automated Parking Lot Occupancy and Tracking System — Final Results Index

Date: 2026-07-29

Status: Stage R post-hoc closure complete

This additive index preserves the frozen historical conclusions in
[`literature_core/RESULTS.md`](../literature_core/RESULTS.md) and adds the
final Stage R interpretation without changing that Stage Q-v2-registered
file.

## Final evidence chain

| Stage | Evidence role | Frozen conclusion |
|---|---|---|
| L | Integrated D1/B1/E1b/F2/E4 workflow | Established component interfaces and exposed the E4 recall/false-occupied trade-off. |
| M | Open-source tracking robustness | Kept tracking separable from slot occupancy. |
| N / N-v2 / N-v3 | LMOT tracking diagnostics | Detector/MOT evidence only; not parking-slot occupancy evidence. |
| O | Low-light detector adaptation | D1-LL improved LMOT detection but did not establish occupancy improvement. |
| P | Parking-domain retention | D1-LL retention decision remained `FAIL`. |
| Q-v2 | External UPM-GTI low-light occupancy | D1 remained the default; no tracker was run. |
| R | Frozen-output component attribution | Selected D1 + B1 + F2 as the default occupancy path; made E4 conditional. |

## Stage R closure

Stage R joins the two frozen Stage Q-v2 `occupancy.csv` files to the same
7,896 truth labels and reads only already-emitted fields:

- R0 = B1 (`detector_occupied`);
- R1 = B1 + F2 (`raw_state`);
- R2 = B1 + F2 + E4 (`state`).

It is a post-hoc analysis, not a new untouched test. It performs no model
inference, retraining, threshold selection or E4 regeneration.

| Detector | Component | Macro F1 | Occupied recall | False-free | False-occupied | Count MAE |
|---|---|---:|---:|---:|---:|---:|
| D1 | R0: B1 | 0.613207 | 0.194236 | 0.805764 | 0.021414 | 1.316489 |
| D1 | R1: B1 + F2 | **0.706681** | 0.370927 | 0.629073 | **0.026768** | **0.962766** |
| D1 | R2: B1 + F2 + E4 | 0.664318 | **0.446115** | **0.553885** | 0.085940 | 1.329787 |
| D1-LL | R0: B1 | 0.597168 | 0.228070 | 0.771930 | **0.055790** | 0.978723 |
| D1-LL | R1: B1 + F2 | **0.666978** | 0.383459 | 0.616541 | 0.060721 | **0.816489** |
| D1-LL | R2: B1 + F2 + E4 | 0.617484 | **0.457393** | **0.542607** | 0.139335 | 1.909574 |

The truth is imbalanced: 798 occupied labels (10.106%) and 7,098 vacant
labels (89.894%). The recommendation therefore uses class-aware metrics,
confusion counts and count error rather than accuracy alone.

## Final configuration

- Default: `D1 -> B1 -> F2 -> Occupancy Output`.
- Conditional: add E4 only for genuinely continuous video after separate
  calibration.
- Negative experiment retained for provenance: D1-LL replaces D1.
- Independent optional research module: TrackTrack after detection for MOT.

E4 is not TrackTrack and maintains no vehicle IDs. TrackTrack was not run in
Stage Q-v2, does not belong to the default occupancy path, and has no
demonstrated slot-level occupancy improvement in this evidence chain.

See
[`STAGE_R_COMPONENT_ATTRIBUTION_REPORT.md`](STAGE_R_COMPONENT_ATTRIBUTION_REPORT.md)
for the complete component, per-sequence, per-slot and temporal-validity
analysis.
