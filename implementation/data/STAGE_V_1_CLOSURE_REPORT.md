# Stage V.1 Closure Report

Date: 2026-07-30  
Status: implementation and model-backed functional smoke complete  
Supersedes: Stage V for the corrected runtime only; all Stage J–U.1 evidence
remains frozen and unchanged.

## Closure decision

Stage V.1 closes the four release-blocking defects found in Stage V:

1. C1/C2 can no longer silently accept a changed P3 configuration while
   claiming frozen identity.
2. The first frame establishes initial state and no longer creates a false
   arrival; non-continuous image inputs never create temporal events.
3. cache-hit Fusion displays and reports attributed processing time, including
   the measured D1 cost, rather than presenting cache-only throughput as
   end-to-end capability.
4. Stage U is verified as its saved historical snapshot, while V.1 and W have
   independent additive release registries and audits.

The default remains `D1 + B1 + E1b + F2`. E4 and vehicle tracking are off by
default. F2 reviews every detector-negative slot, not an unimplemented
confidence interval. TrackTrack, when explicitly selected, operates after D1
and before B1 and supplies vehicle identity continuity only. No occupancy
accuracy improvement is claimed for tracking.

## Frozen-configuration enforcement

The default C1/C2 identity is accepted only when both the formal configuration
SHA-256 and all critical parameters match:

- configuration ID;
- D1 confidence, image size, NMS IoU, class-agnostic NMS and maximum
  detections;
- B1 minimum slot coverage and one-to-one behavior;
- E1b occupied threshold, patch size and perspective warp;
- F2 detector-positive authority and detector-negative review rule;
- E4 default state;
- tracker default state.

The formal configuration SHA-256 is
`198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0`.
A mismatch is rejected unless `--allow-custom-config` is explicit. An allowed
custom run receives custom method IDs, records the supplied file hash and
parameter-level differences, and sets `frozen_parameters_changed: true`.
Explicit E4/tracker variants are similarly recorded as runtime variants and
are not relabeled as the frozen comparison.

## Event contract

| Input role | First frame | Later state changes | Summary flag |
|---|---|---|---|
| Continuous video | initializes state only | may emit arrival/departure | `events_temporally_valid: true` |
| Image directory | initializes each observation only | no events | `events_temporally_valid: false` |

Without E4, continuous-video events are raw frame-level state changes. They
must not be described as stable parking events. With E4 explicitly enabled,
the event stream follows the stabilized slot state.

## Performance contract

Every displayed FPS value is derived from `attributed_backend_total`. A
cache-hit frame reports `cache: hit` but retains the corresponding measured
D1 detector cost. `backend_total` remains available for component diagnostics
and is not presented as comparable end-to-end throughput.

In the final same-input smoke, the steady-state figures were:

| Method | Definition | Mean frame latency (ms) | Steady-state FPS |
|---|---|---:|---:|
| C0 Classic | processing plus render/write | 29.954 | 33.753 |
| C1 D1+B1 | attributed processing plus render/write | 30.418 | 35.253 |
| C2 D1+B1+E1b+F2 | attributed processing plus render/write | 60.060 | 17.851 |

For C2, `detector_execution` is zero on cache-hit frames while the attributed
detector mean remains 18.295 ms. A separate single-Fusion smoke used the same
attribution definition and measured a mean attributed backend time of
46.125 ms. These four-frame measurements are functional diagnostics, not
performance benchmarks.

## Final smoke evidence

The authoritative local runs are:

- `outputs/stage_v_1_multimode_smoke_20260730_v3`
- `outputs/stage_v_1_fusion_smoke_20260730_v3`

Both use a four-frame video made by repeating an already consumed development
image. They use five configured polygons, have no truth bundle, and are
therefore labeled `consumed demonstration only` and
`not_computed_no_truth`. All modes decoded four frames, produced 20 slot-frame
records, rendered every configured slot, and wrote zero false initial events.
They do not establish accuracy, transition latency, tracking improvement or
fixed-camera field performance.

The verified model identities are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| D1 `best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| E1b `best.pt` | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

Weights are local prerequisites and are not copied into the release
candidate.

## Version and release boundary

The Stage U verifier checks the immutable saved registry identity, its saved
candidate coverage, audit/CSV agreement and saved Stage U evidence. It does
not compare the historical Stage U manifest with a live post-U worktree.
Stage V.1/W files are covered by their own registries and cannot change the
meaning of the Stage U snapshot.

The original `STAGE_V_ARTIFACT_REGISTRY.yaml` is retained as historical
evidence. The corrected boundary is registered in
`STAGE_V_1_ARTIFACT_REGISTRY.yaml`.

