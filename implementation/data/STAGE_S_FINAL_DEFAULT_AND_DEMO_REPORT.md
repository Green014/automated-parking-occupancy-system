# Stage S — Final Default Release and Demonstration

Date: 2026-07-29  
Protocol: `STAGE-S-FINAL-DEFAULT-AND-DEMO-20260729-01`  
Status: final default closure from frozen Stage Q-v2 / Stage R evidence

## Scope and claim class

Stage S performs configuration alignment, submission-package auditing, documentation
synchronization, and post-hoc rendering from frozen Stage Q-v2 artifacts. It does not
run detector or classifier inference, train a model, tune a threshold, or create a new
untouched test.

The final default occupancy method is:

`D1 detector -> B1 polygon one-to-one mapping -> E1b/F2 uncertainty-gated fusion -> slot occupancy output`

E4 is disabled by default and is conditional on separate calibration with genuinely
continuous video. Tracking is disabled by default. TrackTrack remains an independent,
optional MOT research module and has no demonstrated slot-level occupancy benefit in
the frozen Stage Q-v2 evidence.

## Configuration closure

The original `p3_integrated_runtime_defaults_20260729.yaml` remains unchanged. The
new explicit release configuration is
`configs/p3_stage_r_recommended_default_20260729.yaml`:

- detector: D1
- mapping: B1
- fusion: E1b/F2
- `temporal.default_enabled: false`
- `tracking.default_backend: none`
- `claims.deployment_ready: false`

Both `parking-run-final` and the existing `parking-run-integrated` entry resolve to
this configuration when `--config` is omitted. E4 and a tracker still require explicit
arguments such as `--temporal` and `--tracker tracktrack`.

Example default invocation:

```powershell
parking-run-final `
  --input <continuous-or-static-input> `
  --slots <slot-polygons.json> `
  --d1-weights <d1-best.pt> `
  --e1b-checkpoint <e1b-best.pt> `
  --output-dir <output-directory>
```

This is a reproducible research interface, not a deployment-readiness statement.

## Corrected final evidence

The detector comparison at the final default component boundary must use R1, not R0
or R2:

| Detector | Final-compatible component | Macro F1 | Decision |
|---|---|---:|---|
| D1 | B1 + F2 (R1) | 0.706681 | default |
| D1-LL | B1 + F2 (R1) | 0.666978 | retained negative experiment |

For D1, R1 occupied recall is only 0.370927. Accuracy alone is not sufficient under
the vacant-heavy class distribution, and the remaining occupied misses prevent a
deployment-ready claim. E4 raised occupied recall relative to R1 but reduced Macro F1
by 0.042363 and increased false-occupied behavior on the sparse Stage Q-v2 samples;
therefore it is not part of the final default.

## Submission-package audit

The candidate set is obtained from Git-tracked plus unignored untracked files. The
audit explicitly rejects model weights, virtual environments, datasets, and runtime
`outputs`/`runs`. Local data are not deleted.

The repository `.gitignore` now excludes at least:

- `implementation/.venv_*/`
- `implementation/data/external/stage_o_training_*/`
- `implementation/data/external/*.partial/`

It additionally excludes the external-data root so the local Stage Q and other
datasets cannot enter the submission candidate set accidentally. The final candidate
counts and byte total are recorded in
`data/stage_s/STAGE_S_SUBMISSION_AUDIT.json`; the complete candidate list is in
`data/stage_s/STAGE_S_SUBMISSION_CANDIDATES.csv`.

## Frozen-output demonstration

`data/stage_s/demo/demo_main.mp4` is a 50.0-second, 1280 x 720, 10 FPS post-hoc
render. It decodes to all 500 expected frames. The requested encoder is `mp4v` and
the decoded FOURCC is `FMP4`.

Timeline:

- 0–20 s: selected consecutive source-frame segments `gopro1 92–116`,
  `gopro4 95–123`, and `gopro26 109–136`
- 20–33 s: frozen same-frame D1 / D1-LL comparison
- 33–43 s: B1-to-F2 occupied recovery examples
- 43–50 s: frozen geometry / false-occupied failure examples

The main visualization uses R1 `raw_state`, not E4 `state`, and is labelled
“selected consecutive source frames, slowed for visualization.” It uses only frozen
images, `occupancy.csv`, `detections.jsonl`, polygons, and manifest hashes. It is
post-hoc presentation material and not a new evaluation.

## TrackEval environment and validation

The Stage N pinned local TrackEval source remains available at commit
`12c8791b303e0a0b50f753af204249e622d0281a` with tree
`ce583e59d96b33aad5f8f62149d96bc6bc4d8f96`. All three tests selected by
`-k official_trackeval` pass when that source is placed on `PYTHONPATH`.

Importing TrackEval emits an optional BURST/`pycocotools` warning. The official MOT
metrics used by the tests are available; no production logic or test requirement was
weakened.

## Validation record

Successful commands:

```powershell
python -m pytest tests/test_stage_s_release.py tests/test_stage_s_demo.py -q
# 8 passed

python -m pytest tests/test_stage_r_component_attribution.py -q
# 6 passed

$env:PYTHONPATH='<pinned TrackEval-12c8791b303e source>'
python -m pytest tests -q
# 264 passed, including all three official TrackEval tests

python -m pytest literature_core/tests -q
# 83 passed

python -m compileall -q implementation/src implementation/scripts `
  implementation/tests implementation/literature_core/src `
  implementation/literature_core/scripts implementation/literature_core/tests

git diff --check
```

The first full implementation invocation without the pinned TrackEval source on
`PYTHONPATH` produced exactly three `ModuleNotFoundError: trackeval` failures. Repeating
the complete suite with the recorded source path passed all 264 tests. This is retained
as an optional-environment requirement rather than hidden by changing production code
or tests. `git diff --check` reports only Git's existing LF-to-CRLF conversion warnings,
not whitespace errors.

## Historical-registry gate

Stage S records an entry snapshot and an exit snapshot for the Stage L–R registries.
Some early registries already referenced living project documents or Stage N files
that were subsequently changed by later historical stages. Those pre-existing
binding differences are retained verbatim rather than “fixed” by rewriting history.

The Stage S exit gate requires every historical registry file hash, byte size, binding
count, verification result, and recorded pre-existing difference to be identical to
the entry snapshot. The gate result is stored in
`data/stage_s/STAGE_S_HISTORICAL_REGISTRY_GATE.json`.

## Conclusion

Stage S closes the submission default as D1 + B1 + F2 with E4 and all trackers off.
The system is ready for research-report, presentation, and submission-material
production, subject to the explicit limitation that occupied recall remains low and
the evidence does not establish deployment readiness.
