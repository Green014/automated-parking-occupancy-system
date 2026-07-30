# Stage H Formal D1 Training

Experiment: `D1-NDISPARK-FT-20260727-01`

Stage H passes. One fixed-seed formal D1 run completed locally from the
original COCO-pretrained `yolov8n.pt`. It used only NDISPark official train
and consumed development-validation data. No counting test, CNR-EXT, PKLot,
VIRAT, remote GPU, or paid GPU was accessed.

## Executed configuration

| Field | Value |
|---|---|
| Model | YOLOv8n, freshly initialized from frozen COCO weights |
| Train data | 112 images / 2,577 vehicle boxes |
| Development validation | 30 images / 725 vehicle boxes |
| Image size | 640 |
| Physical / nominal batch | 4 / 64 |
| Post-warm-up accumulation | 16 configured steps |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.001 / 0.0005 |
| Seed | 20260727 |
| AMP / deterministic | true / true |
| Maximum epochs / patience | 50 / 10 |
| Completed / best epoch | 47 / 37 |

Training stopped after epoch 47 because there had been no fitness improvement
for the frozen patience of 10 epochs. No second seed or hyperparameter search
was run.

## Development-validation diagnostics

These are training-stage diagnostics on the consumed validation split, not a
final detector test and not yet the frozen D0/D1/D2 comparison.

| Checkpoint/epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Best, epoch 37 | 0.93708 | 0.88339 | 0.94478 | 0.67556 |
| Last, epoch 47 | 0.93105 | 0.89396 | 0.94946 | 0.67232 |

All numeric result values were finite. From the first to final epoch, box,
classification, and DFL losses changed by -0.48947, -1.29320, and -0.16406.
Visual inspection of `results.png` confirms declining train/validation losses
and plateauing detection metrics. A short classification-loss increase appears
when mosaic is closed for the final ten scheduled epochs, but it does not
produce NaN, divergence, or a validation-metric collapse.

Selected formal D1:

- `outputs/d1_ndispark_formal_20260727_v1/weights/best.pt`
- 6,255,409 bytes
- SHA-256
  `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64`

The smoke checkpoint was not used.

## Runtime and GPU

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6,441,926,656 bytes total.
- `results.csv` cumulative training time: 299.473 seconds.
- Ultralytics-reported training time: 0.083 hours.
- Runner start through retained audit failure: 337.077 seconds.
- Mean recovered epoch duration: 6.372 seconds.
- Progress-log peak reserved memory: approximately 0.814 GiB.
- No NaN, OOM, or automatic batch reduction was observed.

The 0.814 GiB value comes from Ultralytics' rounded progress display. Exact
callback bytes are unavailable because of the retained post-run audit issue
below; this limitation is not hidden.

## Retained post-run audit failure

The training and final best-checkpoint validation completed, then the first
runner version raised `Resource probe epoch count mismatch`. Ultralytics calls
`on_fit_epoch_end` once more for final evaluation of `best.pt`; after early
stopping, the Stage F callback guard did not distinguish this from a training
epoch and produced one extra resource record.

The failed audit JSON and traceback remain in the v1 artifacts. The callback
was corrected to require an active training epoch, and a unit test covers the
final-evaluation callback. Existing results, weights, logs, and curves were
then checked offline. The recovery loaded no model and ran no training,
validation, or prediction. A second formal run was deliberately not performed.

This is an engineering audit failure, not evidence for or against D1 model
quality. It also means exact callback peak-memory bytes and dataloader wait
statistics are unavailable for this formal run; the successful Stage F
measurements remain the exact source for those quantities.

## Artifacts and interpretation

The ignored output directory contains 37 files / 139,626,147 bytes, including
best/last checkpoints, epoch CSV, training curves, PR curve, confusion matrix,
executed arguments, console log, failure record, recovered summary, and
resource summary. Their hashes are frozen in
`training/d1_formal_training_20260727.yaml`.

The Stage H verifier passed all 14 recorded size and SHA-256 checks and wrote
only a new ignored report:
`outputs/d1_formal_training_verification_20260727_v1.json`.

Stage H does not establish that D1 beats D0 or D2. Stage I must first run all
three detectors through the already frozen canonical one-class evaluator on
the same consumed development validation. The NDISPark count-only test remains
closed until the detector and one common counting operating rule are frozen.

## Verification

Executed after freezing the Stage H record:

- Stage H runner, callback, artifact, and freeze tests: 9 passed;
- complete `implementation` suite: 80 passed;
- complete `literature_core` suite: 82 passed;
- Python AST parse: 133 files;
- YAML load: 12 configuration/training records;
- formal summary and verifier JSON load: 2 passed;
- `git diff --check`: passed, with line-ending notices only;
- Stage H artifact verifier: 14/14 size and SHA-256 checks passed;
- historical static freeze: 17/17 hashes, 4,081 frames, and 144,965 slot
  records passed;
- historical temporal freeze: 11/11 hashes passed.

All verification reports were written to new ignored paths. No historical
artifact was modified.
