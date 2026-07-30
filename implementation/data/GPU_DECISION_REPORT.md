# Stage G Local GPU Decision

Decision ID: `GPU-GATE-NDISPARK-D1-20260727-01`

Stage G passes. Formal D1 training is authorized on the local RTX 3060 Laptop
GPU only. A paid/remote GPU is not authorized or justified, and an A100 is not
needed. Stage G ran no training or prediction and accessed no test data.

## Evidence and calculation

The completed Stage F run at `imgsz=640`, physical batch 4 measured:

| Quantity | Measured value |
|---|---:|
| Total GPU memory | 6,441,926,656 bytes (5.9995 GiB) |
| CUDA free before training | 5,379,194,880 bytes (5.0098 GiB) |
| Peak Torch allocated | 635,307,520 bytes |
| Peak Torch reserved | 767,557,632 bytes (0.7148 GiB) |
| Peak reserved / total | 11.915% |
| Three epoch times | 5.0033, 2.8234, 2.5708 s |
| `model.train` wall time | 21.3680 s |

For a 50-epoch upper limit at the same image size and batch, the central
estimate uses the mean of smoke epochs 2 and 3 plus measured fixed overhead.
It is 145.83 seconds (2.43 minutes). A conservative estimate using the slowest
smoke epoch is 261.13 seconds (4.35 minutes). A stress bound that doubles the
slowest smoke epoch is 511.30 seconds (8.52 minutes), still far below the
two-hour local-training gate. These are extrapolations, not measured
50-epoch durations; early stopping may shorten the actual run.

## Resolution and batch feasibility

Planning estimates scale the measured peak reserved memory by batch ratio,
squared image-size ratio, and a 1.25 safety factor:

`peak * target_batch/4 * (target_imgsz/640)^2 * 1.25`

| Image size | Batch | Reserved estimate | Compared with 5.0098 GiB smoke free memory | Executed? |
|---:|---:|---:|---|---|
| 640 | 4 | 0.894 GiB | fits | yes |
| 640 | 8 | 1.787 GiB | fits | no |
| 960 | 4 | 2.010 GiB | fits | no |
| 960 | 8 | 4.021 GiB | fits with limited margin | no |
| 1280 | 4 | 3.574 GiB | fits with less margin | no |
| 1280 | 8 | 7.148 GiB | does not fit | no |

Only 640/batch 4 is execution-validated. The 960 and 1280 rows are analytical
planning estimates and cannot be reported as measured feasibility. They are
also outside the frozen common detector comparison, so the formal run remains
at 640.

Batch 4 is the largest allowed batch directly measured by Stage F. With the
Ultralytics 8.4.104 nominal batch (`nbs`) of 64, its post-warm-up accumulation
is 16 steps and its nominal effective batch is 64. Ultralytics ramps this
accumulation during warm-up; it is not a separately tuned accuracy parameter.

## Frozen Stage H configuration

The new formal experiment is `D1-NDISPARK-FT-20260727-01`:

- fresh COCO-pretrained `yolov8n.pt` initialization, not the smoke checkpoint;
- 50 maximum epochs, patience 10, one seed (`20260727`);
- `imgsz=640`, physical batch 4, AMP, deterministic execution;
- AdamW, `lr0=0.001`, weight decay `0.0005`;
- exact recorded Ultralytics 8.4.104 augmentation defaults from the smoke;
- NDISPark official train plus consumed development validation only;
- new ignored output directory, with best and last checkpoints retained.

The complete frozen configuration is
`../configs/d1_ndispark_formal_frozen_20260727.yaml`. Data and weight paths
remain CLI/environment inputs; no user-specific absolute path is committed.

## Rental and minimum hardware decision

The selected run fits the measured 6 GiB device with a large margin and the
stress-bound runtime is below nine minutes. Rental duration is therefore zero
hours. No remote benchmark was executed, so no invented local-versus-rental
speedup is claimed; provisioning and transfer would dominate this run.

Four GiB is an operational lower bound inferred for the exact 640/batch-4
configuration, but 6 GiB is the recommended minimum and the reproducibility
target because framework, driver, display, and background allocations vary.
An A100 is unnecessary. YOLO-World fine-tuning remains outside the current
mandatory scope.

## Reproduction

This command reads the ignored Stage F summary and writes a new ignored
decision artifact. It never trains or predicts:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts\analyze_gpu_decision.py `
  --smoke-summary outputs\d1_ndispark_smoke_20260727_v3\smoke_summary.json `
  --output outputs\gpu_decision_<new-id>.json
```

Executed artifact:

- `outputs/gpu_decision_20260727_v1.json`
- 7,375 bytes
- SHA-256
  `10626027fb39344aad180f98fe45c9086d95bb4b4962c37575d8e32acf49baab`

Stage H may now implement and execute the one formal local run. Paid/remote
GPU use, a second seed, changed image size, batch 8, or any hyperparameter
search would require a new protocol decision.

## Verification

Executed on 2026-07-27:

- Stage G unit and freeze tests: 8 passed;
- complete `implementation` suite: 71 passed;
- complete `literature_core` suite: 82 passed;
- Python AST parse: 127 files;
- YAML load: 10 configuration/training records; generated decision JSON load:
  passed;
- `git diff --check`: passed (line-ending notices only);
- historical static freeze: 17/17 hashes, 4,081 frames, and 144,965 slot
  records passed;
- historical temporal freeze: 11/11 hashes passed.

The verifier reports were written only to new ignored output paths. No frozen
artifact was changed.
