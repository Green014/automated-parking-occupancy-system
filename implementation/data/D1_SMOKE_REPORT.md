# D1 NDISPark Local Smoke Run

Date: 27 July 2026  
Experiment: `D1-NDISPARK-SMOKE-20260727-01`  
Completed run: `d1_ndispark_smoke_20260727_v3`

## Outcome

Stage F passed. A COCO-pretrained YOLOv8n was fine-tuned for three epochs on
the frozen NDISPark train split and validated on the consumed NDISPark
development-validation split. The run completed locally on the RTX 3060
Laptop GPU with batch 4 and AMP. It produced finite, changing losses, valid
development metrics, updated checkpoints, and complete resource evidence.

This is a smoke result, not the formal D1 model and not a final detector
comparison. The validation values below prove that training and validation
execute successfully; they must not be compared with historical 1280-pixel
results or reported as untouched-test performance.

## Frozen inputs

| Item | Value |
|---|---|
| Dataset protocol | `DPROTO-NDISPARK-ONLY-20260727-01` |
| Comparison protocol | `D-COMP-NDISPARK-DEV-20260727-01` |
| Initialization | COCO-pretrained `yolov8n.pt` |
| Initial SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| Train | 112 images, 2,577 boxes, one legitimate background image |
| Development validation | 30 night images, 725 boxes |
| Count-only test | Not accessed |

The configuration was 3 epochs, batch 4, image size 640, seed 20260727,
deterministic execution, AMP, AdamW, learning rate 0.001, weight decay 0.0005,
two data-loader workers, and CUDA device 0.

## Executed attempts

All attempts are retained:

| Run | Result | Epochs | Explanation |
|---|---|---:|---|
| `v1` | Failed | 0 | Ultralytics configuration parent did not exist and it fell back to the read-only worktree |
| `v2` | Failed | 0 | `pythonw.exe` provided no stdout stream for the progress writer; failure occurred before the first batch |
| `v3` | Complete | 3 | Standard Python with redirected logs; frozen method settings unchanged |

The first two are engineering execution failures, not model-performance
results. Their `smoke_failure.json` files were not deleted or overwritten.

## Training and validation evidence

| Metric | Epoch 1 | Epoch 3 | Change |
|---|---:|---:|---:|
| train box loss | 1.49829 | 1.38859 | -0.10970 |
| train class loss | 2.36296 | 1.23574 | -1.12722 |
| train DFL loss | 1.10219 | 1.03713 | -0.06506 |
| development precision | 0.06178 | 0.74390 | diagnostic only |
| development recall | 0.76690 | 0.68109 | diagnostic only |
| development mAP@0.5 | 0.12651 | 0.75221 | diagnostic only |
| development mAP@0.5:0.95 | 0.04576 | 0.43950 | diagnostic only |

All logged numeric values were finite. There was no NaN, OOM, batch
auto-reduction, or empty validation label. Validation inference processed all
30 images. The three epoch durations were 5.00, 2.82, and 2.57 seconds. The
`model.train` wall time, including setup and final validation/plotting, was
21.37 seconds.

The best and last stripped checkpoints are identical because epoch 3 was both
the final and best smoke epoch. Their SHA-256 is
`4e57a68f93a050861bc8a9dde4c22b04c102f0984f19c7d39cbdde43ced8f2d1`,
which differs from the initialization hash. The smoke checkpoint is not
promoted to formal D1.

## Resource evidence

| Measurement | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| Total VRAM | 6,441,926,656 bytes |
| Free before training | 5,379,194,880 bytes |
| Peak Torch allocated | 635,307,520 bytes |
| Peak Torch reserved | 767,557,632 bytes |
| Peak reserved fraction | 11.92% |
| Mean train-batch wait fraction | 0.32% |

The callback wait fraction is a practical prefetch/wait proxy, not a full
profiler trace. Under that definition no material data-loader bottleneck was
observed.

## Artifacts and provenance

The machine-readable record is
`training/d1_ndispark_smoke_20260727.yaml`. Runtime outputs are in the ignored
directory `outputs/d1_ndispark_smoke_20260727_v3/`; it contains 31 files and
90,026,576 bytes. This includes `args.yaml`, `results.csv`, `results.png`,
training/validation images, PR/confusion plots, all epoch checkpoints,
`best.pt`, `last.pt`, `resource_metrics.json`, and `smoke_summary.json`.

`src/parking_occupancy/training_smoke.py` is a local integration around the
Ultralytics training API. It is not described as a fork of an external
parking-management system.

## Verification

- smoke-specific tests: 7 passed;
- complete `implementation` suite after the freeze test: 63 passed;
- complete `literature_core` suite: 82 passed;
- read-only AST syntax check: 123 Python files passed;
- historical static freeze: 17/17 artifacts, 4,081 frames, and 144,965 slot
  records verified;
- historical temporal freeze: 11/11 artifacts verified;
- generated output, weights, process logs, and verifier reports confirmed
  Git-ignored;
- `git diff --check`: passed, with line-ending notices only.

## Gate

Stage F is complete. Stage G may now use these measurements to decide local
formal-training feasibility, batch/accumulation strategy, 960/1280
feasibility, expected duration, and whether any rented GPU is justified.

Formal D1 training remains blocked until Stage G is documented. Paid or remote
GPU use remains prohibited.
