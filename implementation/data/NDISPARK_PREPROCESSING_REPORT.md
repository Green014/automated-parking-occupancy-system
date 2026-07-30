# NDISPark preprocessing report

Preparation ID: `DPREP-NDISPARK-ONLY-20260727-01`  
Protocol: `DPROTO-NDISPARK-ONLY-20260727-01`  
Executed: 27 July 2026  
Status: complete

## Outcome

The frozen NDISPark-only source was converted into a one-class Ultralytics
dataset without changing official split membership:

| Split | Images | Source boxes | Written YOLO boxes | Role |
|---|---:|---:|---:|---|
| train | 112 | 2,577 | 2,577 | D1 fine-tuning |
| validation | 30 | 725 | 725 | consumed development validation |
| test | 117 | count truth only | no label files | count-only test |

All 259 image hashes matched the Stage C manifests. No image or annotation was
added to another split. The category mapping is source COCO ID 3 `car` to
project ID 0 `vehicle`.

## Validation and correction log

- exact duplicate-image groups: 0;
- exact duplicate-box exclusions: 0;
- boundary-clipped boxes: 0;
- invalid/empty boxes excluded: 0;
- normalized YOLO values outside `[0, 1]`: 0;
- duplicate normalized label groups: 0;
- corrupt images reported by Ultralytics: 0.

One official train image, `69_1531569735.jpg`, contains no vehicle box and is
retained as a legitimate background image with an empty label file. The
machine-readable `annotation_actions.csv` contains its header and zero action
rows, proving that the run performed no repair or exclusion.

## Ultralytics loading check

Ultralytics 8.4.104 loaded the generated dataset without loading a model or
running prediction:

- train: 112 images, 2,577 boxes, class set `{0}`;
- validation: 30 images, 725 boxes, class set `{0}`;
- label cache: generated successfully for both box-labelled splits.

The generated `dataset.yaml` contains the runtime output path required by
Ultralytics. It is located under the Git-ignored `data/processed/` tree and is
not committed. All committed source code obtains raw and generated roots from
CLI arguments or `PARKING_DATA_ROOT`.

## Generated evidence

The ignored generated directory initially contained 406 files and
110,595,770 bytes before the two later Ultralytics cache files. Important
artifact hashes are:

| Artifact | SHA-256 |
|---|---|
| `dataset.yaml` | `8e000ba2fe912edf09c9386426da58315950c708726468df36a7c25163eab170` |
| `class_mapping.yaml` | `14a55c6e2eb0a88cb97a4e6e3de2467c488be82761dacb0d10d9a5efcbca3c16` |
| `prepared_manifest.csv` | `cf6f653512ac91b1bb7afe40fa5034f378f3e128011cc8e49da890e40a6282c5` |
| `annotation_actions.csv` | `90de297a565a86c7c57dd9c5e209ca7ec4f3358f727fcdf33a2f6f2f6da9cf84` |
| `preparation_summary.yaml` | `2f629bc48494bcc50874f3df52e60ef4246adcaa68bb8ef8d45700756253644b` |

The compact machine-readable execution record is
`preprocessing/ndispark_only_20260727.yaml`.

## Reproduction

```powershell
$env:PARKING_DATA_ROOT = "D:\datasets\ndispark\extracted"
$env:PARKING_OUTPUT_ROOT = "D:\parking_generated\ndispark_only_20260727"

python scripts\prepare_ndispark.py `
  --protocol configs\ndispark_only_dataset_frozen_20260727.yaml `
  --source-root $env:PARKING_DATA_ROOT `
  --output-root $env:PARKING_OUTPUT_ROOT
```

The command refuses to overwrite an existing output directory.

Once count predictions exist under a separately frozen Stage E/I rule:

```powershell
python scripts\evaluate_count_predictions.py `
  --truth-manifest data\manifests\ndispark_only_20260727\ndispark_test_frozen_20260727.csv `
  --predictions predictions.csv `
  --output count_metrics.json
```

The count evaluator reports MAE, RMSE, mean predicted/true counts, and
per-camera metrics. It deliberately omits MAPE and never describes counting
scores as detection mAP.

## Remaining gate

No new model prediction or training was run. Stage E must now validate and
freeze the common D0/D1/D2 comparison implementation, including identical
validation images, class semantics, input size, IoU definitions, version and
weights metadata. Only then may the three-epoch local D1 smoke run start.
