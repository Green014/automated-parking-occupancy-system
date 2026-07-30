# NDISPark-only frozen dataset card

Protocol ID: `DPROTO-NDISPARK-ONLY-20260727-01`  
Frozen: `2026-07-27T15:53:42.1557501+08:00`  
Status: frozen for preprocessing; no new model prediction has been run

## Purpose and decision

This is the user-selected backup detector route for Part II. It supports a
small, local, parking-domain YOLOv8n fine-tuning exercise without a new large
download:

```text
112 daylight box-labelled images
  -> NDISPark-fine-tuned YOLOv8n
  -> 30 night box-labelled images for consumed development validation
  -> 117 night images for count-only test
```

It does **not** create a new box-labelled detection test or an untouched
slot-occupancy test. The stronger CARPK + NDISPark recommendation remains
documented in the Part I alignment audit but is deferred.

## Source and permission

- Dataset: NDISPark version 1.0.
- Official record: <https://zenodo.org/records/6560823>.
- DOI: <https://doi.org/10.5281/zenodo.6560823>.
- License: Open Data Commons Attribution License 1.0 (ODC-By).
- Archive: `ndis_park.zip`, 118,187,828 bytes.
- Archive SHA-256:
  `87ca20dfe5e5a5659a9a41e03724fdc38eed050de6ed6742995955fc0bd785c0`.

The raw archive and images are ignored by Git. The project commits only
protocols, hashes, and compact manifests. Reusers must obtain the dataset from
the official source and comply with its attribution terms.

## Composition and truth

| Split | Images | Truth | Vehicles/boxes | Role |
|---|---:|---|---:|---|
| train | 112 | vehicle bounding boxes | 2,577 boxes | fine-tuning |
| validation | 30 | vehicle bounding boxes | 725 boxes | consumed development validation |
| test | 117 | image-level vehicle count | 1,402 vehicles total | count-only test |

All train and validation instances use source COCO category ID 3, `car`. The
preprocessing target is one project class, ID 0, `vehicle`. The slot polygons
from PKLot/CNRPark are not used as vehicle boxes.

Test counts contain no zero-count image in this release. The implementation
must nevertheless define zero-truth behavior: MAE and RMSE remain valid,
while MAPE is not reported by default.

## Per-camera composition

| Camera | Train images / boxes | Validation images / boxes | Test images / true count |
|---|---:|---:|---:|
| 60 | 21 / 804 | 6 / 244 | 18 / 275 |
| 62 | 20 / 551 | 2 / 109 | 0 / 0 |
| 64 | 21 / 669 | 10 / 203 | 31 / 635 |
| 69 | 12 / 143 | 2 / 27 | 19 / 114 |
| 73 | 12 / 206 | 4 / 94 | 18 / 147 |
| 78 | 14 / 124 | 3 / 28 | 16 / 119 |
| 83 | 12 / 80 | 3 / 20 | 15 / 112 |

## Split and leakage audit

Official membership is preserved. Image SHA-256 auditing found:

- zero missing or undecodable images;
- zero invalid or out-of-bounds train/validation boxes;
- zero exact duplicate-image groups across splits.

This is not a camera-independent split. Six cameras occur in train,
validation, and test; camera 62 occurs in train and validation. The intended
domain shift is daylight training to night validation/test. Report results
per camera and do not claim generalization to new cameras.

The 30-image validation split was already used in historical D0/D2
comparisons. It remains development data and may select the operating rule,
but it is not an untouched test. The 117 test images have counts only; they
cannot support detector mAP, box precision/recall, false-positive box
montages, or false-negative box montages.

## Allowed and prohibited uses in this project

Allowed:

- initialize D1 from the verified COCO-pretrained `yolov8n.pt`;
- fine-tune on the official 112-image train split;
- compare D0/D1/D2 on the same 30-image development validation split;
- select the detector and one counting operating rule on development data;
- report MAE, RMSE, mean counts, and per-camera count metrics once on test.

Prohibited:

- moving validation or test images into training;
- using test counts to select confidence, epochs, augmentation, or weights;
- calling count metrics detection mAP;
- using CNR-EXT, PKLot, or VIRAT to tune D1;
- presenting the backup route as a new final slot-occupancy benchmark.

## Reproducing the source freeze

Set the source and verification-output roots outside committed raw/generated
directories. The freeze tool refuses to overwrite an existing artifact:

```powershell
$env:PARKING_DATA_ROOT = "D:\datasets\ndispark\extracted"
$env:PARKING_VERIFY_ROOT = "D:\parking_protocol_verification\ndispark"

python scripts\freeze_ndispark_protocol.py `
  --source-root $env:PARKING_DATA_ROOT `
  --archive-path "D:\datasets\ndispark\ndis_park.zip" `
  --output-root $env:PARKING_VERIFY_ROOT `
  --protocol-id DPROTO-NDISPARK-ONLY-20260727-01 `
  --frozen-at "2026-07-27T15:53:42.1557501+08:00"
```

Compare the reproduced CSV SHA-256 values with
`data/manifests/ndispark_only_20260727/ndispark_source_manifest_frozen_20260727.yaml`.
No user-specific absolute source path is committed.

## Known limitations

- The training set is only 112 images and 2,577 car boxes.
- Camera identity overlaps all evaluation roles.
- Validation has already been consumed as development evidence.
- Test has counting labels but no box truth.
- NDISPark has no parking-slot polygons or occupied/vacant labels.
- The backup route cannot establish that a detector mAP change improves
  slot-level Macro F1.
