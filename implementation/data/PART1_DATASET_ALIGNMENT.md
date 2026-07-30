# Part I Dataset Alignment Audit

Audit ID: `PART1-DATASET-ALIGNMENT-20260727`  
Access date: 27 July 2026  
Machine-readable record: `part1_dataset_alignment.yaml`

## 1. Outcome

The Part I datasets do not form one interchangeable pool. They support three
different truth types:

1. **vehicle boxes/counts**: CARPK/PUCPR+ and NDISPark;
2. **parking-slot geometry and occupied/vacant state**: PKLot and
   CNRPark-EXT;
3. **general low-light/adverse-condition tracking or detection**: LMOT and
   AODRaw.

PSEX is relevant to the panoramic-parking paper, but no independently
downloadable, licensed dataset package or source-safe split was verified.
VIRAT and Grand Bassin remain Part II support data, not Part I core
parking-occupancy datasets.

The recommended scientific combination from the audit is:

```text
CARPK train (+ balanced NDISPark train)
  -> source-grouped CARPK development + NDISPark night development
  -> untouched CARPK box test + NDISPark count-only night test

PKLot day/camera groups
  -> slot classifier and slot-occupancy development
  -> never-used date/image groups for slot-occupancy test

CNR-EXT
  -> frozen historical external result only

VIRAT / Grand Bassin
  -> case-study or development-stability evidence only
```

The user subsequently selected the **backup local NDISPark-only route** on
2026-07-27. CARPK is therefore deferred and its EULA is not a blocker for the
active protocol. The PKLot full-archive download was stopped with its
2,760,421,376-byte resumable partial file preserved; that incomplete file is
not a dataset source. The Zenodo API record returned `license.id=odc-by` and
`access_right=open`, resolving the NDISPark license gate. No prediction or
training command has been run under the new protocol.

## 2. Evidence and truth rules

- Official dataset pages, official papers, official repositories, and
  deposited records are primary evidence.
- “Used in a paper” does not mean “publicly available.”
- “Downloadable” does not define permission to reuse.
- The license of an article or repository code is not automatically the
  license of its dataset.
- A parking-slot polygon plus an occupied label is not a vehicle bounding box.
- A count is not box ground truth and cannot produce detection mAP.
- Machine-generated preannotations remain preannotations until manually
  corrected and reviewed.
- Unknown fields are recorded as `unknown`.
- Raw-data locations are supplied through `--source-root` or
  `PARKING_DATA_ROOT`; generated locations use `--output-root`. No committed
  configuration may contain a user-specific absolute path.

## 3. Local holding audit

The ignored data in the external original workspace were checked without
copying them into this worktree.

| Dataset | Local state | Verified local evidence | Consequence |
|---|---|---|---|
| PKLot | Partial; acquisition paused after route change | 266,961,322-byte metadata file, 27 consumed development images, and a 2,760,421,376-byte incomplete/resumable archive | Historical development only; no new untouched slot test |
| CNR-EXT | Present, consumed | 1,104,364,544-byte full-image archive, 18,132,695-byte official CSV, 4,081 evaluated frames and 144,965 slot records | Preserve frozen external result; no retuning |
| NDISPark | Present and source-verified | 118,187,828-byte archive; 112 train, 30 validation, 117 test images; 2,577/725 train/validation boxes | ODC-By/open; ready as an input to the Stage C protocol |
| CARPK/PUCPR+ | Absent | No archive or extracted data found | EULA and download required |
| PSEX | Absent | No public package found | Excluded |
| LMOT | Absent | No local release found | Deferred |
| AODRaw | Absent | No local release found | Excluded from download |
| VIRAT | Partial and viewed | Screened videos, annotations, and case-study outputs exist | Case study only |
| Grand Bassin | Present | 1,354 raw files / 943,128,056 bytes plus local derived annotations | Development stability only |

## 4. Dataset comparison

### 4.1 Summary

| Dataset | Part I relationship | View | Primary truth | Official split | Local | Main decision |
|---|---|---|---|---|---|---|
| PKLot | P17 APSD-OC, P18 Improved MobileNetV3, P22/P23 | Fixed high-angle CCTV, sRGB | Slot polygons and occupied/vacant labels | Day-disjoint 50/50 train/test per camera in the dataset paper | Partial | Primary slot data after full archive acquisition |
| CNRPark-EXT | P17 APSD-OC, P18 Improved MobileNetV3 | Nine fixed views for CNR-EXT | Slot patches/states; camera-space coordinates | Project must group by camera/date | Present, consumed | Historical external result only |
| CARPK/PUCPR+ | ICCV 2017 counting/localization method category | CARPK drone; PUCPR+ fixed 10th-floor view | Vehicle boxes and counts | 989/459 and 100/25 train/test | Absent | Primary candidate box dataset after EULA |
| NDISPark | Detection, segmentation, counting, day-to-night domain shift | Seven surveillance views | Boxes/masks for train/validation; counts for test | Day train, night validation/test | Present | Small detector source and count-only low-light test |
| PSEX | P24 panoramic parking paper | Vehicle AVM panoramic images | Parking corners and occupancy-related detection classes | Paper reports random 4,800/1,200 experiment split | Absent/unavailable | Exclude from formal use |
| LMOT | P2 low-light MOT | Ground-level outdoor RAW/sRGB pairs | Boxes and persistent track IDs | Train/validation/test/real; only train/validation currently released | Absent | Reference only |
| AODRaw | P7 adverse-condition detection | Mixed indoor/outdoor RAW and sRGB | COCO boxes and condition tags | 5,445 train / 2,340 test | Absent | Exclude: mismatch, size, license ambiguity |
| VIRAT Ground 2.0 | Part II surveillance support | Fixed ground surveillance video | Tracks and activities; no slot truth | Not a slot-occupancy split | Partial, viewed | Existing case studies only |
| Grand Bassin Traffic | Part II temporal support | Fixed-looking high aerial | Machine-generated boxes; incomplete local slot truth | No official project-task split | Present | Development stability only |

### 4.2 Task-fit matrix

Legend: **P** primary, **S** suitable/supporting, **C** conditional, **N**
not suitable.

| Dataset | Detector fine-tuning | Slot classifier | Slot occupancy | Low-light/adverse | Temporal/tracking |
|---|---:|---:|---:|---:|---:|
| PKLot | N | **P** | **P** | S (weather, no night) | N |
| CNRPark-EXT | N | C (consumed here) | C (historical only) | S | N |
| CARPK | **P** | N | N | N | N |
| PUCPR+ | C (source overlap risk) | N | N | S (weather only) | N |
| NDISPark | **P/S** (small) | N | N | **P** for night count test | N |
| PSEX | N for vehicle boxes | C if released | C if released | S synthetic only | N |
| LMOT | S, domain-mismatched | N | N | S for MOT research | **P** for general low-light MOT, not slot transitions |
| AODRaw | S in principle, weak project fit | N | N | **P** as a general benchmark | N |
| VIRAT | N for current protocol | N | C with manual truth | N | C case study |
| Grand Bassin | N until boxes are manually corrected | N | N | N | S stability development |

## 5. Per-dataset audit

### 5.1 PKLot

- **Part I category:** core fixed-camera parking-slot classification and
  geometry-based occupancy; used by APSD-OC and Improved MobileNetV3.
- **Official sources:** [official page](https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/),
  [official README](https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf),
  and [dataset paper](https://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf).
- **Version/access:** official archive without a semantic version; checked
  2026-07-27.
- **License:** CC BY 4.0.
- **Scale:** 12,417 full 1280×720 images; 695,899 manually checked slot
  instances; three views (UFPR04, UFPR05, PUCPR); more than 30 acquisition
  days; approximately five-minute sampling. The official directory index
  reports `PKLot.tar.gz` as 3,860,376,865 bytes.
- **Annotations:** slot polygons and occupied/vacant labels, with
  sunny/overcast/rainy grouping. No vehicle boxes, masks, or persistent track
  IDs.
- **Official split:** the paper defines 50% train and 50% test for each view
  and requires all images from one day to remain on one side.
- **Leakage:** severe if individual images or slot patches are randomized;
  the same parked vehicle can persist across adjacent snapshots.
- **Fit:** primary for E1b-style classifier training and slot-occupancy
  evaluation; unsuitable for detector mAP or contiguous temporal metrics.
- **Manual work:** none for the documented slot tasks; new human vehicle boxes
  would be required for detector training.
- **Risks:** all three views have already informed development. A newly frozen
  test may use never-used dates/images but cannot be described as a new-camera
  external holdout. The official README says 12,417 images while an older
  project manifest says 12,416; the acquired archive must resolve that
  one-image discrepancy.
- **Decision:** acquire the full archive only with user approval, then freeze
  camera/date/image manifests before new P0/P1/P2 predictions.

### 5.2 CNRPark-EXT

- **Part I category:** core static slot classification/occupancy; used by
  APSD-OC and Improved MobileNetV3.
- **Official source:** [CNRPark+EXT](https://cnrpark.it/), checked 2026-07-27.
- **License:** ODbL 1.0.
- **Scale:** roughly 150,000 occupied/vacant slot patches from a 164-space
  lot. CNRPark contains about 12,000 patches from two cameras; CNR-EXT adds
  nine views under varied weather, lighting, shadows, and occlusion. The local
  external run covered 4,081 frames and 144,965 slot records.
- **Annotations:** occupied/vacant slot truth and camera-space coordinates.
  No vehicle boxes or persistent IDs.
- **Split/leakage:** camera/date groups must remain intact; random slot-patch
  splits leak nearly identical conditions.
- **Fit:** strong static slot dataset, but already consumed by this project.
  It cannot provide temporal transitions.
- **Manual work:** vehicle boxes would require separate human annotation.
- **Risks/decision:** preserve E0 and other frozen results. Any new model run is
  explicitly post-hoc and cannot select thresholds, epochs, fusion gates, or
  mapping rules.

### 5.3 CARPK and PUCPR+

- **Part I category:** high-angle vehicle localization/counting, useful for
  parking-domain detector fine-tuning; not slot occupancy.
- **Official sources:** [LPN project/dataset page](https://lafi.github.io/LPN/)
  and [ICCV 2017 paper](https://openaccess.thecvf.com/content_ICCV_2017/html/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.html),
  checked 2026-07-27.
- **License/access:** academic research only under the project EULA. The
  password-protected combined download is approximately 2 GB. PUCPR+ images
  retain the original owners' copyright.
- **Scale/annotations:**

  | Subset | Images | Boxes | View | Distributed split |
  |---|---:|---:|---|---|
  | CARPK | 1,448 | 89,777 | Drone at about 40 m over four lots | 989 train / 459 test |
  | PUCPR+ | 125 | 16,456 | Fixed 10th-floor PUCPR view | 100 train / 25 test |

- **Truth:** one vehicle bounding box per car; counts can be derived. No slot
  polygons, occupied/vacant labels, masks, or track IDs.
- **Leakage:** frames from the same flight/lot are near duplicates. Exact
  source and sequence membership must be audited from the obtained archive
  before deriving validation from the official training partition.
- **Fit:** CARPK is the strongest reviewed parking-domain box source.
  PUCPR+ is view-relevant but overlaps the PKLot PUCPR source.
- **Manual work:** none for detection/counting; slot occupancy would require
  new slot and state truth.
- **Risks/decision:** use CARPK by default. Do not use PUCPR+ in the primary
  detector protocol because detector exposure to the same source would
  contaminate a later PKLot/PUCPR occupancy assessment.

### 5.4 NDISPark

- **Part I category:** vehicle detection, instance segmentation, counting,
  day-to-night domain shift.
- **Official sources:** [Zenodo version 1.0.0](https://zenodo.org/records/6560823)
  and [CNR AIMH page](https://aimh.isti.cnr.it/ndispark/), checked 2026-07-27.
- **License:** ODC Attribution 1.0. The human-facing Rights field did not
  render a value during the first audit, so the official Zenodo API was
  checked; record 6560823 returned `license.id=odc-by` and
  `access_right=open` on 2026-07-27.
- **Scale:** 259 images from seven cameras. The checked archive is
  118,187,828 bytes.
- **Annotations and official split:**

  | Split | Images | Truth | Lighting | Current role |
  |---|---:|---|---|---|
  | Train | 112 | 2,577 manual boxes/masks; centroids | Day | Fine-tuning candidate |
  | Validation | 30 | 725 manual boxes/masks; centroids | Night | Consumed development validation |
  | Test | 117 | Vehicle count only | Night | Count MAE/RMSE test only |

- **View:** fixed surveillance views with varied angles, weather, shadows, and
  partial occlusion; sRGB.
- **Leakage:** camera IDs encoded in filenames must be audited because official
  day/night splits may retain the same physical camera across conditions.
- **Fit:** highly relevant small fixed-camera detector source and low-light
  evaluation. It has no slot geometry/state or track IDs.
- **Manual work:** none for train/validation detector metrics. The test split
  needs manual boxes before any mAP claim.
- **Risks/decision:** validation remains development data. Test reports MAE,
  RMSE, mean predicted count, mean true count, and per-camera results—not mAP
  and not MAPE by default.

### 5.5 PSEX

- **Part I category:** P24 panoramic/AVM parking-corner and occupancy-related
  detection with GAN-generated adverse-weather styles.
- **Official sources:** [Sensors article](https://www.mdpi.com/1424-8220/25/20/6449)
  and [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12568149/),
  checked 2026-07-27.
- **Availability/license:** no official repository or downloadable PSEX
  package was found. The article is CC BY, but that does not establish a
  separate dataset license. The Data Availability Statement says the data are
  contained within the article.
- **Scale:** derived from PS2.0 using GAN style transfer for fog, snow,
  sandstorm, and rain. The article reports 6,000 generated images and an
  experimental 6,000-image sample (3,000 synthetic + 3,000 original), split
  randomly into 4,800 train and 1,200 test. The full PSEX corpus and annotation
  count are unknown.
- **View/annotations:** vehicle-surround-view panoramic images; T/L parking
  corner boxes and parallel/vertical occupied/free classes. This differs from
  fixed overhead CCTV and from vehicle-box detection.
- **Leakage:** the original/synthetic relationship and random split can put
  style variants of the same source image on both sides; public image IDs are
  unavailable for verification.
- **Decision:** exclude from formal training and evaluation. Keep the paper as
  a method reference only.

### 5.6 LMOT

- **Part I category:** P2 low-light multi-object tracking, not parking-slot
  occupancy.
- **Official sources:** [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Multi-Object_Tracking_in_the_Dark_CVPR_2024_paper.html)
  and the [current official repository](https://github.com/xinzwang/LMOT),
  checked 2026-07-27.
- **License:** dataset CC BY-NC 4.0 for non-commercial research; repository
  code MIT.
- **Scale/view:** the paper describes 32 videos, 35,120 frames, 815,550 boxes,
  and 4,090 trajectories. The repository lists paired RAW/sRGB 1800×1000
  videos at 20 FPS across city roads, overpasses, pedestrians, and
  intersections.
- **Annotations/splits:** six moving classes with persistent IDs in MOT
  format. Repository statistics list 11/4/11 LMOT-dual train/validation/test
  videos and six LMOT-real videos, but explicitly state that only train and
  validation are currently released.
- **Leakage:** paired well-lit/dark versions must stay in the same split.
- **Cost:** official download size is unknown and access is through Baidu
  Drive.
- **Decision:** do not download for the core project. It supports the
  low-light tracking discussion, not slot-transition truth.

### 5.7 AODRaw

- **Part I category:** P7 RAW/sRGB detection under diverse conditions.
- **Official sources:** [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Towards_RAW_Object_Detection_in_Diverse_Conditions_CVPR_2025_paper.html)
  and [official repository](https://github.com/lzyhha/AODRaw), checked
  2026-07-27.
- **License:** unknown for the dataset. The repository explicitly licenses
  **code** CC BY-NC-SA 4.0 but does not say that the same license covers the
  images and annotations.
- **Scale:** 7,785 high-resolution images, 135,601 instances, 62 categories,
  nine conditions; 5,445/94,949 train images/instances and
  2,340/40,652 test images/instances.
- **Storage:** 435 GB original RAW+sRGB, 223 GB downsampled RAW, 4.3 GB
  downsampled sRGB, 439 GB sliced RAW, or 23 GB sliced sRGB.
- **Annotations/view:** COCO boxes and condition tags across mixed
  indoor/outdoor RAW and sRGB scenes; no slot states or persistent IDs.
- **Leakage:** every derivative crop/downsample must inherit its original
  image split.
- **Decision:** do not download. It is large, non-parking, multi-category,
  license-ambiguous, and not necessary to train YOLOv8n on a parking-domain
  box source. Its use of modified MMDetection also does not justify installing
  MMDetection/MMCV here.

### 5.8 VIRAT Ground 2.0 — Part II support

- **Official source:** [VIRAT project](https://viratdata.org/), checked
  2026-07-27.
- **Terms:** individual VIRAT Usage Agreement with redistribution and PII
  duties.
- **Scale/view:** approximately 8.5 hours across 11 fixed outdoor surveillance
  scenes; full original-video collection approximately 37.63 GB.
- **Truth:** tracked objects and activities, but no parking-slot polygons or
  per-slot occupied/vacant state.
- **Decision:** existing screened scenes, including VIRAT 0503, are case
  studies. A future temporal test would require a different physical scene
  and human-frozen slot/interval/transition truth before any prediction.

### 5.9 Grand Bassin Traffic — Part II support

- **Official source:** [deposited record](https://doi.org/10.5281/zenodo.21321449),
  checked 2026-07-27.
- **License:** CC BY-NC-SA 4.0.
- **Scale/view:** 4,000 supplied 1280×720 frames at 2 FPS from 24 fixed-looking
  high-aerial surveillance segments.
- **Truth:** 854,646 supplied boxes are machine-generated. The local reviewed
  seven-slot subset contains occupied intervals but no verified vacant
  transitions or persistent IDs.
- **Decision:** retain as temporal stability development evidence only. Do not
  train a detector from its preannotations or report detector mAP without
  human correction.

## 6. Dataset-to-task decision

| Required role | Decision | Status and restriction |
|---|---|---|
| **A. detection_train** | CARPK official train + balanced NDISPark train | Blocked on CARPK EULA/download |
| **B. detection_validation** | Source-grouped development carved only from CARPK train, plus NDISPark night validation as a separate stratum | Freeze exact lot/sequence IDs after acquisition; NDISPark validation is already development |
| **C. detection_test** | CARPK official 459-image box test; NDISPark 117-image count-only test | CARPK absent; NDISPark test cannot report mAP |
| **D. slot_classifier_train** | Preserve existing E1b checkpoint; retrain only from PKLot official day-disjoint training groups if justified | Existing checkpoint available; full PKLot images absent |
| **E. slot_occupancy_development** | PKLot camera/date groups excluding all future test dates/images | Requires full archive and usage-history reconciliation |
| **F. slot_occupancy_test** | Never-used PKLot date/image groups frozen before P0/P1/P2 predictions | Currently unavailable; cannot use CNR-EXT as a substitute |
| **G. temporal_test** | None currently | VIRAT remains a viewed case study; no fabricated truth |
| **H. low_light_test** | NDISPark 117-image night count-only test | MAE/RMSE and per-camera counts only; no mAP |

## 7. Recommended and backup combinations

### Recommended

Use **CARPK + NDISPark + PKLot** with source balancing:

- CARPK supplies the main parking-domain vehicle boxes and untouched box test.
- NDISPark supplies fixed-surveillance day training examples, night
  development boxes, and a count-only night test.
- PKLot supplies slot-classifier and slot-occupancy truth.
- PUCPR+ stays out of the primary detector protocol because it shares the
  PKLot PUCPR source.
- CNR-EXT remains historical external evidence.

At Stage C, all accepted detection classes map to one `vehicle` class, and
the manifest must report images and box counts per source. A source/lot/video
may belong to only one split. Sampling must prevent CARPK from completely
swamping the 112-image NDISPark train source.

### Backup without new data download

Use NDISPark train/validation/test only for a small local fine-tuning flow:

- 112 day images for training;
- 30 night images for development validation;
- 117 night images for count-only test.

This is acceptable for a smoke run and for demonstrating the professor's
preprocess → pretrained baseline → fine-tune → evaluate workflow. It is not a
strong final benchmark: the training set is tiny, the only box-labelled night
split has already been used for development, and no untouched box-metric or
slot-occupancy test is created.

## 8. Exclusions

| Dataset | Exclusion/defer reason |
|---|---|
| PUCPR+ | Same physical source family as PKLot PUCPR; creates detector-to-occupancy leakage risk |
| CNRPark-EXT | Already consumed; retain negative and positive historical results without retuning |
| PSEX | No verified release, dataset license, annotation package, hashes, or source-safe split |
| LMOT | Non-parking MOT; only train/validation released; download size unknown |
| AODRaw | Dataset license unclear; 4.3–439 GB options; RAW/multi-category task mismatch |
| VIRAT | No slot truth; local material viewed; case study only |
| Grand Bassin | Machine-generated boxes and incomplete occupied-only local slot truth |

## 9. Stage B gate

Stage B itself is complete: required datasets, source/license status, local
holdings, annotation compatibility, leakage risks, task mapping, recommended
combination, backup, and exclusions are recorded in both Markdown and YAML.

The user selected the local NDISPark-only backup route on 2026-07-27. Its
local source and license were verified, and Stage C subsequently froze it as
`DPROTO-NDISPARK-ONLY-20260727-01` without CARPK or a complete PKLot archive.
Stage D preprocessing must pass before detector training; new D0/D1/D2
predictions remain gated on validation of the Stage E comparison
implementation.

This choice accepts the backup route's limitations: the 30-image box-labelled
split remains consumed development validation, the 117-image test supports
count MAE/RMSE rather than detection mAP, and there is no new untouched
slot-occupancy test.

Stage C evidence:

- `../configs/ndispark_only_dataset_frozen_20260727.yaml`;
- `NDISPARK_ONLY_DATASET_CARD.md`;
- `manifests/ndispark_only_20260727/`, containing 112 train, 30 validation,
  and 117 test rows plus source and frozen-artifact hashes.
