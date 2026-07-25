# Dataset Manifest

Audit date: 24 July 2026

This manifest separates suitability for slot occupancy, vehicle detection, and
tracking. "Publicly downloadable" is not treated as a license. A source enters
the main experiments only when its use terms can be cited.

## Selection summary

| Dataset | License/use terms | Fixed/high view | Continuous | Slot truth | Vehicle/track truth | Decision |
|---|---|---:|---:|---:|---:|---|
| PKLot | CC BY 4.0 | Yes | No; about 5-minute intervals | Polygon + occupied/vacant | No track IDs | Primary slot evaluation |
| Grand Bassin Traffic | CC BY-NC-SA 4.0 | Fixed-looking high aerial surveillance view | Yes within each segment; frames supplied at 2 FPS | No slot truth | Machine-generated COCO boxes; no persistent IDs | Continuous stability development only |
| CNRPark+EXT | ODbL 1.0 | Yes | No | Slot patches/labels; camera views | No track IDs | Primary slot/robustness evaluation |
| NDISPark | ODC Attribution 1.0 | Seven surveillance views | No; about 250 images | No slot states | Vehicle boxes/masks | Detection/night failure analysis |
| EPFL Multi-view Multi-class | Copyright; explicit free research use with citation | Six fixed calibrated cameras; several elevated | Yes, 23:57 at 25 FPS | No | Boxes on 242 non-consecutive frames | Engineering/tracking source only |
| Dragon Lake Parking (DLP) | Project terms: non-commercial research/teaching, citation, no redistribution | Overhead but drone, not fixed CCTV | Yes, 3.5 h at 25 FPS | Parking map, but not slot-state labels | Dense trajectories/agents | Strong tracking extension; raw video requires request |
| Barcelona DISCO | CC BY 4.0 | Roadside study, not a parking-camera video set | No raw video in archive | Loading-zone event/status data | No usable visual tracks | Supportive event-data source only |
| KIOS multimodal parking-area data | CC BY 4.0 | Hovering drone at five waypoints | Short sequences converted to frames | No slot states | Car/person boxes | Deferred because Part 1 is 44 GB |
| AGH Parking Database | Explicit free research-community use with citation | Fixed parking entrance camera | Yes | No marked slot truth | Plate coordinates only | Small pipeline smoke source, not occupancy evaluation |
| UFPArk | Download link published in paper; license text not located | Fixed surveillance | Yes, 3.25 h | No verified slot truth | Small labelled subset described | Excluded until written use terms are verified |
| YouTube parking videos | Platform/video-specific terms | Varies | Varies | Usually none | Usually none | Not used as a primary source |

## 1. PKLot

- Official page:
  https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/
- Official download:
  http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
- Convenient public mirror used only for the small feasibility sample:
  https://huggingface.co/datasets/Voxel51/PKLot
- License: Creative Commons Attribution 4.0 International.
  https://creativecommons.org/licenses/by/4.0/
- Size/content: 12,416 full 1280x720 images, about 695,900 slot instances,
  two lots/three views, sunny/cloudy/rainy conditions, XML polygon and
  occupied/vacant labels. The official archive is about 4.6 GB.
- Temporal limitation: captures are approximately five minutes apart. They are
  time series but must not be treated as contiguous video for ByteTrack or
  frame-latency claims.
- Intended use:
  - validate polygon parsing and slot-level occupancy;
  - compare B0 and B1 on fixed views;
  - provide weather-stratified failure cases.
- Split rule: group by camera and acquisition date; never randomly split
  individual frames.
- Attribution: cite P. R. L. de Almeida et al., "PKLot - A robust dataset for
  parking lot classification," Expert Systems with Applications, 2015.

## 2. Grand Bassin Traffic

- Dataset card/download:
  https://huggingface.co/datasets/shivam11/grand-bassin-traffic
- DOI: https://doi.org/10.5281/zenodo.21321449
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
  https://creativecommons.org/licenses/by-nc-sa/4.0/
- Content: 4,000 1280x720 frames sampled at 2 FPS from 24 surveillance
  segments, divided into 2,000 aerial and 2,000 road-view frames. The supplied
  COCO file contains 854,646 machine-generated detections.
- Selected continuous development segment:
  `stream_gbassinexch1.stream_2023-02-16_16-34-07.264`, 793 frames,
  396.5 seconds. All frames decoded successfully and were reconstructed into
  an MP4 with OpenCV. Seven continuously occupied, manually checked bus bays
  are used for a positive-class temporal stability ablation.
- Important truth limitation: the COCO boxes are machine generated and are
  not treated as manual detector ground truth. The seven-slot stability subset
  has no verified arrivals/departures or vacant slots, so it supports
  false-free and flicker analysis but not false-occupied or transition-latency
  claims.
- Split rule: a complete `video_source` belongs to one split. The fine-tuning
  preparation manifest uses `13-30-33` for train, `16-34-07` for validation,
  and `16-25-12` as untouched holdout.

## 3. CNRPark+EXT

- Official page: https://cnrpark.it/
- License: Open Data Commons Open Database License 1.0.
  https://opendatacommons.org/licenses/odbl/1-0/
- Content: roughly 150,000 labelled occupied/vacant parking-space patches from
  a 164-space lot, with multiple camera views and varied weather/lighting.
- Intended use:
  - second fixed-camera slot-level dataset;
  - cross-view and condition robustness checks;
  - no tracking claims.
- Split rule: camera/day grouping. A camera/date group belongs to only one
  split.

## 4. NDISPark

- DOI/download: https://doi.org/10.5281/zenodo.6560823
- Direct record: https://zenodo.org/records/6560823
- License: Open Data Commons Attribution License 1.0.
  https://opendatacommons.org/licenses/by/1-0/
- Size/content: `ndis_park.zip`, 118.2 MB; about 250 images from seven cameras,
  day/night and occlusion conditions; vehicle masks, boxes, centroids, and
  counts.
- Intended use:
  - evaluate pretrained vehicle detection;
  - produce day/night and occlusion failure cases;
  - do not calculate slot occupancy because slot-state truth is absent.

## 5. EPFL Multi-view Multi-class Detection

- Official page:
  https://www.epfl.ch/labs/cvlab/data/data-multiclass/
- Use terms: the official page states that the copyrighted videos/images are
  free to use for research provided the listed publication is cited.
- Content: 23 minutes 57 seconds of synchronized frames at 25 FPS from six
  calibrated fixed cameras; parking slots, road, bus stop, people, cars, and
  buses; box annotations on 242 non-consecutive multi-view frames.
- Intended use:
  - continuous-video engineering and detector/tracker sanity checks;
  - no slot-level score without a separate verified annotation pass;
  - no IDF1/HOTA unless persistent IDs are confirmed in the selected truth.

## 6. Dragon Lake Parking

- Dataset record: https://doi.org/10.5061/dryad.tht76hf5b
- Project/toolkit: https://github.com/MPC-Berkeley/dlp-dataset
- Use terms: the official project page restricts the requested raw video to
  non-commercial research, teaching, publication, and personal
  experimentation; it requires citation and prohibits redistribution. Those
  project-specific terms govern the raw video even if separately deposited
  Dryad JSON records carry broader repository metadata.
- Content: 30 scenes, 317,873 frames, 5,188 agents, 15,383,737 instances;
  approximately 3.5 hours of 4K/25 FPS overhead drone video and a roughly
  400-space parking area.
- Access constraint: the 8.02 GB JSON release is directly downloadable; raw
  video and ground-truth video annotations require a research request through
  the form linked by the dataset README.
- Intended use: optional trajectory/tracking evaluation after access is
  granted. It is not the first occupancy dataset because the camera is a drone
  and slot-state truth is not provided in the required format.

## 7. Barcelona DISCO

- DOI/record: https://doi.org/10.5281/zenodo.20210588
- License: CC BY 4.0.
- Checked archive: `anonymized_disco_dataset.zip`, 77,305,088 bytes, SHA-256
  `b51b7dffa7ca969ca2d9f34a2e22937d35283df65231ad0871c1f28ba3179811`.
- Content check: 805 files (558 CSV, 200 JSON, 36 JPG, 10 PNG, and one
  Markdown file), but no raw continuous video.
- Decision: retain as a supportive loading-zone/event-data reference. It
  cannot drive the OpenCV/YOLO video experiment and is not presented as one.

## 8. KIOS Multimodal Object Detection in a Parking Area

- DOI/record: https://doi.org/10.5281/zenodo.15862267
- License: CC BY 4.0.
- Content: RGB wide/zoom and thermal footage captured while a drone hovered for
  30 seconds at five waypoints, with car/person detection labels. Part 1 stores
  44 GB of frames; later parts make the total much larger.
- Intended use: optional aerial/thermal robustness extension only. The storage
  and mismatch with fixed CCTV make it unsuitable for the first baseline.

## 9. AGH Parking Database

- Official page and downloads:
  https://qoe.agh.edu.pl/parking-database/
- Use terms: the official page makes sequences available to the research
  community free of charge and requests citation of the listed papers/page.
- Example source video:
  https://qoe.agh.edu.pl/wp-content/uploads/2021/02/agh_src1_hrc0.avi
- Checked size: 19,168,052 bytes by HTTP response header.
- Limitation: the sample camera covers a parking entrance/vehicle, not a bank of
  clearly marked slots. It is suitable only for a lightweight video I/O and
  vehicle-detection smoke test.

## 10. UFPArk - license hold

- Paper: "Public Dataset of Parking Lot Videos for Computational Vision Applied
  to Surveillance," ICMLA 2020.
- Published data link:
  https://nextcloud.lasseufpa.org/s/qxWsqNjLSgqqdHM
- Content described by the paper: approximately 3.25 hours of short fixed
  surveillance clips across morning, afternoon, and night.
- Decision: do not download or use in reported experiments until an explicit
  dataset license or written research-use permission is obtained. A paper
  saying "public" does not by itself define redistribution/reuse terms.

## Locally retained feasibility samples

These files are small source-page/sample checks, not an experimental split.

| File | Source/use | Bytes | SHA-256 |
|---|---|---:|---|
| `research_samples/pklot_2012-09-21_14-45-32.jpg` | PKLot mirror; CC BY 4.0 | 377,810 | `df6763f20814b490cdc87ae40d9d1892ddd9143252fe98878c987b9368e2215b` |
| `research_samples/epfl_c0.png` | EPFL official sample; research use | 222,449 | `2651078b2848b52aea1994e668c3d625bc556bd63f43b0565eef1ef51dd4ccf8` |
| `research_samples/epfl_c1.png` | EPFL official sample; research use | 203,512 | `9c025ad0398f241758f17e55d72f167ef2586b83418c64236f90dc339d2e0599` |
| `research_samples/epfl_c2.png` | EPFL official sample; research use | 234,777 | `3a16a1f042b6dd804b43a7664cb2da4cf1eb11be0c62e51190a22c7f7fa03db4` |
| `research_samples/agh_src1_preview.png` | AGH official preview; research use | 747,083 | `923921d88f622bc4d8ae5c7502b7e3b36a6dfe77f3965a4e920cf38d89c61e6c` |

## Local experimental downloads

Raw licensed downloads are ignored by Git; their manifests and checksums are
retained.

| Asset | Items/bytes | Integrity/use |
|---|---:|---|
| PKLot metadata | 266,961,322 bytes | SHA-256 `f2cef3361cb2698cadb59a920fc3a88504607d5f64c900aa549d24ba0aa6d37f` |
| PKLot development images | 27 images / 8,182,054 bytes | Listed in `splits/pklot_development.csv` |
| Grand Bassin `16-34-07` | 793 images / 399,758,079 bytes | Per-frame hashes in `splits/grand_bassin_aerial_development_checksums.csv` |
| Reconstructed `16-34-07` MP4 | 57,254,582 bytes | SHA-256 `5fe48eb0ef687022411b379d9f870934accb68e68dd1cb1feebbdee7b86b3a9b` |
| Grand Bassin `13-30-33` | 361 images / 191,989,846 bytes | Per-frame hashes in `splits/grand_bassin_aerial_candidate_133033_checksums.csv` |
| Reconstructed `13-30-33` MP4 | local derived video | SHA-256 `6d5b187791c579ddae38729ae534a39854f6b20a39563750b88ea51b95792815` |
| Fine-tuning preparation set | 72 selected images; 24 per video-level split | Per-frame hashes in `finetune/grand_bassin_annotation_checksums.csv` |
| NDISPark archive | 118,187,828 bytes | MD5 `2825a2403794d233c278e2532d061359`; SHA-256 `87ca20dfe5e5a5659a9a41e03724fdc38eed050de6ed6742995955fc0bd785c0` |

## Planned data layout

```text
implementation/data/
  DATASET_MANIFEST.md
  research_samples/       # tiny audited feasibility files
  raw/                     # ignored; immutable licensed downloads
    pklot/
    grand_bassin/
    cnrpark_ext/
    ndispark/
    video/
  annotations/             # slot maps and ground truth
  splits/                  # video/camera/date-level manifests
  processed/               # ignored generated clips/frames
```

Each split manifest will contain `source_id`, `camera_id`, `sequence_id`,
`date_or_clip_id`, `split`, `license`, and checksum fields.
