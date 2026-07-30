# Stage Q-v2 UPM-GTI Test Archive Structure Audit

Date: 2026-07-29

Protocol:
`STAGE-Q-V2-UPM-GTI-EXTERNAL-NIGHT-OCCUPANCY-20260729-01`

Status: **NIGHT_TEST_GATE_PASS; BLOCKED_PENDING_HUMAN_POLYGON_CONFIRMATION**

No D1, D1-LL, E1b, tracker, or other model was loaded during this audit.

## Source and use boundary

- Dataset: ETSIT / UPM-GTI Parking Lot Occupancy Database
- Official page:
  <https://gti.ssr.upm.es/data/parking-lot-database>
- Official public storage:
  <https://drive.upm.es/index.php/s/TdqfDr25NAsGIea>
- Associated paper:
  Leyre Encío et al., “Visual Parking Occupancy Detection Using Extended
  Contextual Image Information via a Multi-Branch Output ConvNeXt Network,”
  *Sensors* 23(6), 3329, 2023,
  <https://doi.org/10.3390/s23063329>.
- Authorized archive: `test.zip`
- Official public download: true
- Explicit dataset license found: false
- Use scope: local, non-commercial course research
- Redistribution: prohibited by project policy
- Attribution required: true
- Legal interpretation claimed: false

The article is CC BY 4.0, but that license is not asserted here as the image
archive's license.

## Download and archive integrity

The official public share was accessible without login or a new terms
prompt. The server's direct public WebDAV response identified the file as
`application/zip` and reported:

- content length: `250,698,837` bytes;
- SHA-1:
  `9e462c0720eddf92bb11b4eed7d5e0e597112a5f`;
- MD5:
  `3157555f948f225621bc618656b60e75`;
- last modified: 2020-10-23 11:44:29 GMT.

The downloaded file has:

- exact bytes: `250,698,837`;
- SHA-256:
  `92d61d8f87fe3e7068d8c42ce8dc2c415c08071c92eeddfd4d47260e8922efdc`;
- SHA-1 and MD5 equal to the server checksums;
- 3,259 ZIP entries;
- 3,206 files and 53 directory entries;
- 250,835,482 uncompressed member bytes;
- one archive root, `test/`;
- no path traversal, symbolic link, encrypted member, duplicate
  case-insensitive path, or CRC failure.

The raw archive remains unmodified in a Stage Q-v2-specific Git-ignored
directory. It is not part of the artifact submission payload.

## Extracted structure

Safe extraction produced:

- 26 `gopro` directories;
- 26 `groundtruth.txt` files;
- 3,180 JPEG images;
- one image resolution: `800x600`;
- 3,148 truth records;
- 66,108 binary slot labels:
  28,858 occupied-source labels (`source 0`) and
  37,250 vacant-source labels (`source 1`).

Every parsed vector has exactly 21 binary values. Source semantics are:

- source `1` = available/vacant = project state `0`;
- source `0` = not available/occupied = project state `1`;
- unknown = excluded, although this archive contains zero unknown values.

Twenty-five sequences have an exact raw image/truth bijection. `gopro10`
contains 67 images but only 35 truth rows: 32 images have no truth, while
every truth row has an image. The full `gopro10` sequence is excluded from
the frozen external test rather than silently dropping its unlabelled
images.

The truth-listed images are naturally sortable by GoPro file name. No
timestamp or reliable FPS accompanies the archive. Therefore no
seconds-level transition latency can be calculated or inferred.

Across the complete truth streams, natural filename ordering contains:

- 549 adjacent frames with at least one state change;
- 951 individual slot state changes.

These are index-level changes, not real-time arrival/departure timing.

## Night/low-light evidence and deterministic selection

The fixed sequence contact sheet was rendered before any model output was
viewed. It shows a real parking lot with a stable shared fisheye camera view,
including daylight, twilight, and clearly illuminated night images. The
official paper's Figure 1 and Figure 7 also describe ETSIT low-light
examples.

The source directories mix lighting conditions. Therefore a complete
`gopro` directory is not labelled wholesale as “night.” Before full
per-image selection, the auxiliary image-only rule was fixed as:

`mean grayscale luminance <= 70.0`

Brightness is used only to define an obviously low-light subset; no model
prediction participates. A sequence qualifies only if:

1. it belongs to the official Test archive;
2. its raw image pool has an exact truth bijection;
3. it shares the visually verified stable 800x600 geometry;
4. it has at least one truth-labelled frame satisfying the frozen
   luminance rule;
5. its selected frames collectively contain both occupied and vacant
   labels.

All qualifying sequences sharing this geometry are retained. Seventeen
sequences pass:

`gopro1`, `gopro4`, `gopro5`, `gopro8`, `gopro9`, `gopro11`, `gopro12`,
`gopro19`, `gopro23`, `gopro24`, `gopro25`, `gopro26`, `gopro30`,
`gopro33`, `gopro34`, `gopro36`, and `gopro46`.

The frozen manifest contains:

- 376 selected low-light images;
- 7,896 per-slot truth rows;
- 50 index-adjacent selected frames with a state change;
- 90 selected slot state changes;
- zero unknown labels;
- logical manifest SHA-256:
  `d4d391fd0ad11f5c03f1f44edb268df9e5989da5ad413b595d15c98f12f9791e`;
- CSV SHA-256:
  `8929e6a38b36b578ae2658127625576e632904437d0ba5d2f37470fc0b0746ba`.

Because selected frames can have gaps in their original low-frame-rate
streams, these changes support only state-change agreement and frame-index
differences. They do not support seconds-level or real-time latency.

## Official slot numbering and current blocker

The associated paper's Figure 4(a) numbers the 21 ETSIT spaces from `0` to
`20`. Stage Q-v2 therefore uses `slot_00` through `slot_20` in exact vector
index order.

A pre-model polygon draft was drawn from:

1. official Figure 4(a); and
2. the empty 800x600 Test frame
   `test/gopro32/images/106GOPRO-GOPR1524.JPG`.

The polygons are convex, nonzero-area, within image bounds, and rendered in
`STAGE_Q_V2_POLYGON_VALIDATION_20260729.png`. No detector or classifier
prediction was viewed or used.

Formal inference remains blocked until the user visually confirms that:

- labels `00`–`06` follow the left row;
- labels `07`–`11` follow the foreground row;
- labels `12`–`20` follow the right diagonal row;
- every polygon covers the physical space bearing the same number in
  official Figure 4(a).

Until that confirmation, no formal protocol config or model result may be
created.

