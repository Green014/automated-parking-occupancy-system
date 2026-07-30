# Stage Q — Independent External Night Parking Occupancy Evaluation

Date: 2026-07-29
Protocol:
`STAGE-Q-INDEPENDENT-EXTERNAL-NIGHT-OCCUPANCY-20260729-01`
Final status: **BLOCKED_BEFORE_DOWNLOAD**
Formal inference: **not executed**

## Executive result

Stage Q audited the maximum permitted three official dataset candidates.
None currently satisfies all frozen requirements for a new, independent,
licensed, fixed-camera night parking test with per-slot truth.

The UPM-GTI/ETSIT Parking Lot Occupancy Database is the only structurally
promising candidate. Its official description establishes a real parking
lot, day/night data, Training/Validation/Test groups, low-frame-rate image
sequences, 21-value per-image occupancy vectors, and an official numbered
slot illustration. However, the official dataset and storage pages do not
state a dataset-image license or an explicit research-use grant. Before
archive inspection, they also do not establish that `test.zip` contains an
eligible fixed night Test sequence. Consequently, no UPM archive was
downloaded and no model was loaded.

The user subsequently authorized downloading the CNR-EXT full-frame
package. A read-only check found the exact 1,104,364,544-byte archive and an
existing extraction already present from historical experiments, so no
duplicate download was performed. CNR-EXT remains ineligible because this
project already consumed it and the official source does not establish a
qualifying night test.

Stage P2 remains **FAIL**, Stage P4 remains unexecuted, D1 remains the P3
default detector, and D1-LL remains a secondary frozen comparison only.
Stage Q neither reopens nor reinterprets Stage P.

## Q0 — official online audit

### Candidate 1: UPM-GTI/ETSIT Parking Lot Occupancy Database

- Official dataset page:
  <https://gti.ssr.upm.es/data/parking-lot-database>
- Official public storage:
  <https://drive.upm.es/index.php/s/TdqfDr25NAsGIea>
- Associated paper:
  Leyre Encío et al., “Visual Parking Occupancy Detection Using Extended
  Contextual Image Information via a Multi-Branch Output ConvNeXt Network,”
  *Sensors* 23(6), 3329, 2023,
  <https://doi.org/10.3390/s23063329>.
- Public share status: accessible without an account prompt during this
  audit.
- Displayed archives:
  `test.zip` 239.1 MB, `validation.zip` 239.7 MB,
  `training.zip` 1.9 GB, total 2.3 GB.
- Media: low-frame-rate image sequences, not established as real-time video.
- Truth: one 21-value vector per image; official semantics are
  `1=available/vacant` and `0=not available/occupied`.
- Polygon support: no machine-readable polygon file is stated; the official
  paper supplies the numbered 21-slot visual mapping.
- Time support: no reliable FPS or timestamp is stated.
- Prior project use: none found.
- Gate: **BLOCKED**.

Blocking facts:

1. The image archive's license/research-use permission is not explicitly
   stated on the official dataset or storage page. The paper's CC BY 4.0
   license does not by itself license the image archive.
2. A qualifying night scene inside the Test archive cannot be established
   without an authorized archive inspection.
3. A human must confirm the 21 official slot numbers on a pre-inference
   polygon overlay.

### Candidate 2: CNRPark+EXT

- Official page: <https://cnrpark.it/>
- Associated paper:
  <https://doi.org/10.1016/j.eswa.2016.10.055>
- Official license: ODbL 1.0.
- Structure: nine fixed cameras, 4,081 labelled full frames, 144,965
  parking-slot records, per-camera geometry CSVs, and minute-bearing file
  names.
- Official condition groups: sunny, overcast, and rainy; no qualifying
  night test is established.
- Prior project use: already consumed as the once-only CNR-EXT static
  external holdout.
- Gate: **INELIGIBLE**.

The authorized archive was supplied locally as
`CNR-EXT_FULL_IMAGE_1000x750.tar`. Its machine-specific path is omitted from
the public source release.

Read-only verification:

- bytes: `1,104,364,544`;
- SHA-256:
  `02a66d936433f02dfceae37a25eb3ee100969ff63eb38d891621e8bf42d1256f`;
- existing extraction:
  `FULL_IMAGE_1000x750`;
- duplicate download: **not performed**.

The user's size authorization is recorded, but it cannot make previously
consumed data independent or turn a non-night-qualified source into the
Stage Q final test.

### Candidate 3: Action-Camera Parking Dataset (ACPDS)

- Official repository:
  <https://github.com/martin-marek/parking-space-occupancy>
- Associated paper: <https://arxiv.org/abs/2107.12207>
- Repository license: MIT.
- Official archive: `rois_gopro.zip`, 380,796,292 bytes.
- Structure: 293 images, 11,236 annotated slot views, quadrilateral
  annotations.
- Gate: **INELIGIBLE**.

The paper states that the GoPro was moved on a telescoping pole and that
each image has a unique view. This is not a reusable fixed-camera scene, an
ordered night sequence, or a stable scene-level polygon/transition source.

## Q1 — download gate

Final gate: **BLOCKED_BEFORE_DOWNLOAD**.

- UPM-GTI: no download because archive permission is unclear.
- CNR-EXT: user authorized the large archive, but the exact archive and
  extraction already existed; no repeat transfer was needed. It remains
  ineligible.
- ACPDS: no download because it fails the fixed-camera/night-sequence gate.

The gate executes before any model construction or inference callback. The
verification command reported:

```json
{
  "status": "BLOCKED_BEFORE_DOWNLOAD",
  "formal_inference_authorized": false,
  "model_callback_called": false,
  "primary": "P3-D1",
  "secondary": "P3-D1-LL",
  "default_detector": "D1"
}
```

## Q2–Q5 execution status

These stages were not executed:

| Stage | Status | Reason |
|---|---|---|
| Q2 archive acquisition/structure audit | Not executed for the only promising candidate | UPM image-use permission is unclear |
| Q3 polygon and truth conversion | Not executed | No authorized UPM source images; slot mapping cannot be verified |
| Q4 formal protocol freeze | Not created | Archive, scene, manifest, polygon, and truth hashes do not exist |
| Q5 P3-D1/P3-D1-LL inference | Not executed | Formal data gate is blocked |

The following files are intentionally absent rather than populated with
placeholder or fabricated evidence:

- `STAGE_Q_SOURCE_ARCHIVE_AUDIT_20260729.yaml`;
- `STAGE_Q_TEST_SCENE_SELECTION_20260729.yaml`;
- `STAGE_Q_TEST_IMAGE_MANIFEST_20260729.csv`;
- `STAGE_Q_SLOT_POLYGONS_20260729.json`;
- `STAGE_Q_OCCUPANCY_TRUTH_20260729.csv`;
- `STAGE_Q_POLYGON_VALIDATION_20260729.png`;
- `STAGE_Q_ANNOTATION_FREEZE_20260729.yaml`;
- `stage_q_external_night_occupancy_frozen_20260729.yaml`;
- formal P3-D1 or P3-D1-LL output directories.

## Implemented safeguards

Stage Q adds reusable, inference-independent safeguards for a later
authorized run:

- strict parsing of contiguous or delimited 21-value occupancy vectors;
- source-to-project mapping:
  `1=vacant -> state 0`, `0=occupied -> state 1`;
- explicit unknown exclusion and accounting;
- one-to-one image/truth membership validation;
- exact 21-slot polygon ID, convexity, nonzero-area, and coordinate-bound
  validation;
- per-file byte/SHA-256 verification and a canonical scene-manifest hash;
- frozen primary/secondary roles and exact D1/D1-LL inference equality;
- enforcement that D1 remains the default detector;
- prohibition of seconds-level transition latency for low-frame-rate image
  sequences while allowing state-change agreement and frame-index
  transition differences;
- complete per-method output schema validation;
- registry byte/SHA-256 validation;
- a blocked gate that raises before a model callback can run.

No raw detector output, tracker-emitted box output, or slot-occupancy output
was produced in Stage Q.

## Verification

- Stage Q targeted tests: **18 passed**.
- Full `implementation/tests`: **221 passed**.
- Full `implementation/literature_core/tests`: **83 passed**.
- Combined repository test count: **304 passed**.
- `compileall`: passed for implementation and literature source, scripts,
  and tests.
- `git diff --check`: passed; Git emitted only existing line-ending
  conversion warnings.
- Blocked-gate proof: passed with `model_callback_called=false`.
- Stage O frozen registry: **213/213 artifacts verified**, registry SHA-256
  `efbbb63b77aefae00c1c4758b8df1dd463c2fd4a54bc32626ebf441073b87660`.
- Stage P frozen registry: **39/39 artifacts verified**, registry SHA-256
  `206db0c3b9cd1a3575eeb7e5ba56a23f3f7f621caee139e61f47d9ee6efe06fa`.

The first implementation-suite run in the isolated Stage O environment
reported three failures because the optional official `trackeval` package
was not on that environment's import path. Re-running against the project's
pre-existing pinned TrackEval vendor copy made all 221 tests pass. This was
an environment-path issue, not a functional regression.

## Quantitative result boundary

There are no Stage Q slot-level values for Macro F1, occupied precision,
occupied recall, vacant recall, false-free rate, false-occupied rate,
accuracy, occupancy-count MAE/RMSE, confusion matrix, steady-state latency,
or FPS. Reporting any such value would fabricate an evaluation.

No qualifying scene or temporal truth was obtained. Therefore:

- state changes are not established;
- temporal stability was not evaluated;
- seconds-level transition latency is not supported;
- P3-D1 and P3-D1-LL have no Stage Q quantitative comparison;
- D1 remains the production/default P3 detector regardless of any future
  Stage Q single-scene result.

## Claims supported now

- A maximum-three official-source audit was completed.
- UPM-GTI is structurally promising but blocked by unclear archive
  permission and unverified Test/night/slot-mapping facts.
- The CNR-EXT archive already exists locally with the recorded size and
  SHA-256, but CNR-EXT is consumed and lacks an officially established
  qualifying night test.
- The Stage Q implementation enforces the intended data, comparison,
  temporal, output, and artifact boundaries without loading a model.

## Claims not supported

- No new external night parking occupancy performance is established.
- Neither P3-D1 nor P3-D1-LL has a Stage Q occupancy result.
- D1-LL has not become the default parking detector.
- LMOT vehicle-detection improvements are not parking-slot occupancy
  improvements.
- A low-frame-rate image sequence is not evidence of real-time tracking or
  seconds-level transition latency.
- A single future scene would not establish universal parking-lot
  generalization.

## Minimum unblock action

Before any UPM-GTI download or inference:

1. Obtain written confirmation from UPM-GTI that the Parking Lot Occupancy
   Database image archives may be used for non-commercial course research,
   including attribution and redistribution restrictions. Suggested
   contacts are Ana I. Maqueda or Carlos R. del-Blanco at UPM-GTI.
2. After permission is confirmed, download the official `test.zip`
   (displayed as 239.1 MB) manually or explicitly authorize Codex to do so,
   then provide the absolute path to the unmodified archive.
3. After archive audit identifies a qualifying fixed night Test sequence,
   manually confirm the official 21-slot numbering on the rendered
   pre-inference polygon overlay.

Only after those gates pass may Q2–Q5 continue. No login was requested by
the public UPM share during this audit; the current blocker is permission
clarity, not authentication.
