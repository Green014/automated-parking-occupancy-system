# Grand Bassin Continuous-Truth Audit

Run date: 25 July 2026 (Asia/Shanghai)

## Decision

No reviewed Grand Bassin hypothesis is suitable for mixed-class or transition
ground truth. The existing seven-bay, continuously occupied bus annotation
remains valid for the restricted positive-only stability check. No vacant
slot, arrival, or departure label has been added.

This is a data-suitability result, not a model-accuracy result. Running E4 on
an access lane, a cropped vehicle, no-parking hatching, or a detector-induced
false transition would create invalid metrics.

## Sources and coverage

| Sequence | Form | Timeline samples | Rate | Nominal duration (`N/FPS`) | Automated candidates |
|---|---|---:|---:|---:|---:|
| `grand_bassin_aerial_development` | MP4 | 793 | 2 FPS | 396.5 s | 3 |
| `grand_bassin_aerial_candidate_133033` | MP4 | 361 | 2 FPS | 180.5 s | 7 |
| `grand_bassin_aerial_holdout_162512` | ordered images | 195 | 2 FPS | 97.5 s | 6 |
| **Total** | - | **1,349** | - | **674.5 s** | **16** |

The holdout initially had only 24 local images. The existing manifest
downloader acquired the other 171 files, after which all 195 ordered images
were available. File hashes are recorded in
`outputs/candidate_search/holdout_162512_checksums.csv`.

The automated search generated 16 spatial hypotheses and 34 proposed state
changes. Review sheets contain the proposed-transition neighborhoods plus
uniformly distributed samples: 119 candidate-frame appearances for the
development video, 300 for the second video, and 384 for the holdout
sequence. Overlapping appearances are deliberately not reported as unique
source frames.

## Acceptance protocol

A hypothesis was accepted only if all of the following were visually
supported:

1. it described one fixed, complete parking-space polygon;
2. the area was a legal marked bay rather than a lane, queue, separator, or
   hatched exclusion area;
3. occupied and vacant states could both be seen without assigning one
   vehicle to two overlapping spaces; and
4. a proposed transition corresponded to an actual arrival or departure,
   rather than detector dropout, association drift, glare, or occlusion.

`scripts/build_transition_review.py` creates full-scene and ROI contact sheets
from video or ordered-image manifests. It includes uniform samples,
transition-centred windows, optional global coordinate grids, source hashes,
and a review manifest. `scripts/draw_slot_map.py` creates labelled polygon
overlays for boundary checks.

Machine detections were used only to propose review locations. They were never
treated as occupancy or transition truth.

## Adjudication

All 16 automated candidates were rejected:

| Rejection class | Count |
|---|---:|
| Circulation/access vehicle rather than a fixed slot | 8 |
| Complete slot unavailable at an image boundary | 3 |
| Marked-row target with no human-visible state change | 3 |
| Queued/overlapping vehicle, no independent bay | 1 |
| Not a parking space | 1 |

The per-candidate coordinates, transition frames, review counts, and notes are
stored in
`data/annotations/grand_bassin_transition_candidate_adjudication.csv`.

Seven additional targeted hypotheses were rejected. These included the
original apparent bus departure, a focused holdout departure, two
upper-right hatched areas, a circulation-lane gap, and a dark vehicle/glare
mistaken for an empty stall. Their evidence is recorded in
`data/annotations/grand_bassin_rejected_manual_hypotheses.csv`. Re-runnable
review inputs are stored separately as
`grand_bassin_development_rejected_review_candidates.csv` and
`grand_bassin_holdout_rejected_review_candidates.csv`; the word `rejected`
in each filename prevents them from being mistaken for truth.

Finally, uniform full-scene and coordinate-grid checks covered the bus area,
central car rows, lower rows, upper-right rows, and the large left-foreground
parking rows. The visible marked spaces in the last region were occupied; its
large dark gaps were circulation aisles.

## Consequence for E4/E5

The following metrics remain unsupported on the Grand Bassin continuous data:
vacant recall, false-occupied rate, mixed-class macro F1, transition latency,
mixed-class flicker, IDF1, and HOTA. The already executed positive-only check
may report occupied recall and unsupported state changes, but it cannot be
promoted to a full temporal evaluation.

A later, separate VIRAT search met the minimum occupancy-truth requirements
for a bounded two-scene case study and is documented in `RESULTS.md`. It does
not change this Grand Bassin adjudication. Each VIRAT partition contains only
one slot and one departure, so it cannot support broad tracking
generalization, arrival robustness, IDF1, or HOTA claims.

A stronger next experiment still requires continuous sequences with:

- multiple manually bounded marked bays with both classes;
- multiple unambiguous arrivals and departures;
- complete per-frame slot state near every transition; and
- identity ground truth only if tracking metrics are claimed.

## Reproduction

Run the candidate finder once per sequence using the matching COCO
preannotations and ordered manifest:

```powershell
implementation\.venv\Scripts\python.exe `
  implementation\scripts\find_temporal_slot_candidates.py `
  --coco implementation\data\annotations\grand_bassin_aerial_development_coco.json `
  --manifest implementation\data\splits\grand_bassin_aerial_development.csv `
  --project-root implementation `
  --output-csv implementation\literature_core\outputs\candidate_search\development.csv `
  --overlay-output implementation\literature_core\outputs\candidate_search\development.jpg
```

Then build the evidence sheets:

```powershell
implementation\.venv\Scripts\python.exe `
  implementation\literature_core\scripts\build_transition_review.py `
  --video implementation\data\raw\grand_bassin\grand_bassin_aerial_development.mp4 `
  --candidates implementation\literature_core\outputs\candidate_search\development.csv `
  --output-dir implementation\literature_core\outputs\candidate_search\development_review `
  --radius 10 --uniform-samples 12 --padding 45
```

Equivalent runs use `--manifest` and `--project-root` for the ordered-image
holdout. Review outputs are evidence artifacts, not labels.
