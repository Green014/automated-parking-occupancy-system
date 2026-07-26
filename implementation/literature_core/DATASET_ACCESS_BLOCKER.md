# Temporal Dataset Access Blocker

Status: active as of 26 July 2026.

## Blocking condition

The project has no legally usable, locally verified continuous video that
supports mixed occupied/vacant slot truth and real arrivals/departures.

VIRAT Ground Release 2.0 is the conditional primary candidate, but its usage
agreement states that every individual with access must accept the terms. That
acceptance is a personal legal action and has not been recorded. The project
therefore has not downloaded or used VIRAT video content.

The next-best candidate, DLP, requires an author-approved raw-video request and
uses a drone camera whose motion may invalidate fixed slot ROIs. The official
sample URL currently reaches a OneDrive permission-validation page rather than
an anonymously downloadable file listing. EPFL's current official page exposes
only non-consecutive ground-truth frames, not the described full video.

## Exact unblock action

The user must personally:

1. open the [VIRAT official page](https://viratdata.org/);
2. read the linked VIRAT Video Dataset Usage Agreement;
3. accept it through the official access flow; and
4. confirm only that acceptance was completed.

No name, email, affiliation, signature, or account credential should be added
to Git. After confirmation, the next run will acquire only a small official
screening subset, hash it, verify the video properties, and freeze distinct
development/holdout scenes before annotation or tuning.

If the user does not accept VIRAT terms, the alternative is to request DLP raw
video or obtain written reuse permission for another fixed-camera dataset.
Those requests require the user's own identity, affiliation, and intended-use
statement and will not be submitted automatically.

## Scientific consequence

Until access and visual screening are complete:

- `configs/temporal_protocol_pending.yaml` remains `pending_access`;
- E4 mixed-class metrics, E5 tracking-aware occupancy, and Fusion V2 are
  prohibited;
- Grand Bassin's occupied-only negative result remains unchanged;
- CNR-EXT remains consumed once-only and cannot be used to choose any new
  parameter;
- no transition latency, IDF1, or HOTA result will be reported.
