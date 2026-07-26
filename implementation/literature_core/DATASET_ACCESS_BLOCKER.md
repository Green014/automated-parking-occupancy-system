# Temporal Dataset Access Blocker

Status: access condition resolved and bounded visual screening completed on
26 July 2026; a leakage-safe two-scene temporal split is still blocked.

## Blocking condition

The project now has one legally usable VIRAT clip with complete marked bays,
mixed occupied/vacant states, and a visually verified departure:
`VIRAT_S_050202_10_002159_002233.mp4` (physical scene `0502`). It does not yet
have a second independent physical scene with a verified slot transition.
The surviving clip's fixed polygon and frame-state boundary are finalized as
candidate truth (last occupied 1659, first vacant 1660), but its partition
role remains unassigned and it has not been used for tuning.

VIRAT Ground Release 2.0 is the conditional primary candidate. Its usage
agreement states that every individual with access must accept the terms. The
user confirmed personal acceptance on 26 July 2026. No name, email,
affiliation, signature, credential, or agreement copy is stored in Git.

The next-best dataset candidate, DLP, requires an author-approved raw-video request and
uses a drone camera whose motion may invalidate fixed slot ROIs. The official
sample URL currently reaches a OneDrive permission-validation page rather than
an anonymously downloadable file listing. EPFL's current official page exposes
only non-consecutive ground-truth frames, not the described full video.

## Completed access and screening action

The user personally:

1. open the [VIRAT official page](https://viratdata.org/);
2. read the linked VIRAT Video Dataset Usage Agreement;
3. accept it through the official access flow; and
4. confirmed only that acceptance was completed.

The initial project pass downloaded 21 official clips totaling 961,643,821
bytes, below its 1,000,000,000-byte screening budget. A separate targeted
`0503` pass added three official videos totaling 468,762,635 bytes plus their
event/mapping/object annotations. Every downloaded file has an official item
ID and SHA-256 in the corresponding manifest.

Official Release 2.0 section 2.2 defines the first four filename digits
`XXYY` as the physical scene and the next two `ZZ` as the sequence. The first
screening pass incorrectly treated six-digit prefixes as scene IDs; the
download tool and tests now enforce the official four-digit grouping. Of the
initial 21 clips, 20 were rejected for missing complete parking slots or a
within-clip slot transition. The remaining clip now has verified candidate
truth. The three `0503` additions were rejected because the associated vehicle
either stayed parked or departed only from an unmarked curb/edge row. This is
still insufficient for a distinct-scene development/holdout pair, and the
`0503` screen remains explicitly non-exhaustive.

If the user does not accept VIRAT terms, the alternative is to request DLP raw
video or obtain written reuse permission for another fixed-camera dataset.
Those requests require the user's own identity, affiliation, and intended-use
statement and will not be submitted automatically.

## Scientific consequence

Until a second physical scene and manual frame truth are complete:

- `configs/temporal_protocol_pending.yaml` remains `candidate_screening`;
- E4 mixed-class metrics, E5 tracking-aware occupancy, and Fusion V2 are
  prohibited;
- Grand Bassin's occupied-only negative result remains unchanged;
- CNR-EXT remains consumed once-only and cannot be used to choose any new
  parameter;
- no transition latency, IDF1, or HOTA result will be reported.
