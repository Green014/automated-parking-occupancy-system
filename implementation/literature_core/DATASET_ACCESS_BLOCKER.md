# Temporal Dataset Access Blocker (Resolved)

Status: access and two-scene split conditions resolved on 26 July 2026.

## Resolution

The project now has two legally usable VIRAT clips from distinct physical
scenes with fixed marked bays, mixed occupied/vacant states, and a visually
verified departure in each. `0502` is development (last occupied 1659, first
vacant 1660); `0503` is the locked holdout (last occupied 1549, first vacant
1550). The protocol validator verified both videos, SHA-256 values, frame
bounds, polygons, interval coverage, class counts, transitions, and scene
separation.

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
`0503` pass added five official videos. The final bounded set contains 26
videos / 1,605,720,653 video bytes and 15,925,504 annotation bytes. Every
downloaded file has an official item ID and SHA-256 in the corresponding
manifest.

Official Release 2.0 section 2.2 defines the first four filename digits
`XXYY` as the physical scene and the next two `ZZ` as the sequence. The first
screening pass incorrectly treated six-digit prefixes as scene IDs; the
download tool and tests now enforce the official four-digit grouping. Of the
initial 21 clips, 20 were rejected for missing complete parking slots or a
within-clip slot transition. After three targeted `0503` rejections, bounded
retry recovered two videos previously blocked by temporary HTTP 502 responses.
The second of those contains the eligible holdout transition. Screening then
stopped to prevent outcome-based holdout replacement.

## Scientific consequence

The access blocker is resolved, and `configs/temporal_protocol_pending.yaml`
now validates as frozen and experiment-ready despite its compatibility
filename. E4 and E5 were executed under
`configs/temporal_e4_e5_frozen.yaml`. Both are negative case-study results:
neither beat T0 on holdout, and E5 had zero vacant recall on development.

The remaining blocker is methodological rather than access-related:

- Fusion V2 remains prohibited because E5 did not pass its development
  reliability gate;
- Grand Bassin's occupied-only negative result remains unchanged;
- CNR-EXT remains consumed once-only and cannot select any new parameter;
- no IDF1 or HOTA is reported because identity ground truth is absent; and
- no general arrival/departure or tracking claim is made from two one-slot,
  one-departure videos.
