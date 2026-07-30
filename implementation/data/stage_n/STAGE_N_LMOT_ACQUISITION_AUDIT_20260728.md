# Stage N LMOT limited-acquisition audit

Audit time: 2026-07-28 (Asia/Shanghai)

Decision: **blocked before download**. No LMOT byte was downloaded, no
archive was opened, and no LMOT prediction was run.

## Official release facts

The [official LMOT repository](https://github.com/xinzwang/LMOT) states that
the paired benchmark uses aligned low-light/well-lit cameras, provides RAW and
sRGB forms, records at 20 FPS with 10 ms exposure and 1800 x 1000 resolution,
and labels `car`, `person`, `bicycle`, `motorcycle`, `bus`, and `truck`. Its
published counts are 11 train, 4 validation, 11 test, and 6 LMOT-real videos,
but the current download note says only train and validation are released.

The dataset license is
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); the repository
code is MIT. Stage N therefore treats LMOT as non-commercial validation
diagnostic material only.

The official repository publishes one Baidu Netdisk link,
`https://pan.baidu.com/s/1OHojTQSTdDaybuflYGwaMw`, with extraction code
`xedx`. The README describes a release tree containing `train/` and `val/`,
and per-sequence `gt/gt.txt`, `img_dark`, `img_dark_rgb`, `img_light`,
`img_light_rgb`, and `seqinfo.ini`.

## Required pre-download questions

| Question | Verified answer | Consequence |
|---|---|---|
| Download package size | Not published in the official README | Block |
| Validation selectable alone | Not stated | Block |
| sRGB selectable without RAW | Not stated | Block |
| Baidu account/login required | Not verifiable without entering the interactive share | Block |
| Baidu desktop client required | Not verifiable without entering the interactive share | Block |
| Internal archive directory and member sizes | README gives an illustrative tree, but the actual archive was not inspected | Block |
| Free space on `C:` | 117.10 GiB at audit time | Insufficient basis for an unknown potentially large package |
| Free space on `D:` | 77.24 GiB at audit time | Insufficient basis for an unknown potentially large package |

An attempt to inspect the interactive share with the signed-in in-app browser
was rejected by the browser security policy. That policy explicitly prohibits
workarounds or alternate browser automation, so Stage N did not bypass it.

## Approved scope

Only these validation members may ever pass the local archive inspector:

- `val/*/img_dark_rgb/**`
- `val/*/img_light_rgb/**`
- `val/*/gt/gt.txt`
- `val/*/seqinfo.ini`

RAW streams, any TIFF member, train, test, LMOT-real, a complete release of
hundreds of gigabytes, and LTrack training dependencies are prohibited. The
new `scripts/acquire_stage_n_lmot.py` performs no network access. It hashes and
lists a user-supplied ZIP/TAR, blocks the full archive if any unapproved member
is present, and extracts only after the complete member list passes.

## Additional truth blockers

The official README defines the nine columns as
`fn,id,x,y,width,height,ignore,classid,visibility`, but it does not explicitly
map numeric `classid` values to the six names. The listed order is not accepted
as evidence. It also does not define which `ignore` values are evaluated.
Therefore conversion and formal evaluation require a separately frozen,
officially supported numeric mapping and ignore-value interpretation.

## Acquisition record

- Source URL: official Baidu share above
- Audit date: 2026-07-28
- License: CC BY-NC 4.0
- Downloaded bytes: **0**
- Archive SHA-256: not applicable
- Extracted-file SHA-256: not applicable
- File/video/frame counts: not available because acquisition was blocked
- Next admissible action: the user may manually inspect the official share and
  provide package/member sizes and selectivity. Any login, client installation,
  or large transfer still requires explicit permission before transfer.
