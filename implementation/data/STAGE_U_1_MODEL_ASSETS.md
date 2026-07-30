# Stage U.1 Model Asset Requirements

Date: 2026-07-30

Model assets are intentionally excluded from the portable submission and ZIP.
The runtime accepts paths supplied by the user; it does not download weights.

| Asset | Recommended local filename | Bytes | Required SHA-256 for frozen D1/E1b identity |
|---|---|---:|---|
| D1 detector | `D1_NDISPark_best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| E1b classifier | `E1b_CBAM_best.pt` | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

## Acquisition

These are project-produced frozen artifacts, not upstream pretrained
downloads. They are prepared as separate future GitHub Release assets:

- Release URL: pending

Until a real Release exists, obtain them from the preserved local experiment
archive:

- D1: frozen `d1_ndispark_formal_20260727_v1/weights/best.pt`;
- E1b: frozen `mobilenet_variant_ablation/cbam_supplement/best.pt`.

Do not invent or infer a download URL and do not substitute a file based only
on its name. After any manual or future Release download, verify byte count and
SHA-256:

```powershell
Get-Item <asset-path> | Select-Object Length
Get-FileHash -Algorithm SHA256 <asset-path>
```

The generic P3-TT runtime permits different user-supplied checkpoints, but
records `custom_weights=true` and
`stage_t_result_comparison_applicable=false`. Frozen Stage T reproduction
requires the exact hashes above.

Model cards and the machine-readable release manifest are:

- `MODEL_CARD_D1.md`;
- `MODEL_CARD_E1B.md`;
- `STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml`.

The source repository continues to exclude every `.pt` file.
