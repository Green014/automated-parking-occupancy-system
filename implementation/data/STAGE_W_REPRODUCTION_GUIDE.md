# Stage W Reproduction Guide

Date: 2026-07-30

## Prerequisites

- Python 3.10 or newer;
- base dependencies installed from the frozen `pyproject.toml`, including
  OpenCV, PyTorch/torchvision and Ultralytics;
- the additive dashboard dependency installed from
  `stage_w_requirements.txt`;
- a readable video file, camera index or RTSP URL;
- a polygon slot map accepted by the existing parking-space loader;
- for `detection`/`fusion`, the local D1 checkpoint;
- for `fusion`, the local E1b checkpoint.

The runtime never downloads or silently substitutes a model. Verify the local
artifacts before use:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\literature_core
.\.venv\Scripts\python.exe -m pip install -e ".[integrated,dashboard,dev]"
```

For compatibility with an already installed base environment, the dashboard
dependency alone can still be installed with
`python -m pip install -r stage_w_requirements.txt`. Both declarations use
the same `Flask>=3.1,<4` range.

| Model | Bytes | Required SHA-256 |
|---|---:|---|
| D1 `best.pt` | 6,255,409 | `0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64` |
| E1b `best.pt` | 8,045,704 | `f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3` |

The intended GitHub Release URL is pending. No public URL is asserted before a
real Release is created. After a future download, or when using a preserved
local checkpoint, pass the path explicitly and verify the table above:

```powershell
Get-Item <asset-path> | Select-Object Length
Get-FileHash -Algorithm SHA256 <asset-path>
```

## Start the default dashboard

From `implementation`:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video-path|camera-index|rtsp-url> `
  --slots <parking-spaces.json> `
  --mode fusion `
  --d1-weights <D1-best.pt> `
  --e1b-checkpoint <E1b-best.pt> `
  --output-dir outputs\stage_w_dashboard_<new-id> `
  --host 127.0.0.1 `
  --port 5000
```

Open `http://127.0.0.1:5000/`. The default server is local-only, debug and
the reloader are disabled, and a non-loopback host requires the explicit
`--allow-remote-bind` option.

Do not place RTSP URLs containing credentials in committed scripts or
documentation. API, summary and error payloads redact such credentials, but
shell history is outside the application's control.

## Select a mode

```powershell
# Clean-room OpenCV teaching baseline
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video> --slots <slots.json> --mode classic

# D1 + B1
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video> --slots <slots.json> --mode detection `
  --d1-weights <D1-best.pt>

# D1 + B1 + E1b + F2
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video> --slots <slots.json> --mode fusion `
  --d1-weights <D1-best.pt> --e1b-checkpoint <E1b-best.pt>
```

E4 is enabled only by `--temporal` in Fusion. ByteTrack or TrackTrack is
enabled only by `--tracker bytetrack` or `--tracker tracktrack`. These are
recorded variants, not the default frozen C2 comparison.

## Optional external member-reference mode

The member runtime is an external local dependency:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video> --slots <slots.json> --mode member-reference `
  --member-reference-root <audited-member-checkout>
```

The checkout must be at commit
`12271576be39a4ac0eb456526eca122685799e8c` unless the adapter is explicitly
changed and re-audited. Its configured model must already be local. Failure
is reported explicitly; the dashboard does not switch to another backend.
For public-source provenance, see
`PUBLIC_PERMISSION_AND_PROVENANCE.md`. Private authorization evidence and the
older local permission record are intentionally absent from the public source
candidate.

## Headless functional smoke

Use a fresh output directory; existing directories are rejected:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_w_dashboard.py `
  --input <video> --slots <slots.json> --mode fusion `
  --d1-weights <D1-best.pt> --e1b-checkpoint <E1b-best.pt> `
  --output-dir outputs\stage_w_dashboard_smoke_<new-id> `
  --max-frames 4 --no-serve
```

The directory contains:

- `annotated.mp4`;
- `events.jsonl`;
- `status.json`;
- `summary.json`;
- `configuration_snapshot.yaml`.

The source in publishable metadata is reduced to a safe label. Accuracy is
`not_computed_no_truth` unless an independently governed truth workflow is
explicitly added.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_stage_v_multimode.py `
  tests\test_stage_w_ui_adapter.py `
  tests\test_stage_w_server.py

.\.venv\Scripts\python.exe scripts\verify_stage_v_w_registries.py
```

For each real run, confirm that:

- the input decodes and the expected frame count is recorded;
- `rendered_slots` equals `total` on every frame;
- API occupied/vacant counts sum to total and match the visualization;
- the first frame creates no event;
- paths and RTSP credentials are absent from publishable metadata;
- capture/writer/thread are released on completion;
- no prior output directory was overwritten.
