# Stage W UI Integration Report

Date: 2026-07-30  
Status: local course-project integration complete  
Default mode: `fusion` (`D1 + B1 + E1b + F2`)

## Result

Stage W connects the unified occupancy runtime to a Flask dashboard through an
adapter boundary. The Flask routes do not contain or duplicate D1/B1/E1b/F2
logic. A single synchronized background processor owns the model backend,
video capture, optional writer, latest frame and event/session state; browser
refreshes and multiple clients reuse that processor and do not reload models.

```text
fixed camera video/image + configured polygons
  -> D1 detection
  -> optional ByteTrack/TrackTrack identity continuity
  -> B1 one-to-one polygon coverage
  -> F2 detector-positive authority
       + E1b review of every detector-negative slot
  -> optional E4 slot-state stabilization
  -> FrameOccupancyResult
  -> Stage W UI adapter
  -> Flask JSON + MJPEG + events/sessions
```

## Modes

| Mode | Meaning | Default tracking / temporal |
|---|---|---|
| `classic` | Independently implemented OpenCV foreground-ratio teaching baseline | off / off |
| `detection` | D1 + B1 | off / not applicable |
| `fusion` | D1 + B1 + E1b + F2 | off / off |
| `member-reference` | Optional external member runtime, only when its audited checkout, configuration and local model dependency are available | member-defined |

`classic` is not the member system. `member-reference` is never a silent
fallback. Missing models, invalid configuration, unreadable video or a member
dependency error produces an explicit error state and HTTP 503 status payload.

## UI/API adapter contract

`frame_result_to_ui_payload` emits exactly one slot record per configured
polygon and enforces:

- `occupied + vacant = total`;
- `rendered_slots = total`;
- slot IDs are distinct from vehicle detection boxes and track IDs;
- `track_id` is `null` when tracking is disabled;
- local absolute paths and RTSP credentials are removed from API payloads;
- runtime contains attributed processing milliseconds/FPS and cache status;
- temporal and tracker state are explicit.

The dashboard exposes:

- `/` — dashboard;
- `/video_feed` — MJPEG annotated video;
- `/api/status` — current slot and runtime state;
- `/api/events` — recent state changes;
- `/api/sessions` — in-memory parking sessions;
- `/api/health` — lifecycle/error state.

The annotated frame renders all polygons, green for vacant, red for occupied,
yellow for vehicle detections, and cyan track labels only when tracking is
enabled and IDs exist. Polygon labels use short numeric IDs to reduce
overlap. The header shows mode, rendered slot count, attributed FPS, cache,
temporal and tracker state.

## Runtime and lifecycle behavior

- The server binds to `127.0.0.1` by default.
- A non-loopback bind requires `--allow-remote-bind`.
- Flask debug mode and the reloader are disabled.
- Video files, camera indices and RTSP URLs are accepted.
- RTSP credentials are redacted from status, summaries and errors.
- warm-up is performed once by default for model-backed modes and excluded
  from the recorded stream; cache/state is reset afterward.
- capture, writer and background thread are released on stop or input end.
- an existing output directory is rejected.

## Final model-backed smoke

The authoritative local output is
`outputs/stage_w_dashboard_smoke_20260730_v3`.

Observed final status:

- health: `completed`;
- mode: `fusion`;
- decoded frames: 4;
- configured/rendered slots per frame: 5/5;
- final occupied/vacant/total: 2/3/5;
- temporal/tracker: off/off;
- recent events: 0;
- first-frame event: false;
- accuracy status: `not_computed_no_truth`;
- published source label: `input.mp4` (no absolute path).

The output `annotated.mp4` is decodable and its SHA-256 is
`ac2f0c3dae9053c055b434f640a5bd61a6493833ff640c735976c8f8096f99e9`.
It is a repeated consumed-development demonstration and supports no accuracy
claim.

## Longer interface demonstration

`dashboard_ui_demo.mp4` is a 10-second, 1280×720, 5 FPS post-hoc composite of
the four-frame Stage W annotated smoke with its dashboard state. It contains
50 frames and has SHA-256
`12c1e85328ecb5cf5e8e5385ec4e97d24c2398a8f8ff26d0940dc843092288af`.

No model inference was rerun to create this longer video; the four annotated
frames are looped. The video itself states that it is an interface
demonstration, not fixed-camera performance validation. No suitable longer,
licensed, unconsumed fixed-camera parking sequence with frozen truth was
available, so no substitute aerial or moving-camera video was presented.

## Claim boundary

Stage W demonstrates integration, lifecycle handling, visualization, schema
invariants and real D1/E1b execution on consumed smoke input. It does not
establish occupancy accuracy, event accuracy, transition latency, real-time
production throughput, parking duration accuracy or a TrackTrack improvement.

