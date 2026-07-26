import numpy as np

from literature_core.detector import track_ids_from_output


def test_track_ids_keep_unmatched_raw_detections() -> None:
    tracks = np.asarray(
        [
            [0, 0, 10, 10, 205, 0.9, 0, 2],
            [0, 0, 10, 10, 101, 0.8, 0, 0],
        ],
        dtype=float,
    )
    assert track_ids_from_output(tracks, 3) == [101, None, 205]


def test_track_ids_from_empty_output() -> None:
    assert track_ids_from_output(np.empty((0, 8)), 2) == [None, None]
