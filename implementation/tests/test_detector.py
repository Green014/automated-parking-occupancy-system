import numpy as np

from parking_occupancy.detector import track_ids_from_output


def test_track_ids_from_output_keeps_unmatched_detections() -> None:
    tracks = np.asarray(
        [
            [10, 20, 30, 40, 101, 0.8, 2, 1],
            [50, 60, 70, 80, 205, 0.7, 5, 0],
        ],
        dtype=np.float32,
    )

    assert track_ids_from_output(tracks, 3) == [205, 101, None]


def test_track_ids_from_empty_output() -> None:
    assert track_ids_from_output(np.empty((0, 8)), 2) == [None, None]
