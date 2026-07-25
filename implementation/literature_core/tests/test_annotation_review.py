from literature_core.annotation_review import (
    nearest_available_indices,
    parse_transition_frames,
    state_intervals,
    temporal_review_indices,
    uniform_frame_indices,
)


def test_parse_transition_frames_accepts_common_delimiters():
    assert parse_transition_frames("17, 25;31|40") == [17, 25, 31, 40]
    assert parse_transition_frames("") == []


def test_uniform_frame_indices_include_endpoints():
    assert uniform_frame_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_frame_indices(3, 10) == [0, 1, 2]


def test_temporal_review_indices_add_dense_transition_window():
    assert temporal_review_indices(
        20, [10], radius=2, uniform_samples=2
    ) == [0, 8, 9, 10, 11, 12, 19]


def test_nearest_available_indices_deduplicates_and_prefers_earlier_tie():
    assert nearest_available_indices([0, 9, 10, 11, 30], [0, 5, 15, 25]) == [
        0,
        5,
        15,
        25,
    ]


def test_state_intervals_compress_binary_runs():
    assert state_intervals([1, 1, 0, 0, 0, 1]) == [
        (0, 1, 1),
        (2, 4, 0),
        (5, 5, 1),
    ]
