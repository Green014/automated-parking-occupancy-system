from parking_occupancy.sequence_io import evenly_spaced_indices


def test_evenly_spaced_indices_include_both_endpoints() -> None:
    assert evenly_spaced_indices(10, 4) == [0, 3, 6, 9]


def test_evenly_spaced_indices_do_not_duplicate_small_inputs() -> None:
    assert evenly_spaced_indices(2, 10) == [0, 1]
