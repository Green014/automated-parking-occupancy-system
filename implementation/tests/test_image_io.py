import numpy as np

from parking_occupancy.image_io import read_image, write_image


def test_image_round_trip_with_unicode_path(tmp_path) -> None:
    path = tmp_path / "停车场" / "frame.png"
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    image[:, :, 1] = 127

    write_image(path, image)
    loaded = read_image(path)

    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)
