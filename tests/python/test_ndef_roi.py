import numpy as np

from neurodic.ndef_roi import _shared_observations


def test_shared_observations_use_pair_track_union_and_source_uv():
    observations = {
        "cam_indices": np.asarray([0, 0, 0, 0, 1, 1, 2, 2]),
        "point_indices": np.asarray([10, 11, 12, 13, 10, 12, 11, 12]),
        "uv": np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [101, 101], [103, 103], [202, 202], [203, 203]]),
    }

    uv, support, counts = _shared_observations(observations, source=0, neighbors=[1, 2])

    np.testing.assert_array_equal(uv, [[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(support, [1, 2, 3])
    assert counts == {1: 2, 2: 2}
