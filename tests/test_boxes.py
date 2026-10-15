import numpy as np

from conftest import LATS, LONS
from paleotc.boxes import box_mask

TLAT = np.tile(LATS[:, None], (1, LONS.size))
TLON = np.tile(LONS, (LATS.size, 1))


def test_single_box():
    m = box_mask(TLAT, TLON, np.array([[0.0, 0.0]]), 10.0)
    assert m.sum() == 1 and m[1, 0]


def test_box_wraps_longitude():
    # storm at 355E: the grid point at 0E is 5 degrees east of it
    m = box_mask(TLAT, TLON, np.array([[355.0, 15.0]]), 10.0)
    assert m.sum() == 1 and m[2, 0]


def test_box_is_half_open():
    # point at lat 0 is exactly +10 from center -10 -> excluded; lat -20 is -10 -> included
    m = box_mask(TLAT, TLON, np.array([[0.0, -10.0]]), 10.0)
    assert m[0, 0] and not m[1, 0]


def test_union_of_storms():
    m = box_mask(TLAT, TLON, np.array([[0.0, 0.0], [180.0, 20.0]]), 10.0)
    assert m.sum() == 2 and m[1, 0] and m[2, 2]


def test_no_storms():
    assert not box_mask(TLAT, TLON, np.empty((0, 2)), 10.0).any()
