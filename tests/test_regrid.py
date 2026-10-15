from pathlib import Path

import numpy as np
import pytest

from conftest import NCOL, NLAT, NLON, write_map
from paleotc.regrid import Regridder

REAL_MAP = Path(
    "/glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne120_g16/map_ne120np4_TO_gx1eocene_blin.260410.nc"
)


def test_identity_map_reshapes_to_grid(tmp_path):
    write_map(tmp_path / "m.nc")
    rg = Regridder(tmp_path / "m.nc")
    assert rg.shape == (NLAT, NLON)
    data = np.arange(2 * NCOL, dtype="f4").reshape(2, NCOL)
    out = rg(data)
    assert out.shape == (2, NLAT, NLON)
    np.testing.assert_array_equal(out[1], data[1].reshape(NLAT, NLON))


def test_weights_are_applied(tmp_path):
    # dst 1 = 0.25*src1 + 0.75*src2 ; all other dst points have no weights
    write_map(tmp_path / "m.nc", weights=([1, 1], [1, 2], [0.25, 0.75]))
    out = Regridder(tmp_path / "m.nc")(np.arange(1.0, NCOL + 1))
    assert out[0, 0] == pytest.approx(0.25 * 1 + 0.75 * 2)
    assert np.count_nonzero(out) == 1


def test_wrong_source_size_raises(tmp_path):
    write_map(tmp_path / "m.nc")
    with pytest.raises(ValueError, match="n_a"):
        Regridder(tmp_path / "m.nc")(np.zeros(NCOL + 1))


@pytest.mark.glade
@pytest.mark.skipif(not REAL_MAP.exists(), reason="GLADE map file not available")
def test_real_map_rows_sum_to_one():
    rg = Regridder(REAL_MAP)
    assert rg.n_a == 777602 and rg.shape == (384, 320)
    sums = np.asarray(rg.matrix.sum(axis=1)).ravel()
    covered = sums > 0
    np.testing.assert_allclose(sums[covered], 1.0, atol=1e-6)
