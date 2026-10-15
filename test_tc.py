"""Tests on a tiny synthetic case: 12 'native' columns map 1:1 onto a 3x4 ocean grid."""
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

from extract_tcbot import RDAIR, ZVIR, box_mask, extract_year, read_tracks

NLAT, NLON = 3, 4
LATS = np.array([-20.0, 0.0, 20.0])
LONS = np.array([0.0, 90.0, 180.0, 270.0])
TLAT = np.tile(LATS[:, None], (1, NLON))
TLON = np.tile(LONS, (NLAT, 1))
TIMES = [cftime.DatetimeNoLeap(31, 1, 1, h) for h in (0, 6, 12, 18)]
TIME_ENC = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}
RHO = 0.99e5 / (RDAIR * 300 * (1 + ZVIR * 0.02))  # PBOT = 0.99*PS, TREFHT 300 K, QREFHT 0.02
# 00Z: storms at (0,0) and (0,-10) [box of the 2nd covers only the land point (0,0)]
# 06Z: storms at (2,0.5) and (-5,15) [wraps to 355E, box reaches the 0E column]
# 12-31 18Z: not in the HR file -> warning, skipped
TRACKS = (
    "start\t2\t31\t1\t1\t0\n"
    "\t1\t0.0\t0.0\t9.9e+04\t30.0\t0.0\t0.0\t31\t1\t1\t0\n"
    "\t2\t2.0\t0.5\t9.9e+04\t30.0\t0.0\t0.5\t31\t1\t1\t6\n"
    "start\t1\t31\t1\t1\t6\n"
    "\t3\t-5.0\t15.0\t9.8e+04\t40.0\t0.0\t15.0\t31\t1\t1\t6\n"
    "start\t1\t31\t1\t1\t0\n"
    "\t4\t0.0\t-10.0\t9.9e+04\t30.0\t0.0\t10.0\t31\t1\t1\t0\n"
    "start\t1\t31\t12\t31\t18\n"
    "\t5\t180.0\t-20.0\t9.9e+04\t30.0\t0.0\t20.0\t31\t12\t31\t18\n"
)


@pytest.fixture
def cfg(tmp_path):
    n = NLAT * NLON
    xr.Dataset({  # identity ESMF map
        "S": ("n_s", np.ones(n)), "row": ("n_s", np.arange(1, n + 1)), "col": ("n_s", np.arange(1, n + 1)),
        "dst_grid_dims": ("dst_grid_rank", [NLON, NLAT]),
        "frac_a": ("n_a", np.ones(n)), "frac_b": ("n_b", np.ones(n)),
    }).to_netcdf(tmp_path / "map.nc")
    mask = np.ones((NLAT, NLON), "i4")
    mask[0, 0] = 0  # land at (-20, 0)
    xr.Dataset({"xc": (("nj", "ni"), TLON), "yc": (("nj", "ni"), TLAT),
                "mask": (("nj", "ni"), mask)}).to_netcdf(tmp_path / "domain.nc")
    ones = np.ones((len(TIMES), n))
    fields = {"UBOT": (10.0 + np.arange(len(TIMES)))[:, None] * ones, "VBOT": -5 * ones,
              "U10": 20 * ones, "PS": 1e5 * ones, "TREFHT": 300 * ones, "QREFHT": 0.02 * ones}
    (tmp_path / "hr").mkdir()
    for v, d in fields.items():
        xr.Dataset({v: (("time", "ncol"), d.astype("f4")), "hyam": ("lev", [0.1, 0.0]),
                    "hybm": ("lev", [0.5, 0.99]), "P0": 1e5}, coords={"time": TIMES}).to_netcdf(
            tmp_path / "hr" / f"fake.cam.h2.{v}.0031010100-0032010100.nc", encoding={"time": TIME_ENC})
    (tmp_path / "tracks.txt").write_text(TRACKS)
    return {"case": "fake", "hr_dir": str(tmp_path / "hr"), "track_file": str(tmp_path / "tracks.txt"),
            "map_file": str(tmp_path / "map.nc"), "ocn_domain_file": str(tmp_path / "domain.nc"),
            "box_half_width": 10.0, "out_bot_dir": str(tmp_path / "bot"),
            "out_tau_dir": str(tmp_path / "tau"), "tau_prefix": "fake.tctau",
            "lr_year_offset": 70, "tau_hours": [0, 6, 12, 18]}


def test_read_tracks_groups_by_time(cfg):
    tracks = read_tracks(cfg["track_file"])
    assert sorted(tracks[(31, 1, 1, 6)]) == [(2.0, 0.5), (355.0, 15.0)]
    assert len(tracks[(31, 1, 1, 0)]) == 2


def test_box_wraps_longitude():
    m = box_mask(TLAT, TLON, [(355.0, 15.0)], 10.0)
    assert m.sum() == 1 and m[2, 0]


def test_extract_end_to_end(cfg, capsys):
    assert extract_year(31, cfg) == 2
    assert "(31, 12, 31, 18)" in capsys.readouterr().out
    a = xr.open_dataset(Path(cfg["out_bot_dir"]) / "tcbot.0031-01-01-00.nc")
    b = xr.open_dataset(Path(cfg["out_bot_dir"]) / "tcbot.0031-01-01-06.nc")
    assert a.TCMASK.values.sum() == 1 and a.TCMASK.values[1, 0] == 1  # land box masked out
    assert a.attrs["ntc"] == 2
    assert b.TCMASK.values.sum() == 2 and b.TCMASK.values[2, 0] == 1  # wrapped storm
    assert a.UBOT.values[1, 0] == 10.0 and b.UBOT.values[1, 0] == 11.0
    assert a.U10.values[0, 0] == 0.0 and a.UBOT.values[2, 2] == 0.0
    assert a.RHO.values[1, 0] == pytest.approx(RHO, rel=1e-6)


def test_rerun_overwrites(cfg):
    extract_year(31, cfg)
    extract_year(31, cfg)
    assert len(list(Path(cfg["out_bot_dir"]).glob("*"))) == 2
