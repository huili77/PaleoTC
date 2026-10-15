"""Synthetic mini case: 12 native columns map 1:1 onto a 3x4 ocean grid."""
import cftime
import numpy as np
import pytest
import xarray as xr
import yaml

CASE = "fake.case"
NLAT, NLON = 3, 4
NCOL = NLAT * NLON
LATS = np.array([-20.0, 0.0, 20.0])
LONS = np.array([0.0, 90.0, 180.0, 270.0])
TIMES = [cftime.DatetimeNoLeap(31, 1, 1, h) for h in (0, 6, 12, 18)] + [
    cftime.DatetimeNoLeap(32, 1, 1, 0)
]
# PS, TREFHT, QREFHT, hyam_bot, hybm_bot, P0 used in the fake h2 files
RHO_EXPECTED_INPUTS = (1.0e5, 300.0, 0.02, 0.0, 0.99, 1.0e5)
TIME_ENCODING = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}

# Track A: (0,0)@00 and (2,0.5)@06.  Track B: lon -5 (=355) lat 15 @06, which tests the wrap.
# Track C: (0,-10)@00 covers only the land point (j=0,i=0).
# Track D: 12-31 18 is not in the HR file, so it is reported as missing.
TRACKS = (
    "start\t2\t31\t1\t1\t0\n"
    "\t1\t0.000000\t0.000000\t9.9e+04\t3.0e+01\t0.0\t0.0\t31\t1\t1\t0\n"
    "\t2\t2.000000\t0.500000\t9.9e+04\t3.0e+01\t0.0\t0.5\t31\t1\t1\t6\n"
    "start\t1\t31\t1\t1\t6\n"
    "\t3\t-5.000000\t15.000000\t9.8e+04\t4.0e+01\t0.0\t15.0\t31\t1\t1\t6\n"
    "start\t1\t31\t1\t1\t0\n"
    "\t4\t0.000000\t-10.000000\t9.9e+04\t3.0e+01\t0.0\t10.0\t31\t1\t1\t0\n"
    "start\t1\t31\t12\t31\t18\n"
    "\t5\t180.000000\t-20.000000\t9.9e+04\t3.0e+01\t0.0\t20.0\t31\t12\t31\t18\n"
)


def write_map(path, weights=None):
    """ESMF-style map file. Default: identity n_a = n_b = NCOL."""
    if weights is None:
        rows = cols = np.arange(1, NCOL + 1, dtype="i4")
        s = np.ones(NCOL)
    else:
        rows, cols, s = (np.asarray(a) for a in weights)
    xr.Dataset(
        {
            "S": ("n_s", s),
            "row": ("n_s", rows.astype("i4")),
            "col": ("n_s", cols.astype("i4")),
            "dst_grid_dims": ("dst_grid_rank", np.array([NLON, NLAT], dtype="i4")),
            "frac_a": ("n_a", np.ones(NCOL)),
            "frac_b": ("n_b", np.ones(NCOL)),
        }
    ).to_netcdf(path)


def write_domain(path):
    mask = np.ones((NLAT, NLON), dtype="i4")
    mask[0, 0] = 0  # one land point at lat -20, lon 0
    xr.Dataset(
        {
            "xc": (("nj", "ni"), np.tile(LONS, (NLAT, 1))),
            "yc": (("nj", "ni"), np.tile(LATS[:, None], (1, NLON))),
            "mask": (("nj", "ni"), mask),
        }
    ).to_netcdf(path)


def write_h2(hr_dir):
    nt = len(TIMES)
    ones = np.ones((nt, NCOL))
    ps, trefht, qrefht, _, _, p0 = RHO_EXPECTED_INPUTS
    fields = {
        "UBOT": (10.0 + np.arange(nt))[:, None] * ones,  # 10, 11, 12, ... per time
        "VBOT": -5.0 * ones,
        "U10": 20.0 * ones,
        "PS": ps * ones,
        "TREFHT": trefht * ones,
        "QREFHT": qrefht * ones,
    }
    for var, data in fields.items():
        xr.Dataset(
            {
                var: (("time", "ncol"), data.astype("f4")),
                "hyam": ("lev", [0.1, 0.0]),
                "hybm": ("lev", [0.5, 0.99]),
                "P0": p0,
            },
            coords={"time": TIMES},
        ).to_netcdf(
            hr_dir / f"{CASE}.cam.h2.{var}.0031010100-0032010100.nc",
            encoding={"time": TIME_ENCODING},
        )


@pytest.fixture
def fake_case(tmp_path):
    hr = tmp_path / "hr"
    hr.mkdir()
    write_h2(hr)
    (tmp_path / "tracks.txt").write_text(TRACKS)
    write_map(tmp_path / "map.nc")
    write_domain(tmp_path / "domain.nc")
    cfg = {
        "case": CASE,
        "hr_dir": str(hr),
        "hr_stream": "h2",
        "track_file": str(tmp_path / "tracks.txt"),
        "first_year": 31,
        "last_year": 31,
        "map_file": str(tmp_path / "map.nc"),
        "ocn_domain_file": str(tmp_path / "domain.nc"),
        "box_half_width": 10.0,
        "time_chunk": 1,  # forces multiple chunks even in the tiny case
        "output_root": str(tmp_path / "out"),
        "export": {"prefix": "fake.tctau", "lr_year_offset": 70, "hours": [0, 6, 12, 18]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
