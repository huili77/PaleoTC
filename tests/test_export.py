import dataclasses

import netCDF4
import numpy as np
import xarray as xr

from paleotc.config import load_config
from paleotc.export import export_year, pop_filename
from paleotc.extract import extract_year
from paleotc.stress import stress_year


def test_pop_filename_matches_get_forcing_filename():
    # forcing_tools.F90: root.YYYY.DDD.HH with 4/3/2-digit zero padding, day 1-based
    assert pop_filename("p", 101, 1, 0) == "p.0101.001.00.nc"
    assert pop_filename("p", 1130, 365, 18) == "p.1130.365.18.nc"


def test_export_files_are_pop_ready(fake_case):
    cfg = load_config(fake_case)
    extract_year(cfg, 31)
    stress_year(cfg, 31)
    summary = export_year(cfg, 31)
    names = sorted(p.name for p in cfg.export_dir.glob("*.nc"))
    assert names == ["fake.tctau.0101.001.00.nc", "fake.tctau.0101.001.06.nc"]  # HR 31 + 70
    assert summary["n_files"] == 2

    f = cfg.export_dir / names[0]
    with netCDF4.Dataset(f) as nc:
        assert nc.file_format == "NETCDF3_64BIT_OFFSET"
        for v in ("TAUX", "TAUY", "TCMASK"):
            assert nc[v].dimensions == ("nlat", "nlon")
            assert "_FillValue" not in nc[v].ncattrs()
    ds = xr.open_dataset(f)
    stress = xr.open_dataset(cfg.stress_path(31))
    np.testing.assert_array_equal(ds.TAUX.values, stress.TAUX.values[0])
    assert np.isfinite(ds.TAUX.values).all() and np.isfinite(ds.TAUY.values).all()


def test_export_respects_hours(fake_case):
    cfg = load_config(fake_case)
    cfg = dataclasses.replace(cfg, export=dataclasses.replace(cfg.export, hours=(0,)))
    extract_year(cfg, 31)
    stress_year(cfg, 31)
    assert export_year(cfg, 31)["n_files"] == 1
