import numpy as np
import pytest
import xarray as xr

from conftest import RHO_EXPECTED_INPUTS
from paleotc.config import load_config
from paleotc.extract import extract_year
from paleotc.flux import air_density, wind_stress
from paleotc.stress import stress_year


def test_stress_year(fake_case):
    cfg = load_config(fake_case)
    extract_year(cfg, 31)
    summary = stress_year(cfg, 31)
    ds = xr.open_dataset(cfg.stress_path(31))
    rho = air_density(*RHO_EXPECTED_INPUTS)
    taux, tauy = wind_stress(10.0, -5.0, 20.0, rho)
    assert ds.TAUX.values[0, 1, 0] == pytest.approx(taux, rel=1e-5)
    assert ds.TAUY.values[0, 1, 0] == pytest.approx(tauy, rel=1e-5)
    assert ds.TAUX.values[0][ds.TCMASK.values[0] == 0].max() == 0.0
    assert np.isfinite(ds.TAUX.values).all()
    assert summary["n_times"] == 2 and summary["max_tau"] > 0
    assert ds.TAUX.attrs["units"] == "N/m2"
