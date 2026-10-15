import numpy as np
import pytest
import xarray as xr

from conftest import RHO_EXPECTED_INPUTS
from paleotc.config import load_config
from paleotc.extract import extract_year
from paleotc.flux import air_density


def test_extract_selects_tc_times_and_boxes(fake_case):
    cfg = load_config(fake_case)
    summary = extract_year(cfg, 31)
    assert summary["n_times"] == 2
    ds = xr.open_dataset(cfg.boxed_path(31))
    assert [t.hour for t in ds.time.values] == [0, 6]
    assert ds.ntc.values.tolist() == [2, 2]
    # 00Z: only (j=1,i=0); 06Z: (1,0) and (2,0) via the wrapped storm at 355E
    assert ds.TCMASK.values[0].sum() == 1 and ds.TCMASK.values[0, 1, 0] == 1
    assert ds.TCMASK.values[1].sum() == 2 and ds.TCMASK.values[1, 2, 0] == 1
    assert ds.UBOT.values[0, 1, 0] == 10.0 and ds.UBOT.values[1, 1, 0] == 11.0
    assert ds.UBOT.values[0, 2, 2] == 0.0  # outside boxes
    assert ds.RHO.values[0, 1, 0] == pytest.approx(air_density(*RHO_EXPECTED_INPUTS), rel=1e-6)


def test_extract_masks_land(fake_case):
    cfg = load_config(fake_case)
    extract_year(cfg, 31)
    ds = xr.open_dataset(cfg.boxed_path(31))
    # storm at (0,-10) boxes only the land point (0,0): counted, but masked out
    assert ds.ntc.values[0] == 2
    assert ds.TCMASK.values[:, 0, 0].sum() == 0 and ds.U10.values[:, 0, 0].sum() == 0


def test_extract_reports_missing_times(fake_case, caplog):
    cfg = load_config(fake_case)
    summary = extract_year(cfg, 31)
    assert summary["n_missing"] == 1
    assert "(31, 12, 31, 18)" in caplog.text


def test_extract_rerun_overwrites(fake_case):
    cfg = load_config(fake_case)
    extract_year(cfg, 31)
    extract_year(cfg, 31)
    assert list(cfg.boxed_path(31).parent.glob("*.tmp")) == []
    assert xr.open_dataset(cfg.boxed_path(31)).sizes["time"] == 2
