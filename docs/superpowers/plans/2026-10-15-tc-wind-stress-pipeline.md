# PaleoTC TC Wind-Stress Scripts — Implementation Plan (simplified)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (chosen: native/inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn TC winds from the Eocene HR run into the wind-stress files that the LR-TC POP SourceMods read. This is done with a few plain Python scripts that translate the Li et al. 2023 NCL and notebook workflow.

**Architecture:** There are two scripts, mirroring the old workflow. `extract_tcbot.py YEAR` translates `extract_tcbot_batch.ncl` and writes one boxed-wind file per TC time step. `get_tau.py YEAR` replaces `get_tau_dask.ipynb`: it turns those files into POP-ready stress files. `run_casper.py FIRST LAST` runs both scripts for many years in parallel through dask-jobqueue on Casper. Every setting lives in `config.yaml`, which is read with `yaml.safe_load`.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, xarray, netCDF4, cftime, PyYAML, dask and dask-jobqueue, matplotlib (QC only), pytest. Uses the conda env `/glade/work/huili7/miniconda3/envs/paleotc`, which already exists. No package install is needed.

**Spec:** `docs/paleo_tc.docx`, plus the decisions below (agreed in the 2026-10-15 discussion).

**History:** This replaces the first version of the plan (a `paleotc` package with 9 modules). Its Tasks 1–9 were implemented as commits `55460e4..57eefaa` on `feature/tc-stress-pipeline`. Task 1 below removes that code. It stays in git history, but nothing here depends on it.

## Design decisions (unchanged from v1; each one affects the results)

| Topic | Decision | Evidence |
|---|---|---|
| HR input | `cam.h2` files (instantaneous 6-hourly). `h4` is a time mean. | `h4` variables have `cell_methods = "time: mean"` |
| Target grid | LR Eocene ocean grid `gx1eocene` (320×384). Its geometry is **not** standard gx1v6 (latitudes differ by up to about 29°). | Compared `grid.1.pop.da` with `horiz_grid_20010402.ieeer8` |
| Weights | `/glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne120_g16/map_ne120np4_TO_gx1eocene_blin.260410.nc` (bilinear) | Destination points and mask match the LR domain file |
| Ocean grid/mask | `/glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne30_g16/domain.ocn.gx1eocene.260408.nc` (`xc`, `yc`, `mask`) | This is the LR case's `OCN_DOMAIN_FILE` |
| u10n | Use CAM `U10` as the neutral 10 m wind: `ustar = sqrt(cdn(U10))·U10` | `cam_in%u10 = Sx_u10 = sqrt(duu10n)` |
| cdn | `0.0027/U + 0.000142 + 0.0000764·U`, with no high-wind branch | iCESM1.3.1 `shr_flux_mod.F90:183` |
| Air density | `RHO = PBOT/(Rd·TREFHT·(1+zvir·QREFHT))`, with `PBOT = hyam(bottom)·P0 + hybm(bottom)·PS` | Monthly check: about −0.2%; a fixed 1.2 is +6% (about +12% in TC cores) |
| Direction | Along (UBOT, VBOT), east/north. Ocean currents are ignored. | Coupler `WORK1`/`WORK2` are east/north |
| Box | Half-open box: `[-10°, 10°)` in latitude and in wrapped longitude around each center. The union over all storms, AND the ocean mask. | Same as the Li et al. 2023 NCL, plus the longitude-wrap fix |
| Coupler | **Replace** the coupler stress where `TCMASK = 1` (your decision). The Li et al. 2023 SourceMod *added* it. | Task 7 |
| File name | `{tau_prefix}.{YYYY}.{DDD}.{HH}.nc`. YYYY = HR (track) year + `lr_year_offset` (0); DDD = 1-based noleap day of year. POP maps model year → track year itself, as a repeating 30-year cycle (Task 7). | POP `get_forcing_filename` format; cycle in `check_tc_ws` |
| Hours exported | 00, 06, 12, 18. POP only reads 00Z because the ocean couples daily, but a missing file is skipped silently, so all hours are written. | `OCN_NCPL=1` |

## Global Constraints

- Scripts sit at the repo root and take `YEAR` (and optionally a config path) as command-line arguments. No package, no `pip install`.
- Use the `noleap` calendar. Match track times to model times with `(year, month, day, hour)` tuples.
- `scripts_Li2023/` is read-only reference code.
- Units: m/s, kg/m³, N/m². POP applies its own `momentum_factor`.
- Tau files: `NETCDF3_64BIT` format, dims `(nlat, nlon)`, float32 `TAUX`, `TAUY` and `TCMASK`, **no `_FillValue`**, and no NaN or Inf.

## Review Focus

1. **Storm near 0°/360° longitude:** the box must wrap (`test_box_wraps_longitude`).
2. **Track time missing from the HR output** (for example `0061-01-01 00`): print a warning and skip it, without crashing (`test_extract_end_to_end`).
3. **Box over land:** TCMASK and all fields must be 0 there (`test_extract_end_to_end`).
4. **Calm wind:** no NaN or Inf (`test_wind_stress_calm_is_finite`).
5. **Rerunning a year:** files are overwritten in place, with no leftovers (`test_rerun_overwrites`).

---

## Layout (final)

```
PaleoTC/
├── config.yaml          # every path and parameter
├── extract_tcbot.py     # HR h2 → tc_bot/tcbot.YYYY-MM-DD-HH.nc  (UBOT VBOT U10 RHO TCMASK)
├── get_tau.py           # tc_bot → tc_tau/{prefix}.YYYY.DDD.HH.nc (TAUX TAUY TCMASK)
├── run_casper.py        # PBSCluster on Casper: both scripts for a range of years
├── qc_tau.py            # stats + plot of the strongest snapshot of one year
├── test_tc.py           # 7 tests on a tiny synthetic case
├── environment.yml  README.md  .gitignore
├── sourcemods/src.pop/  # LR-TC POP SourceMods (Task 7)
├── docs/                # spec, plan, WORKLOG.md, notes/get_tau_review.md
└── scripts_Li2023/      # reference (read-only)
```

Outputs, each in its own directory under `/glade/derecho/scratch/huili7/PaleoTC/`: `tc_bot/` (zlib-compressed, small) and `tc_tau/` (about 1.5 MB per file × about 1,400 files per year). Scratch is purged after 180 days, so copy `tc_tau/` to `/glade/work` or campaign storage.

---

### Task 1: Remove the package; add a plain `config.yaml`

**Files:**
- Delete: `src/`, `tests/`, `pyproject.toml`, `config/`
- Create: `config.yaml`
- Modify: `.gitignore`, `docs/WORKLOG.md`

- [ ] **Step 1: Uninstall and remove the package**

```bash
cd /glade/work/huili7/PaleoTC
module load conda && conda activate /glade/work/huili7/miniconda3/envs/paleotc
pip uninstall -y paleotc
git rm -r -q src tests pyproject.toml config
```

- [ ] **Step 2: Write `config.yaml`**

```yaml
# Eocene 3x CO2: HR (ne120_t12) TC winds -> LR (ne30_g16, gx1eocene) wind stress
case: b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002
hr_dir: /glade/campaign/cesm/community/palwg/ASD_paleoWeather/b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002/atm/proc/tseries/hour_6
track_file: /glade/campaign/cgd/ppc/brady/paleoweather/TCs/Eo3x/F3/tracks/TC_Eo3x.002_F3tracks_yrs31-60.txt
map_file: /glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne120_g16/map_ne120np4_TO_gx1eocene_blin.260410.nc
ocn_domain_file: /glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne30_g16/domain.ocn.gx1eocene.260408.nc
box_half_width: 10.0          # degrees -> 20x20 box

out_bot_dir: /glade/derecho/scratch/huili7/PaleoTC/tc_bot
out_tau_dir: /glade/derecho/scratch/huili7/PaleoTC/tc_tau
tau_prefix: b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002.tctau
lr_year_offset: 0             # LR-TC model year = HR year + offset (revisit before LR-TC run)
tau_hours: [0, 6, 12, 18]

casper:
  account: P93300014
  memory: 16GB
  walltime: "02:00:00"
  max_jobs: 30
  log_dir: /glade/derecho/scratch/huili7/PaleoTC/dask-logs
```

- [ ] **Step 3: Update `.gitignore`**

```
*~
__pycache__/
.pytest_cache/
dask-worker-space/
*.png
```

- [ ] **Step 4: Commit**

```bash
git add -A config.yaml .gitignore
git commit -m "Replace paleotc package with plain config.yaml (simplified plan)"
```

---

### Task 2: `extract_tcbot.py` (translation of `extract_tcbot_batch.ncl`)

**Files:**
- Create: `extract_tcbot.py`, `test_tc.py`

**Interfaces:**
- Produces: `read_tracks(path) -> dict[(y,m,d,h), list[(lon, lat)]]`; `box_mask(tlat, tlon, centers, half_width) -> bool array`; `extract_year(year, cfg) -> int` (the number of TC time steps written). Output files are `out_bot_dir/tcbot.YYYY-MM-DD-HH.nc` with `UBOT, VBOT, U10, RHO` (float32), `TCMASK` (int8), scalar `time`, and attribute `ntc`.

- [ ] **Step 1: Write the failing tests (`test_tc.py`, extraction part)**

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest test_tc.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'extract_tcbot'`

- [ ] **Step 3: Write `extract_tcbot.py`**

```python
"""Extract TC winds and air density from the HR run, one year at a time.

Python version of scripts_Li2023/extract_tcbot_batch.ncl, adapted to the Eocene runs:
  - reads the instantaneous 6-hourly h2 files (UBOT, VBOT, U10; PS, TREFHT, QREFHT)
  - regrids ne120 -> LR ocean grid (gx1eocene) with the ESMF bilinear weights
  - keeps 20x20 deg boxes around each TC center (longitude wrap fixed); zero elsewhere and on land
  - adds air density RHO = PBOT / (Rd*TREFHT*(1+zvir*QREFHT)), PBOT from hybrid coefficients
Writes one file per TC time step: <out_bot_dir>/tcbot.YYYY-MM-DD-HH.nc

Usage: python extract_tcbot.py YEAR [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xarray as xr
import yaml

RDAIR = 6.02214e26 * 1.38065e-23 / 28.966  # CESM SHR_CONST_RDAIR
ZVIR = 28.966 / 18.016 - 1.0               # CESM SHR_CONST_ZVIR = RWV/RDAIR - 1
CHUNK = 120                                # time steps read at once (~2 GB)
HR_VARS = ("UBOT", "VBOT", "U10", "PS", "TREFHT", "QREFHT")
TIME_ENC = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}


def read_tracks(path):
    """TempestExtremes trajectory file -> {(year, month, day, hour): [(lon, lat), ...]}."""
    centers = {}
    for line in open(path):
        parts = line.split()
        if not parts or parts[0] == "start":
            continue
        key = tuple(int(v) for v in parts[-4:])
        centers.setdefault(key, []).append((float(parts[1]) % 360.0, float(parts[2])))
    return centers


def read_weights(map_file):
    """ESMF weight file -> sparse matrix (n_b x n_a) and destination shape (nlat, nlon)."""
    with xr.open_dataset(map_file) as w:
        nlon, nlat = (int(v) for v in w.dst_grid_dims.values)
        matrix = sp.csr_matrix((w.S.values, (w.row.values - 1, w.col.values - 1)),
                               shape=(w.sizes["n_b"], w.sizes["n_a"]))
    return matrix, (nlat, nlon)


def box_mask(tlat, tlon, centers, half_width):
    """True where a point is within [-hw, hw) of any center in latitude and wrapped longitude."""
    mask = np.zeros(tlat.shape, dtype=bool)
    for lon, lat in centers:
        dlon = (tlon - lon + 180.0) % 360.0 - 180.0
        dlat = tlat - lat
        mask |= (dlon >= -half_width) & (dlon < half_width) & (dlat >= -half_width) & (dlat < half_width)
    return mask


def hr_file(cfg, var, year):
    pattern = f"{cfg['case']}.cam.h2.{var}.{year:04d}01*.nc"
    files = sorted(Path(cfg["hr_dir"]).glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"{pattern} in {cfg['hr_dir']}: found {len(files)} files")
    return files[0]


def extract_year(year, cfg):
    tracks = {k: v for k, v in read_tracks(cfg["track_file"]).items() if k[0] == year}
    matrix, shape = read_weights(cfg["map_file"])
    with xr.open_dataset(cfg["ocn_domain_file"]) as dom:
        tlat, tlon, ocean = dom.yc.values, dom.xc.values % 360.0, dom.mask.values == 1
    hr = {v: xr.open_dataset(hr_file(cfg, v, year)) for v in HR_VARS}

    ps = hr["PS"]
    pbot_a = float(ps.hyam[-1] * ps.P0)  # PBOT = pbot_a + pbot_b * PS
    pbot_b = float(ps.hybm[-1])
    times = ps.time.values
    keys = [(t.year, t.month, t.day, t.hour) for t in times]
    itc = [i for i, k in enumerate(keys) if k in tracks]
    for k in sorted(set(tracks) - set(keys)):
        print(f"warning: track time {k} not in HR output, skipped")

    out_dir = Path(cfg["out_bot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(itc), CHUNK):
        idx = itc[start:start + CHUNK]
        f = {v: hr[v][v].isel(time=idx).values.astype("f8") for v in HR_VARS}
        f["RHO"] = (pbot_a + pbot_b * f["PS"]) / (RDAIR * f["TREFHT"] * (1.0 + ZVIR * f["QREFHT"]))
        on_ocn = {v: (matrix @ f[v].T).T.reshape(len(idx), *shape) for v in ("UBOT", "VBOT", "U10", "RHO")}
        for n, i in enumerate(idx):
            t = times[i]
            mask = box_mask(tlat, tlon, tracks[keys[i]], cfg["box_half_width"]) & ocean
            ds = xr.Dataset(
                {v: (("nlat", "nlon"), np.where(mask, on_ocn[v][n], 0.0).astype("f4")) for v in on_ocn}
                | {"TCMASK": (("nlat", "nlon"), mask.astype("i1"))},
                coords={"time": t},
                attrs={"ntc": len(tracks[keys[i]]), "source": cfg["case"], "map_file": cfg["map_file"]},
            )
            enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars} | {"time": TIME_ENC}
            name = f"tcbot.{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}.nc"
            ds.to_netcdf(out_dir / name, encoding=enc)
        print(f"year {year}: {start + len(idx)}/{len(itc)} TC time steps written")
    for ds in hr.values():
        ds.close()
    return len(itc)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "config.yaml"))
    extract_year(int(sys.argv[1]), cfg)
```

- [ ] **Step 4: Run it and confirm it passes**

Run: `pytest test_tc.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add extract_tcbot.py test_tc.py
git commit -m "Add extract_tcbot.py: Python translation of extract_tcbot_batch.ncl"
```

---

### Task 3: `get_tau.py` (replaces `get_tau_dask.ipynb`)

**Files:**
- Create: `get_tau.py`
- Modify: `test_tc.py` (append tests)

**Interfaces:**
- Consumes: `tcbot.*.nc` from Task 2.
- Produces: `cdn(u)`, `wind_stress(ubot, vbot, u10, rho) -> (taux, tauy)`, `pop_filename(prefix, year, doy, hour) -> str`, `get_tau_year(year, cfg) -> int` (the number of tau files written).

- [ ] **Step 1: Append the failing tests to `test_tc.py`**

```python
import netCDF4

from get_tau import get_tau_year, pop_filename, wind_stress


def test_wind_stress_magnitude_and_direction():
    taux, tauy = wind_stress(np.array(30.0), np.array(40.0), np.array(40.0), np.array(1.1))
    tau = 1.1 * (0.0027 / 40 + 0.000142 + 0.0000764 * 40) * 40.0**2
    assert taux == pytest.approx(0.6 * tau) and tauy == pytest.approx(0.8 * tau)


def test_wind_stress_calm_is_finite():
    z = np.zeros(3)
    taux, tauy = wind_stress(z, z, z, np.full(3, 1.1))
    assert np.isfinite(taux).all() and np.isfinite(tauy).all() and (taux == 0).all()


def test_get_tau_writes_pop_files(cfg):
    assert pop_filename("p", 101, 1, 0) == "p.0101.001.00.nc"  # POP get_forcing_filename format
    extract_year(31, cfg)
    assert get_tau_year(31, cfg) == 2
    f = Path(cfg["out_tau_dir"]) / "fake.tctau.0101.001.06.nc"  # HR year 31 + offset 70, day 1, 06Z
    with netCDF4.Dataset(f) as nc:
        assert nc.file_format == "NETCDF3_64BIT_OFFSET"
        assert all("_FillValue" not in nc[v].ncattrs() for v in ("TAUX", "TAUY", "TCMASK"))
    ds = xr.open_dataset(f)
    taux, _ = wind_stress(11.0, -5.0, 20.0, RHO)
    assert ds.TAUX.values[1, 0] == pytest.approx(taux, rel=1e-5)
    assert ds.TAUX.values[0, 0] == 0.0 and np.isfinite(ds.TAUY.values).all()
    assert ds.TCMASK.values.sum() == 2
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest test_tc.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'get_tau'`

- [ ] **Step 3: Write `get_tau.py`**

```python
"""Compute TC wind stress from extract_tcbot.py output; write the files the POP SourceMod reads.

Replaces scripts_Li2023/get_tau_dask.ipynb (see docs/notes/get_tau_review.md for the fixes):
  tau        = RHO * cdn(U10) * U10**2      (U10 = neutral 10 m wind from shr_flux_atmOcn)
  TAUX, TAUY = tau * (UBOT, VBOT) / |(UBOT, VBOT)|   eastward/northward, N/m2
Output: <out_tau_dir>/<tau_prefix>.YYYY.DDD.HH.nc with YYYY = HR year + lr_year_offset

Usage: python get_tau.py YEAR [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

UMIN = 0.5  # shr_flux_atmOcn minimum wind speed (m/s)


def cdn(u):
    """Neutral 10 m drag coefficient, iCESM1.3.1 shr_flux_mod.F90 (no high-wind branch)."""
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def wind_stress(ubot, vbot, u10, rho):
    u10 = np.maximum(u10, UMIN)
    tau = rho * cdn(u10) * u10**2
    vmag = np.maximum(np.hypot(ubot, vbot), UMIN)
    return tau * ubot / vmag, tau * vbot / vmag


def pop_filename(prefix, year, doy, hour):
    """Same as POP forcing_tools.F90:get_forcing_filename (root.YYYY.DDD.HH), plus '.nc'."""
    return f"{prefix}.{year:04d}.{doy:03d}.{hour:02d}.nc"


def get_tau_year(year, cfg):
    out_dir = Path(cfg["out_tau_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(cfg["out_bot_dir"]).glob(f"tcbot.{year:04d}-*.nc")):
        with xr.open_dataset(f) as ds:
            t = ds.time.item()
            if t.hour not in cfg["tau_hours"]:
                continue
            taux, tauy = wind_stress(ds.UBOT.values, ds.VBOT.values, ds.U10.values, ds.RHO.values)
            mask = ds.TCMASK.values == 1
        units = {"units": "N/m2"}
        out = xr.Dataset(
            {"TAUX": (("nlat", "nlon"), np.where(mask, taux, 0.0).astype("f4"), units | {"long_name": "TC wind stress, eastward"}),
             "TAUY": (("nlat", "nlon"), np.where(mask, tauy, 0.0).astype("f4"), units | {"long_name": "TC wind stress, northward"}),
             "TCMASK": (("nlat", "nlon"), mask.astype("f4"), {"long_name": "1 where TC stress replaces coupler stress"})},
            attrs={"hr_time": t.isoformat(), "lr_year_offset": cfg["lr_year_offset"]},
        )
        name = pop_filename(cfg["tau_prefix"], t.year + cfg["lr_year_offset"], t.dayofyr, t.hour)
        out.to_netcdf(out_dir / name, format="NETCDF3_64BIT",
                      encoding={v: {"_FillValue": None} for v in out.data_vars})
        n += 1
    print(f"year {year}: {n} tau files written")
    return n


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "config.yaml"))
    get_tau_year(int(sys.argv[1]), cfg)
```

- [ ] **Step 4: Run it and confirm it passes**

Run: `pytest test_tc.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add get_tau.py test_tc.py
git commit -m "Add get_tau.py: TC wind stress and POP-ready files"
```

---

### Task 4: `run_casper.py` (replaces `sub-batch.sh` and the batched dask loop)

**Files:**
- Create: `run_casper.py`

**Interfaces:**
- Consumes: `extract_year(year, cfg)`, `get_tau_year(year, cfg)`.

- [ ] **Step 1: Write `run_casper.py`**

```python
"""Run extract_tcbot + get_tau for a range of HR years on Casper (one PBS job per year).

Usage (from a Casper login node, qinteractive session, or JupyterHub):
    python run_casper.py FIRST LAST [config.yaml]
"""
import sys
from pathlib import Path

import yaml
from dask.distributed import Client, as_completed
from dask_jobqueue import PBSCluster

from extract_tcbot import extract_year
from get_tau import get_tau_year

HERE = Path(__file__).resolve().parent


def run_year(year, cfg):
    return extract_year(year, cfg), get_tau_year(year, cfg)


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    cfg = yaml.safe_load(open(sys.argv[3] if len(sys.argv) > 3 else HERE / "config.yaml"))
    c = cfg["casper"]
    years = list(range(first, last + 1))

    cluster = PBSCluster(cores=1, processes=1, memory=c["memory"], account=c["account"],
                         queue="casper", walltime=c["walltime"], interface="ext",
                         resource_spec=f"select=1:ncpus=1:mem={c['memory']}",
                         log_directory=c["log_dir"])
    cluster.scale(jobs=min(len(years), c["max_jobs"]))
    client = Client(cluster)
    for script in ("extract_tcbot.py", "get_tau.py"):  # make the scripts importable on workers
        client.upload_file(str(HERE / script))

    futures = {client.submit(run_year, y, cfg, key=f"year-{y}"): y for y in years}
    failed = []
    for fut in as_completed(futures):
        y = futures[fut]
        try:
            n_bot, n_tau = fut.result()
            print(f"year {y}: {n_bot} TC time steps, {n_tau} tau files")
        except Exception as exc:
            failed.append(y)
            print(f"year {y}: FAILED {exc!r}")
    client.close()
    cluster.close()
    print(f"done; failed years: {failed or 'none'}")
```

- [ ] **Step 2: Check that it imports**

Run: `python -c "import run_casper"`
Expected: no output (the import works; nothing starts, because the cluster is created only under `__main__`)

- [ ] **Step 3: Commit**

```bash
git add run_casper.py
git commit -m "Add run_casper.py: dask-jobqueue driver for Casper"
```

---

### Task 5: Real run for year 31, QC, then all years

**Files:**
- Create: `qc_tau.py`

- [ ] **Step 1: Write `qc_tau.py`**

```python
"""Quick QC of one year's tau files: summary stats and a plot of the strongest snapshot.

Usage: python qc_tau.py YEAR [config.yaml]   -> prints stats, writes qc_tau_YYYY.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from extract_tcbot import read_tracks

year = int(sys.argv[1])
cfg = yaml.safe_load(open(sys.argv[2] if len(sys.argv) > 2 else "config.yaml"))
files = sorted(Path(cfg["out_tau_dir"]).glob(f"{cfg['tau_prefix']}.{year + cfg['lr_year_offset']:04d}.*.nc"))
grid = xr.open_dataset(cfg["ocn_domain_file"])

peak, worst, bad = 0.0, None, 0
for f in files:
    ds = xr.open_dataset(f)
    tau = np.hypot(ds.TAUX.values, ds.TAUY.values)
    bad += int((~np.isfinite(tau)).sum())
    if tau.max() > peak:
        peak, worst = float(tau.max()), (tau, ds.TCMASK.values == 1, ds.attrs["hr_time"])
print(f"{len(files)} files, max |tau| = {peak:.2f} N/m2, non-finite values = {bad}")

tau, mask, hr_time = worst
key = (int(hr_time[:4]), int(hr_time[5:7]), int(hr_time[8:10]), int(hr_time[11:13]))
centers = read_tracks(cfg["track_file"]).get(key, [])
fig, ax = plt.subplots(figsize=(11, 4.5))
sc = ax.scatter(grid.xc.values[mask], grid.yc.values[mask], c=tau[mask], s=3)
ax.plot([c[0] for c in centers], [c[1] for c in centers], "rx", ms=8, label="track centers")
fig.colorbar(sc, label="|tau| (N/m2)")
ax.set(xlim=(0, 360), ylim=(-60, 60), title=f"Strongest TC stress snapshot, HR {hr_time}")
ax.legend(loc="lower left")
fig.savefig(f"qc_tau_{year:04d}.png", dpi=120, bbox_inches="tight")
print(f"wrote qc_tau_{year:04d}.png")
```

- [ ] **Step 2: Run year 31 on Casper**

Run: `python run_casper.py 31 31`
Expected: `year 31: N TC time steps, N tau files`, with N between 500 and 1,460 and the two numbers equal; no `warning: track time` lines; `failed years: none`.

- [ ] **Step 3: QC**

Run: `python qc_tau.py 31`
Expected: `non-finite values = 0`; max |tau| between about 2 and 15 N/m² (U10 = 50 m/s gives about 11 N/m²). In the PNG, the filled boxes are centered on the red crosses and none are on land.

- [ ] **Step 4: Commit, then run all years**

```bash
git add qc_tau.py
git commit -m "Add qc_tau.py"
python run_casper.py 31 60
```

Expected: 30 `year … ` lines and `failed years: none`. The `61-01-01 00` track point belongs to year 61, which is not processed, so it produces no warning. Copy `tc_tau/` off scratch.

---

### Task 6: Notes for tasks 2b/2c, and README

**Files:**
- Create: `docs/notes/get_tau_review.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/notes/get_tau_review.md`**

```markdown
# Review of scripts_Li2023/get_tau_dask.ipynb

Reference: iCESM1.3.1 `cime/src/share/util/shr_flux_mod.F90` (`shr_flux_atmOcn`).
New code: `get_tau.py` (stress) and `extract_tcbot.py` (air density).

## (b) Does it reproduce the model's wind stress?

1. **Drag coefficient.** `cdn1` tests `umps.any() > 12.5`, which compares a bool to 12.5 and is always False, so the low-wind formula is always used. iCESM1.3.1 has *only* that formula, so the result is correct by accident. The high-wind branch must not be "fixed" back in.
2. **ustar from U10.** The notebook uses `sqrt(cdn(U10))·U10`, while the model iterates from UBOT at zbot. CAM's U10 is `Sx_u10 = sqrt(duu10n)` from this same routine, so the shortcut is consistent: measured within 0.9% (slightly high) for 10–70 m/s. Kept.
3. **Air density.** The notebook uses a fixed 1.2 kg/m³. That is +6% over the Eocene tropical ocean and about +12% in TC cores. Now `PBOT/(Rd·TREFHT·(1+zvir·QREFHT))`, which is within about 0.2%.
4. **Ocean currents** are ignored in the stress direction. The effect is unbiased, with ±6% scatter. Kept.
5. **"grid-x direction" label.** The coupler stress is eastward/northward, and POP rotates it by ANGLET. The values are right; the label was wrong.
6. **"Patch" remap label.** The weights are bilinear. The old ne120→gx1v6 weights are also on the wrong grid for the Eocene runs; use the gx1eocene map instead.
7. **Unused definitions.** `alz`, `qsat`, `psimhu` and `SHR_CONST_G` were defined but never used.

## (c) Parallelization

1. Batches of 20 `dask.delayed` calls with a barrier after each batch leave workers idle. Now: one task per year with `client.submit` and `as_completed`.
2. Tasks were very small (one ~1 MB file each). Now each task handles a whole year.
3. Filenames came from string slicing (`date_str[0:2]`, `'00'+yyy`), which breaks for years ≥ 100. Now: cftime `.year`, `.dayofyr` and `.hour`.
4. Each file was written in three passes (`to_netcdf` plus two appends) after an `os.remove`. Now: a single write.
5. `PBSCluster(project=...)` is deprecated; use `account=`. Run one worker per PBS job on Casper.
```

- [ ] **Step 2: Write `README.md`**

```markdown
# PaleoTC

Role of tropical cyclones in equable (Eocene) climates. TC winds from the HR Eocene run
(ne120/t12) are converted to surface wind stress on the LR ocean grid (gx1eocene) and
prescribed in LR-TC through POP SourceMods (Li et al. 2023 method).

## Workflow (Casper)

    module load conda
    conda activate /glade/work/huili7/miniconda3/envs/paleotc

    python extract_tcbot.py 31      # one year: HR h2 -> tc_bot/tcbot.YYYY-MM-DD-HH.nc
    python get_tau.py 31            # tc_bot -> tc_tau/<prefix>.YYYY.DDD.HH.nc (POP reads these)
    python run_casper.py 31 60      # all years in parallel (PBSCluster, one job per year)
    python qc_tau.py 31             # stats + snapshot plot
    pytest test_tc.py               # tests on a tiny synthetic case

All paths and parameters are in `config.yaml`. Notes: `docs/notes/get_tau_review.md`.
SourceMods for LR-TC: `sourcemods/src.pop/` (TC stress replaces coupler stress where TCMASK=1).
```

- [ ] **Step 3: Commit**

```bash
git add docs/notes/get_tau_review.md README.md
git commit -m "Add get_tau review notes and README"
```

---

### Task 7: Port the POP SourceMods to the Eocene LR case (replace semantics)

**Files:**
- Create: `sourcemods/src.pop/forcing_coupled.F90` (the LR case's file plus a hook)
- Create: `sourcemods/src.pop/forcing_ws.F90` (the iCESM1.3.1 file plus the TC subroutine)

Why: the Li et al. 2023 `forcing_coupled.F90` is based on a 2011 POP revision (26603; the tag has 87926), while the Eocene LR case already has its own newer `forcing_coupled.F90` SourceMod. That SourceMod holds Jiang Zhu's estuary changes, which compute `ROFF_L`, used by the case's `baroclinic.F90` and `forcing_fields.F90` SourceMods. **Do not use Li's `forcing_coupled.F90`:** it would drop `ROFF_L`, and with it the LR-cntl salt/runoff physics. Only its two hooks are ported. Li's `forcing_ws.F90` differs from the iCESM1.3.1 version only in the TC blocks, but those *add* the stress, ignore `TCMASK`, and hard-code the old CESM1 path (`tc_filename` is not a namelist variable). So the TC subroutine is rewritten below rather than copied. **Repeating track cycle (decided 2026-10-15, for runs such as 300 years):** POP's `get_forcing_filename` always uses the model year. Instead, `check_tc_ws` builds the name itself: track year = `tc_first_year + modulo(iyear - tc_model_year0, tc_nyears)`, with `tc_first_year=31`, `tc_nyears=30` and `tc_model_year0=1` (for now). So model years 1–30 read track years 31–60, 31–60 read 31–60 again, and so on. The format `prefix.YYYY.DDD.HH.nc` is the same as `get_tau.py` and `get_forcing_filename`. This replaces `lr_year_offset` for the POP side; keep it 0 in `config.yaml`. The cost is one seam every 30 years (31 Dec of year 60 → 1 Jan of year 31). Every symbol used below (`p5`, `c0`, `iyear`, `iday_of_year`, `ihour`, io routines) is already available through `forcing_ws`'s existing `use` statements.

- [ ] **Step 1: Copy the base files and commit them unmodified**

```bash
mkdir -p sourcemods/src.pop
cp /glade/u/home/jiangzhu/cesm/cases/b.e13.B1850C5.ne30_g16.icesm13_ihesp.eocene_03x.004/SourceMods/src.pop/forcing_coupled.F90 sourcemods/src.pop/
cp /glade/campaign/cesm/development/palwg/cesm_tags/iCESM1.3.1/components/pop/source/forcing_ws.F90 sourcemods/src.pop/
git add sourcemods && git commit -m "Add unmodified base files for LR-TC POP SourceMods"
```

- [ ] **Step 2: In `forcing_coupled.F90`, add the hook**

Replace `   use forcing_ws, only: ws_data_type` with:

```fortran
!----- PaleoTC mod: TC wind stress
   use forcing_ws, only: ws_data_type, check_tc_ws
!----- PaleoTC mod end
```

In `rotate_wind_stress`, change the intent of `WORK1, WORK2` from `intent(in)` to `intent(inout)`. Without this the build fails, because an `intent(in)` dummy is passed to an `intent(inout)` one. It is safe: the only caller (`drivers/cpl/ocn_import_export.F90:183`) passes local arrays. Replace

```fortran
   real (r8), dimension(nx_block,ny_block,max_blocks_clinic), intent(in) ::   &
      WORK1, WORK2        ! contains taux and tauy from coupler
```

with

```fortran
!----- PaleoTC mod: intent(in) -> intent(inout), check_tc_ws modifies WORK1/WORK2
   real (r8), dimension(nx_block,ny_block,max_blocks_clinic), intent(inout) ::   &
!----- PaleoTC mod end
      WORK1, WORK2        ! contains taux and tauy from coupler
```

Then, directly before the line `   SMFT(:,:,1,:) = (WORK1(:,:,:)*cos(ANGLET(:,:,:)) +  &`, insert:

```fortran
!----- PaleoTC mod: replace coupler stress inside TC boxes
   call check_tc_ws(WORK1,WORK2)
!----- PaleoTC mod end
```

- [ ] **Step 3: In `forcing_ws.F90`, add the TC code**

In the public list, change `             set_ws` to:

```fortran
             set_ws,  &
             check_tc_ws
```

After the `ws_interp_last` public declaration, add:

```fortran
!----- PaleoTC mod
   real (r8), public, dimension(nx_block,ny_block,2,max_blocks_clinic) :: &
      SMFT_TC       ! TC momentum flux read in, on T grid (east/north, N/m2)
   real (r8), dimension(nx_block,ny_block,max_blocks_clinic) :: &
      TC_MASK       ! 1 inside TC boxes over ocean, 0 elsewhere

   character (char_len), parameter :: tc_filename = &
      '/glade/derecho/scratch/huili7/PaleoTC/tc_tau/b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002.tctau'

   ! TC files exist for track years tc_first_year..tc_first_year+tc_nyears-1;
   ! model year tc_model_year0 reads track year tc_first_year, and the
   ! track years repeat every tc_nyears model years.
   integer (int_kind), parameter :: &
      tc_first_year  = 31, &! first track year with TC files
      tc_nyears      = 30, &! number of track years (cycle length)
      tc_model_year0 = 1    ! model year that reads tc_first_year
!----- PaleoTC mod end
```

Directly before `end module forcing_ws`, add:

```fortran
!----- PaleoTC mod
!***********************************************************************
!BOP
! !IROUTINE: check_tc_ws
! !INTERFACE:

 subroutine check_tc_ws(WORK1,WORK2)

! !DESCRIPTION:
!  If a TC wind-stress file exists for the current coupling time, replace
!  the coupler wind stress (east/north, N/m2) inside the TC boxes (TCMASK=1).
!  File name: tc_filename.YYYY.DDD.HH.nc, where YYYY is the track year that
!  the current model year maps to (repeating cycle), DDD the day of year.

   real (r8), dimension(nx_block,ny_block,max_blocks_clinic), intent(inout) :: &
      WORK1, WORK2        ! taux and tauy from coupler

   integer (int_kind) :: iblock, tc_year
   character (char_len) :: forcing_filename
   type (datafile) :: forcing_file
   type (io_field_desc) :: io_taux, io_tauy, io_mask
   type (io_dim) :: i_dim, j_dim
   logical (log_kind) :: iexist

   tc_year = tc_first_year + modulo(iyear - tc_model_year0, tc_nyears)
   write(forcing_filename,'(a,".",i4.4,".",i3.3,".",i2.2,".nc")') &
         trim(tc_filename), tc_year, iday_of_year, ihour
   inquire (file=trim(forcing_filename), exist=iexist)
   if (.not. iexist) return

   SMFT_TC = c0
   TC_MASK = c0
   forcing_file = construct_file('nc', full_name=trim(forcing_filename), &
                                 record_length=rec_type_dbl,             &
                                 recl_words=nx_global*ny_global)
   call data_set(forcing_file,'open_read')
   i_dim = construct_io_dim('i',nx_global)
   j_dim = construct_io_dim('j',ny_global)
   io_taux = construct_io_field('TAUX', dim1=i_dim, dim2=j_dim,          &
                field_loc=field_loc_center, field_type=field_type_vector, &
                d2d_array=SMFT_TC(:,:,1,:))
   io_tauy = construct_io_field('TAUY', dim1=i_dim, dim2=j_dim,          &
                field_loc=field_loc_center, field_type=field_type_vector, &
                d2d_array=SMFT_TC(:,:,2,:))
   io_mask = construct_io_field('TCMASK', dim1=i_dim, dim2=j_dim,        &
                field_loc=field_loc_center, field_type=field_type_scalar, &
                d2d_array=TC_MASK)
   call data_set(forcing_file,'define',io_taux)
   call data_set(forcing_file,'define',io_tauy)
   call data_set(forcing_file,'define',io_mask)
   call data_set(forcing_file,'read'  ,io_taux)
   call data_set(forcing_file,'read'  ,io_tauy)
   call data_set(forcing_file,'read'  ,io_mask)
   call destroy_io_field(io_taux)
   call destroy_io_field(io_tauy)
   call destroy_io_field(io_mask)
   call data_set(forcing_file,'close')
   call destroy_file(forcing_file)

   if (my_task == master_task) then
      write(stdout,'(a,i6,a,a)') ' TC file read (model year', iyear, '): ', &
                                 trim(forcing_filename)
   endif

   !$OMP PARALLEL DO PRIVATE(iblock)
   do iblock=1,nblocks_clinic
      where (TC_MASK(:,:,iblock) > p5)
         WORK1(:,:,iblock) = SMFT_TC(:,:,1,iblock)
         WORK2(:,:,iblock) = SMFT_TC(:,:,2,iblock)
      end where
   end do
   !$OMP END PARALLEL DO

 end subroutine check_tc_ws
!EOC
!----- PaleoTC mod end
```

- [ ] **Step 4: Review the diff**

Run: `git diff -- sourcemods/`
Expected: only the `PaleoTC mod` blocks have changed. In `forcing_coupled.F90` that is the `use` line, the intent change, and the `call check_tc_ws`.

- [ ] **Step 5: Commit**

```bash
git add sourcemods/src.pop
git commit -m "Port TC wind-stress SourceMods to Eocene LR with replace semantics"
```

- [ ] **Step 6 (you do this): build and run a 5-day test**

Clone LR-cntl into LR-TC, copy `sourcemods/src.pop/*.F90` into `SourceMods/src.pop/`, and run `./case.build`. Expected: it compiles cleanly. Then run 5 days on dates that have TCs. Expected: `ocn.log` shows a `TC file read:` line on each day that has a 00Z file. With `tc_model_year0 = 1`, model year 1 reads track year 31, so the 5-day test can start at year 1.

---

## Self-review notes

- **Coverage against the spec:** task 2a (translate the NCL + parallelization) is Tasks 2 and 4; task 2b is Task 3 plus the notes; task 2c is the notes plus Task 4; organizing the output is the layout above plus Task 5; prescribing the stress in LR-TC is Task 7.
- **Still open for you:** `tc_model_year0` in `forcing_ws.F90` (1 for now). Set it to the LR-TC model year that should read track year 31. `lr_year_offset` in `config.yaml` stays 0, because the year mapping now lives in the SourceMod.
- **Storage:** POP reads only the 00Z files (about 14 GB of the 54 GB). Copy them off scratch to `/glade/work` or campaign storage before a long run, and change `tc_filename` to the new location.
- **Still open for you (daily coupling):** with `OCN_NCPL=1`, POP reads only the 00Z file. That one instantaneous snapshot replaces the *daily-mean* coupler stress for the whole day, and the 06/12/18Z files are never read. Li et al. 2023 had the same behavior. An alternative is to write the mean of the four 6-hourly TC stresses (and the union of their masks) into the 00Z file. Decide before the LR-TC run.
