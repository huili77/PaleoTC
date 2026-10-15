# PaleoTC TC Wind-Stress Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python package and Casper workflow that extracts tropical-cyclone (TC) winds from the Eocene HR simulation, converts them to surface wind stress on the LR ocean grid, and writes the per-timestep files that the LR-TC POP SourceMods read.

**Architecture:** A small `src/`-layout package `paleotc` runs three stages for each HR year: **extract** (TC-time snapshots of UBOT, VBOT, U10 and air density, regridded with the existing ESMF weights and masked to 20°×20° boxes) → **stress** (bulk formula from CESM `shr_flux_atmOcn`) → **export** (one POP-readable file per time step, named the way POP's `get_forcing_filename` expects). Each year is one dask task, so the stages chain without barriers. dask-jobqueue's `PBSCluster` on Casper runs one worker per year.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, xarray, netCDF4, cftime, pandas, PyYAML, dask.distributed and dask-jobqueue (PBS on Casper), matplotlib (QC only), pytest.

**Spec:** `docs/paleo_tc.docx`, together with the decisions from the 2026-10-15 design discussion recorded under *Design decisions* below.

## Design decisions (from the design discussion; verified against data and source)

| Topic | Decision | Evidence |
|---|---|---|
| HR input stream | `cam.h2` (instantaneous 6-hourly). Do **not** use `h4`, which is a time mean. | `h4` variables have `cell_methods = "time: mean"`; `h2` variables have no cell_methods. |
| Target grid | The LR Eocene ocean grid `gx1eocene` (320×384, gx1v6 layout). This is **not** the standard gx1v6 geometry: U-point latitudes differ by up to 0.5 rad. | Compared `grid.1.pop.da` with `horiz_grid_20010402.ieeer8`. |
| Weights | `/glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne120_g16/map_ne120np4_TO_gx1eocene_blin.260410.nc` (bilinear, the same method the coupler uses for atm→ocn state and vector fields). | Its `yc_b`/`xc_b`/`mask_b` match `domain.ocn.gx1eocene.260408.nc` (used by the LR case) to within 1e-5°. Note: the old `ne120_to_gx1v6_patch_weight.nc` is bilinear despite its name, and is on the wrong grid. |
| Ocean grid and mask | `/glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne30_g16/domain.ocn.gx1eocene.260408.nc` (`xc`, `yc`, `mask`). | This is the LR case's `OCN_DOMAIN_FILE`. |
| u10n | Use CAM `U10` as the neutral 10 m wind. `ustar = sqrt(cdn(U10))·U10`. | `atm_import_export.F90:90`: `cam_in%u10 = x2a(Sx_u10)`, which is `sqrt(duu10n)` from `shr_flux_atmOcn`. |
| cdn | `0.0027/U + 0.000142 + 0.0000764·U` only, with **no** high-wind branch. | iCESM1.3.1 `cime/src/share/util/shr_flux_mod.F90:183`. |
| Air density | `rho = PBOT / (Rd·TREFHT·(1+zvir·QREFHT))`, with `PBOT = hyam(nlev)·P0 + hybm(nlev)·PS`, computed on the ne120 grid and then regridded. | Monthly check: the PS-based proxy is +0.45%, the PBOT correction removes the pressure part, and a fixed 1.2 kg/m³ is +6.3% (about +12% in TC cores). |
| Direction | Stress points along (UBOT, VBOT), east/north. Surface currents are ignored (unbiased, ±6% scatter). | The coupler's `WORK1`/`WORK2` are east/north, and POP rotates them by `ANGLET`. |
| Box | Half-open box: `|Δlat|` and wrapped `|Δlon|` in [−10°, 10°), with Δ = point − storm center. The union of all storm boxes is taken, then intersected with the ocean mask. | Same as the Li et al. 2023 NCL, plus the 0°/360° longitude-wrap fix. |
| Coupler semantics | **Replace** the coupler stress inside the boxes (your decision). Exported files carry `TCMASK` so the SourceMod can do this. Task 12 ports the SourceMod. | The Li et al. 2023 SourceMod *adds* `SMFT_TC` (`forcing_ws.F90`, `WORK1 = WORK1 + SMFT_TC`). |
| File naming | `{prefix}.{YYYY}.{DDD}.{HH}.nc`, where YYYY is the **LR-TC model year** (HR year + `lr_year_offset`), DDD is the 1-based day of year in the noleap calendar, and HH is the hour. | POP `forcing_tools.F90:get_forcing_filename` (called with `thour00`). |
| Exported hours | All four (00, 06, 12, 18). With `OCN_NCPL=1` POP only reads the 00Z files, and a missing file is silently skipped, so writing all hours avoids silent gaps if the coupling ever changes. | LR case `env_run.xml`: `OCN_NCPL=1`. |
| Output root | `/glade/derecho/scratch/huili7/PaleoTC` (configurable). Scratch is purged after 180 days, so copy final exports to `/glade/work` or campaign storage. | — |

## Global Constraints

- Python ≥ 3.10. Install the conda env from `environment.yml` at `/glade/work/huili7/miniconda3/envs/paleotc`.
- Calendar is `noleap`. Represent times as `cftime.DatetimeNoLeap`, and match track times to model times with `(year, month, day, hour)` tuples.
- Reference code in `scripts_Li2023/` is read-only. Never edit it.
- Units: winds m/s, density kg/m³, stress N/m² (POP applies `momentum_factor` itself). The stress is the stress on the ocean, positive in the wind direction.
- Exported files: `NETCDF3_64BIT` format, dims `(nlat, nlon)` = (384, 320), variables `TAUX`, `TAUY`, `TCMASK` as float32, **no `_FillValue`**, no NaN or Inf.
- All output files are written to a temporary path first and then moved into place with `os.replace`, so they are never half-written.
- Account `P93300014`, queue `casper`.

## Review Focus

1. **Storms straddling 0°/360° longitude:** the box must wrap; a storm at 355°E must cover points at 0–5°E. Tested in Task 4 (`test_box_wraps_longitude`).
2. **Track times missing from the HR output** (for example the `0061-01-01 00` track point, or gaps): log them, count them, skip them, and don't crash. Tested in Task 6 (`test_extract_reports_missing_times`).
3. **Boxes over land:** TCMASK and stress must be 0 on land points, while `ntc` still counts the storm. Tested in Task 6 (`test_extract_masks_land`).
4. **Weak or zero wind** (`cdn` is singular at U=0): the outputs must contain no NaN or Inf. Tested in Task 5 (`test_wind_stress_calm_is_finite`) and Task 8 (`test_export_files_are_pop_ready`).
5. **Rerunning a year after a failed or partial run:** it must overwrite cleanly and leave no `*.tmp` files behind. Tested in Task 6 (`test_extract_rerun_overwrites`).

---

## Repo layout (final)

```
PaleoTC/
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
├── config/
│   └── eocene_03x.yaml            # all paths and parameters for this experiment
├── src/paleotc/
│   ├── __init__.py
│   ├── config.py                  # YAML → frozen dataclasses; HR file lookup; output paths
│   ├── tracks.py                  # TempestExtremes track parser; storm centers per time
│   ├── regrid.py                  # ESMF weight file → scipy.sparse regridder
│   ├── boxes.py                   # 20°×20° box mask on the curvilinear ocean grid (lon wrap)
│   ├── flux.py                    # cdn, air density, wind stress (shr_flux_atmOcn subset)
│   ├── extract.py                 # stage 1: HR h2 → boxed UBOT/VBOT/U10/RHO per year
│   ├── stress.py                  # stage 2: boxed winds → TAUX/TAUY per year
│   ├── export.py                  # stage 3: per-timestep POP files
│   ├── pipeline.py                # run_year(): chains the stages for one year
│   ├── cluster.py                 # PBSCluster (Casper) or LocalCluster
│   └── cli.py                     # `paleotc-run` entry point
├── scripts/
│   ├── submit_driver.pbs          # runs paleotc-run as a small Casper job
│   └── qc_year.py                 # stats and a snapshot plot for one processed year
├── sourcemods/src.pop/            # LR-TC POP SourceMods (Task 12)
│   ├── forcing_coupled.F90
│   └── forcing_ws.F90
├── tests/
│   ├── conftest.py                # synthetic mini case (3×4 grid) used by unit tests
│   ├── test_config.py  test_tracks.py  test_regrid.py  test_boxes.py
│   ├── test_flux.py    test_extract.py test_stress.py  test_export.py  test_cli.py
├── docs/
│   ├── paleo_tc.docx
│   ├── notes/get_tau_review.md    # answers to tasks 2b and 2c
│   └── superpowers/plans/…
└── scripts_Li2023/                # reference code (read-only)
```

Output layout under `output_root`:

```
boxed/tc_boxed.YYYY.nc       (time, nlat, nlon): UBOT VBOT U10 RHO TCMASK; ntc(time)
stress/tc_stress.YYYY.nc     (time, nlat, nlon): TAUX TAUY TCMASK; ntc(time)
pop_forcing/{prefix}.YYYY.DDD.HH.nc               ← what LR-TC reads
logs/run_<timestamp>.json    per-year summaries and failures
```

---

### Task 1: Branch, environment, scaffold, config

**Files:**
- Create: `environment.yml`, `pyproject.toml`, `.gitignore`, `config/eocene_03x.yaml`, `src/paleotc/__init__.py`, `src/paleotc/config.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path) -> Config`; `Config` with fields `case, hr_dir, track_file, map_file, ocn_domain_file, output_root, first_year, last_year, export: ExportConfig, cluster: ClusterConfig | None, hr_stream="h2", box_half_width=10.0, time_chunk=120`; property `years`; methods `hr_file(var, year) -> Path`, `boxed_path(year)`, `stress_path(year)`; property `export_dir`. `ExportConfig(prefix: str, lr_year_offset: int, hours: tuple[int, ...])`. `ClusterConfig(account, queue, cores, memory, walltime, n_workers, interface, local_directory, log_directory)`.
- Produces (tests): fixture `fake_case` (path to a YAML config for a synthetic case), plus constants `CASE, NLAT, NLON, NCOL, LATS, LONS, TIMES, RHO_EXPECTED_INPUTS`.

- [ ] **Step 1: Create a branch and the environment**

```bash
cd /glade/work/huili7/PaleoTC
git checkout -b feature/tc-stress-pipeline
```

Write `environment.yml`:

```yaml
name: paleotc
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - scipy
  - pandas
  - xarray
  - netcdf4
  - cftime
  - pyyaml
  - dask
  - distributed
  - dask-jobqueue
  - matplotlib
  - pytest
  - pip
```

```bash
module load conda
conda env create -f environment.yml -p /glade/work/huili7/miniconda3/envs/paleotc
conda activate /glade/work/huili7/miniconda3/envs/paleotc
```

- [ ] **Step 2: Write `pyproject.toml`, `.gitignore`, and the package init**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "paleotc"
version = "0.1.0"
description = "Extract TC wind stress from Eocene HR CESM output for LR-TC forcing"
requires-python = ">=3.10"
dependencies = []  # managed by environment.yml

[project.scripts]
paleotc-run = "paleotc.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["glade: needs NCAR GLADE data (skipped when unavailable)"]
```

`.gitignore`:

```
*~
__pycache__/
*.egg-info/
.pytest_cache/
dask-worker-space/
*.png
```

`src/paleotc/__init__.py`:

```python
"""Extract TC wind stress from Eocene HR CESM output for LR-TC forcing."""
```

```bash
pip install -e .
```

- [ ] **Step 3: Write the synthetic-case fixture (`tests/conftest.py`)**

The mini case uses an identity map from 12 "native" columns onto a 3×4 ocean grid, so the expected values can be read straight off the inputs.

```python
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
```

- [ ] **Step 4: Write the failing config test (`tests/test_config.py`)**

```python
import pytest

from paleotc.config import load_config


def test_load_config_types_and_paths(fake_case):
    cfg = load_config(fake_case)
    assert cfg.years == [31]
    assert cfg.export.hours == (0, 6, 12, 18)
    assert cfg.export.lr_year_offset == 70
    assert cfg.cluster is None
    assert cfg.boxed_path(31).name == "tc_boxed.0031.nc"
    assert cfg.stress_path(31).name == "tc_stress.0031.nc"
    assert cfg.export_dir.name == "pop_forcing"


def test_hr_file_found(fake_case):
    cfg = load_config(fake_case)
    assert cfg.hr_file("UBOT", 31).name == "fake.case.cam.h2.UBOT.0031010100-0032010100.nc"


def test_hr_file_missing_raises(fake_case):
    cfg = load_config(fake_case)
    with pytest.raises(FileNotFoundError, match="UBOT"):
        cfg.hr_file("UBOT", 45)
```

- [ ] **Step 5: Run it and confirm it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.config'`

- [ ] **Step 6: Implement `src/paleotc/config.py`**

```python
"""Experiment configuration: YAML file -> frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExportConfig:
    prefix: str
    lr_year_offset: int  # LR-TC model year = HR year + offset
    hours: tuple[int, ...] = (0, 6, 12, 18)


@dataclass(frozen=True)
class ClusterConfig:
    account: str
    queue: str = "casper"
    cores: int = 1
    memory: str = "16GB"
    walltime: str = "02:00:00"
    n_workers: int = 30
    interface: str | None = "ext"
    local_directory: str = "/tmp"
    log_directory: str = "dask-logs"


@dataclass(frozen=True)
class Config:
    case: str
    hr_dir: Path
    track_file: Path
    map_file: Path
    ocn_domain_file: Path
    output_root: Path
    first_year: int
    last_year: int
    export: ExportConfig
    cluster: ClusterConfig | None = None
    hr_stream: str = "h2"
    box_half_width: float = 10.0
    time_chunk: int = 120

    @property
    def years(self) -> list[int]:
        return list(range(self.first_year, self.last_year + 1))

    def hr_file(self, var: str, year: int) -> Path:
        pattern = f"{self.case}.cam.{self.hr_stream}.{var}.{year:04d}01*.nc"
        matches = sorted(self.hr_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected 1 file matching {pattern} in {self.hr_dir}, found {len(matches)}"
            )
        return matches[0]

    def boxed_path(self, year: int) -> Path:
        return self.output_root / "boxed" / f"tc_boxed.{year:04d}.nc"

    def stress_path(self, year: int) -> Path:
        return self.output_root / "stress" / f"tc_stress.{year:04d}.nc"

    @property
    def export_dir(self) -> Path:
        return self.output_root / "pop_forcing"


_PATH_KEYS = ("hr_dir", "track_file", "map_file", "ocn_domain_file", "output_root")


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    export = raw.pop("export")
    export["hours"] = tuple(export.get("hours", (0, 6, 12, 18)))
    cluster = raw.pop("cluster", None)
    for key in _PATH_KEYS:
        raw[key] = Path(raw[key])
    return Config(
        export=ExportConfig(**export),
        cluster=ClusterConfig(**cluster) if cluster else None,
        **raw,
    )
```

- [ ] **Step 7: Run it and confirm it passes**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 8: Write the experiment config `config/eocene_03x.yaml`**

```yaml
# Eocene 3x CO2: HR (ne120_t12) TC winds -> LR (ne30_g16, gx1eocene) wind stress
case: b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002
hr_dir: /glade/campaign/cesm/community/palwg/ASD_paleoWeather/b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002/atm/proc/tseries/hour_6
hr_stream: h2            # instantaneous 6-hourly; h4 is time-mean
track_file: /glade/campaign/cgd/ppc/brady/paleoweather/TCs/Eo3x/F3/tracks/TC_Eo3x.002_F3tracks_yrs31-60.txt
first_year: 31
last_year: 60

map_file: /glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne120_g16/map_ne120np4_TO_gx1eocene_blin.260410.nc
ocn_domain_file: /glade/work/jiangzhu/data/inputdata/cesm13_eocene/ne30_g16/domain.ocn.gx1eocene.260408.nc
box_half_width: 10.0     # degrees; 20x20 box
time_chunk: 120          # time steps read per chunk (~2.2 GB of native fields)

output_root: /glade/derecho/scratch/huili7/PaleoTC

export:
  prefix: b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002.tctau
  lr_year_offset: 0      # CONFIRM: LR-TC model year = HR year + offset
  hours: [0, 6, 12, 18]

cluster:
  account: P93300014
  queue: casper
  cores: 1
  memory: 16GB
  walltime: "02:00:00"
  n_workers: 30
  interface: ext
  local_directory: /glade/derecho/scratch/huili7/dask-tmp
  log_directory: /glade/derecho/scratch/huili7/PaleoTC/dask-logs
```

- [ ] **Step 9: Commit**

```bash
git add environment.yml pyproject.toml .gitignore config/ src/paleotc/__init__.py src/paleotc/config.py tests/conftest.py tests/test_config.py docs/paleo_tc.docx docs/superpowers scripts_Li2023/*.ncl scripts_Li2023/*.sh scripts_Li2023/*.ipynb scripts_Li2023/*.F90
git commit -m "Scaffold paleotc package, environment and experiment config"
```

---

### Task 2: Track parsing

**Files:**
- Create: `src/paleotc/tracks.py`, `tests/test_tracks.py`

**Interfaces:**
- Produces: `read_tracks(path) -> pandas.DataFrame` with columns `track_id, lon, lat, psl, wind, year, month, day, hour`. Longitudes are in [0, 360).
- Produces: `centers_by_time(tracks, year) -> dict[tuple[int, int, int, int], np.ndarray]`, which maps `(year, month, day, hour)` to an `(n_storms, 2)` array of `(lon, lat)`.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import numpy as np
import pytest

from conftest import TRACKS
from paleotc.tracks import centers_by_time, read_tracks

REAL_TRACKS = Path(
    "/glade/campaign/cgd/ppc/brady/paleoweather/TCs/Eo3x/F3/tracks/TC_Eo3x.002_F3tracks_yrs31-60.txt"
)


def test_read_tracks_parses_points(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    df = read_tracks(f)
    assert len(df) == 5
    assert df.track_id.tolist() == [0, 0, 1, 2, 3]
    assert df.iloc[0][["lon", "lat", "psl", "wind"]].tolist() == [0.0, 0.0, 9.9e4, 30.0]
    assert df.iloc[1][["year", "month", "day", "hour"]].tolist() == [31, 1, 1, 6]


def test_read_tracks_wraps_negative_longitude(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    assert read_tracks(f).iloc[2].lon == pytest.approx(355.0)


def test_centers_by_time_groups_storms(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text(TRACKS)
    centers = centers_by_time(read_tracks(f), 31)
    assert set(centers) == {(31, 1, 1, 0), (31, 1, 1, 6), (31, 12, 31, 18)}
    np.testing.assert_allclose(
        sorted(map(tuple, centers[(31, 1, 1, 6)])), [(2.0, 0.5), (355.0, 15.0)]
    )
    assert centers_by_time(read_tracks(f), 32) == {}


@pytest.mark.glade
@pytest.mark.skipif(not REAL_TRACKS.exists(), reason="GLADE track file not available")
def test_real_track_file():
    df = read_tracks(REAL_TRACKS)
    assert df.track_id.nunique() == 2987
    assert set(df.hour.unique()) <= {0, 6, 12, 18}
    assert df.year.min() == 31 and df.year.max() == 61
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_tracks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.tracks'`

- [ ] **Step 3: Implement `src/paleotc/tracks.py`**

```python
"""Read TempestExtremes trajectory files.

Format: a header line ``start <npts> <yyyy> <mm> <dd> <hh>`` per track, followed by
one line per point: ``<i> <lon> <lat> <psl> <wind> ... <yyyy> <mm> <dd> <hh>``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["track_id", "lon", "lat", "psl", "wind", "year", "month", "day", "hour"]

TimeKey = tuple[int, int, int, int]


def read_tracks(path: str | Path) -> pd.DataFrame:
    rows = []
    track_id = -1
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "start":
                track_id += 1
                continue
            year, month, day, hour = (int(v) for v in parts[-4:])
            rows.append(
                (track_id, float(parts[1]) % 360.0, float(parts[2]),
                 float(parts[3]), float(parts[4]), year, month, day, hour)
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def centers_by_time(tracks: pd.DataFrame, year: int) -> dict[TimeKey, np.ndarray]:
    sub = tracks[tracks.year == year]
    return {
        tuple(int(v) for v in key): group[["lon", "lat"]].to_numpy()
        for key, group in sub.groupby(["year", "month", "day", "hour"])
    }
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_tracks.py -v`
Expected: 4 passed on Casper (the GLADE test runs there; elsewhere it is skipped and 3 pass).

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/tracks.py tests/test_tracks.py
git commit -m "Add TempestExtremes track parser"
```

---

### Task 3: Regridder

**Files:**
- Create: `src/paleotc/regrid.py`, `tests/test_regrid.py`

**Interfaces:**
- Produces: `class Regridder(map_file)` with attributes `n_a: int`, `shape: tuple[int, int]` (`(nlat, nlon)`), and `matrix` (scipy CSR, `n_b × n_a`). Calling it as `__call__(data: ndarray[..., n_a]) -> ndarray[..., nlat, nlon]` returns float64; destination points with no weights get 0.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_regrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.regrid'`

- [ ] **Step 3: Implement `src/paleotc/regrid.py`**

```python
"""Apply ESMF offline regridding weights as a sparse matrix."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import xarray as xr


class Regridder:
    def __init__(self, map_file: str | Path):
        with xr.open_dataset(map_file) as ds:
            n_a, n_b = ds.sizes["n_a"], ds.sizes["n_b"]
            nlon, nlat = (int(v) for v in ds["dst_grid_dims"].values)  # ESMF order: (ni, nj)
            self.matrix = sp.csr_matrix(
                (ds["S"].values, (ds["row"].values - 1, ds["col"].values - 1)),
                shape=(n_b, n_a),
            )
        self.n_a = n_a
        self.shape = (nlat, nlon)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.shape[-1] != self.n_a:
            raise ValueError(f"last dim is {data.shape[-1]}, map expects n_a={self.n_a}")
        lead = data.shape[:-1]
        flat = data.reshape(-1, self.n_a).astype("f8", copy=False)
        out = (self.matrix @ flat.T).T
        return out.reshape(*lead, *self.shape)
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_regrid.py -v`
Expected: 4 passed on Casper

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/regrid.py tests/test_regrid.py
git commit -m "Add sparse-matrix regridder for ESMF weight files"
```

---

### Task 4: TC box mask

**Files:**
- Create: `src/paleotc/boxes.py`, `tests/test_boxes.py`

**Interfaces:**
- Produces: `box_mask(tlat: ndarray[nlat, nlon], tlon: ndarray[nlat, nlon], centers: ndarray[n, 2] of (lon, lat), half_width: float) -> ndarray[bool, (nlat, nlon)]`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_boxes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.boxes'`

- [ ] **Step 3: Implement `src/paleotc/boxes.py`**

```python
"""Box masks around TC centers on a curvilinear grid."""
from __future__ import annotations

import numpy as np


def box_mask(tlat, tlon, centers, half_width):
    """True where a grid point lies in [-hw, hw) of any center in both lat and wrapped lon."""
    mask = np.zeros(tlat.shape, dtype=bool)
    for lon, lat in centers:
        dlon = (tlon - lon + 180.0) % 360.0 - 180.0
        dlat = tlat - lat
        mask |= (dlon >= -half_width) & (dlon < half_width) & (dlat >= -half_width) & (dlat < half_width)
    return mask
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_boxes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/boxes.py tests/test_boxes.py
git commit -m "Add TC box mask with longitude wrap"
```

---

### Task 5: Flux formulas

**Files:**
- Create: `src/paleotc/flux.py`, `tests/test_flux.py`

**Interfaces:**
- Produces: constants `RDAIR`, `ZVIR`, `UMIN`; `cdn(u)`; `air_density(ps, trefht, qrefht, hyam_bot, hybm_bot, p0)`; `wind_stress(ubot, vbot, u10n, rho) -> (taux, tauy)`. All work on numpy arrays.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from paleotc.flux import RDAIR, ZVIR, air_density, cdn, wind_stress


def test_constants_match_shr_const():
    assert RDAIR == pytest.approx(287.04, abs=0.01)
    assert ZVIR == pytest.approx(0.6078, abs=1e-4)


def test_cdn_low_wind_formula_only():
    assert cdn(10.0) == pytest.approx(0.0027 / 10 + 0.000142 + 0.0000764 * 10)
    assert cdn(40.0) == pytest.approx(0.0027 / 40 + 0.000142 + 0.0000764 * 40)  # no high-wind cap


def test_air_density_uses_pbot():
    rho = air_density(1.0e5, 300.0, 0.02, 0.0, 0.99, 1.0e5)
    assert rho == pytest.approx(0.99e5 / (RDAIR * 300.0 * (1 + ZVIR * 0.02)))


def test_wind_stress_magnitude_and_direction():
    taux, tauy = wind_stress(np.array(30.0), np.array(40.0), np.array(40.0), np.array(1.1))
    tau = 1.1 * cdn(40.0) * 40.0**2
    assert taux == pytest.approx(0.6 * tau)
    assert tauy == pytest.approx(0.8 * tau)


def test_wind_stress_calm_is_finite():
    z = np.zeros(3)
    taux, tauy = wind_stress(z, z, z, np.full(3, 1.1))
    assert np.isfinite(taux).all() and np.isfinite(tauy).all()
    np.testing.assert_array_equal(taux, 0.0)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_flux.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.flux'`

- [ ] **Step 3: Implement `src/paleotc/flux.py`**

```python
"""Wind stress following CESM shr_flux_atmOcn (iCESM1.3.1).

Source: cime/src/share/util/shr_flux_mod.F90 and shr_const_mod.F90.
ustar is taken from the neutral 10 m wind, ustar = sqrt(cdn(u10n)) * u10n, where
u10n is CAM's U10 (= sqrt(duu10n) computed by the same routine in the HR run).
"""
from __future__ import annotations

import numpy as np

RGAS = 6.02214e26 * 1.38065e-23  # SHR_CONST_AVOGAD * SHR_CONST_BOLTZ
RDAIR = RGAS / 28.966            # SHR_CONST_RDAIR
RWV = RGAS / 18.016              # SHR_CONST_RWV
ZVIR = RWV / RDAIR - 1.0         # SHR_CONST_ZVIR
UMIN = 0.5                       # shr_flux_atmOcn minimum wind speed (m/s)


def cdn(u):
    """Neutral 10 m drag coefficient. iCESM1.3.1 has no high-wind branch."""
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def air_density(ps, trefht, qrefht, hyam_bot, hybm_bot, p0):
    """Lowest-level density: exact PBOT from hybrid coefficients, 2 m T and q as proxies."""
    pbot = hyam_bot * p0 + hybm_bot * ps
    return pbot / (RDAIR * trefht * (1.0 + ZVIR * qrefht))


def wind_stress(ubot, vbot, u10n, rho):
    """Eastward/northward stress on the ocean (N/m^2), positive in the wind direction."""
    u10n = np.maximum(u10n, UMIN)
    tau = rho * cdn(u10n) * u10n**2
    vmag = np.maximum(np.hypot(ubot, vbot), UMIN)
    return tau * ubot / vmag, tau * vbot / vmag
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_flux.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/flux.py tests/test_flux.py
git commit -m "Add wind-stress formulas from shr_flux_atmOcn"
```

---

### Task 6: Extraction stage

**Files:**
- Create: `src/paleotc/extract.py`, `tests/test_extract.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `read_tracks`/`centers_by_time` (Task 2), `Regridder` (Task 3), `box_mask` (Task 4), `air_density` (Task 5).
- Produces: `extract_year(cfg, year, regridder=None) -> dict` with keys `year, n_times, n_missing, path`. It writes `cfg.boxed_path(year)` containing `UBOT, VBOT, U10, RHO` (float32), `TCMASK` (int8) on `(time, nlat, nlon)`, `ntc(time)`, and coordinates `time, TLAT, TLONG`.
- Produces: `load_ocean_grid(domain_file) -> (tlat, tlon, ocean_mask_bool)`; `write_atomic(ds, path, encoding)`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.extract'`

- [ ] **Step 3: Implement `src/paleotc/extract.py`**

```python
"""Stage 1: HR 6-hourly fields at TC times -> boxed fields on the LR ocean grid."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr

from .boxes import box_mask
from .config import Config
from .flux import air_density
from .regrid import Regridder
from .tracks import centers_by_time, read_tracks

log = logging.getLogger(__name__)

WIND_VARS = ("UBOT", "VBOT", "U10")
RHO_VARS = ("PS", "TREFHT", "QREFHT")
TIME_ENCODING = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}
UNITS = {"UBOT": "m/s", "VBOT": "m/s", "U10": "m/s", "RHO": "kg/m3"}


def load_ocean_grid(domain_file):
    with xr.open_dataset(domain_file) as ds:
        return ds["yc"].values, ds["xc"].values % 360.0, ds["mask"].values.astype(bool)


def write_atomic(ds: xr.Dataset, path: Path, encoding: dict, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    ds.to_netcdf(tmp, encoding=encoding, **kwargs)
    os.replace(tmp, path)


def extract_year(cfg: Config, year: int, regridder: Regridder | None = None) -> dict:
    regridder = regridder or Regridder(cfg.map_file)
    tlat, tlon, ocean = load_ocean_grid(cfg.ocn_domain_file)
    centers = centers_by_time(read_tracks(cfg.track_file), year)

    files = {v: xr.open_dataset(cfg.hr_file(v, year)) for v in WIND_VARS + RHO_VARS}
    try:
        ref = files["PS"]
        nlev = ref.sizes["lev"]
        hyam_bot = float(ref["hyam"][nlev - 1])
        hybm_bot = float(ref["hybm"][nlev - 1])
        p0 = float(ref["P0"])
        file_times = ref["time"].values
        for v, ds in files.items():
            if ds.sizes["time"] != file_times.size:
                raise ValueError(f"{v} has {ds.sizes['time']} times, PS has {file_times.size}")

        keys = [(t.year, t.month, t.day, t.hour) for t in file_times]
        tidx = [i for i, k in enumerate(keys) if k in centers]
        missing = sorted(set(centers) - set(keys))
        for k in missing:
            log.warning("year %d: track time %s not in HR output; skipped", year, k)

        nt = len(tidx)
        mask = np.zeros((nt, *regridder.shape), dtype="i1")
        ntc = np.zeros(nt, dtype="i4")
        for k, i in enumerate(tidx):
            c = centers[keys[i]]
            mask[k] = box_mask(tlat, tlon, c, cfg.box_half_width) & ocean
            ntc[k] = len(c)

        out = {v: np.zeros((nt, *regridder.shape), dtype="f4") for v in (*WIND_VARS, "RHO")}
        for start in range(0, nt, cfg.time_chunk):
            sel = tidx[start : start + cfg.time_chunk]
            native = {v: files[v][v].isel(time=sel).values for v in WIND_VARS + RHO_VARS}
            native["RHO"] = air_density(
                native["PS"], native["TREFHT"], native["QREFHT"], hyam_bot, hybm_bot, p0
            )
            inside = mask[start : start + len(sel)].astype(bool)
            for v in out:
                out[v][start : start + len(sel)] = np.where(inside, regridder(native[v]), 0.0)
            log.info("year %d: %d/%d TC times done", year, start + len(sel), nt)
        times = file_times[tidx]
    finally:
        for ds in files.values():
            ds.close()

    dims = ("time", "nlat", "nlon")
    ds_out = xr.Dataset(
        {v: (dims, out[v], {"units": UNITS[v]}) for v in out}
        | {
            "TCMASK": (dims, mask, {"long_name": "1 inside TC boxes over ocean"}),
            "ntc": ("time", ntc, {"long_name": "number of tracked TCs at this time"}),
        },
        coords={"time": times, "TLAT": (("nlat", "nlon"), tlat), "TLONG": (("nlat", "nlon"), tlon)},
        attrs={
            "source_case": cfg.case,
            "track_file": str(cfg.track_file),
            "map_file": str(cfg.map_file),
            "box_half_width_deg": cfg.box_half_width,
        },
    )
    encoding = {v: {"zlib": True, "complevel": 1} for v in (*out, "TCMASK")}
    encoding["time"] = TIME_ENCODING
    path = cfg.boxed_path(year)
    write_atomic(ds_out, path, encoding)
    return {"year": year, "n_times": nt, "n_missing": len(missing), "path": str(path)}
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_extract.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/extract.py tests/test_extract.py
git commit -m "Add extraction stage: TC-time boxed fields on LR ocean grid"
```

---

### Task 7: Stress stage

**Files:**
- Create: `src/paleotc/stress.py`, `tests/test_stress.py`

**Interfaces:**
- Consumes: `cfg.boxed_path(year)` written by `extract_year` (Task 6); `wind_stress` (Task 5); `write_atomic`, `TIME_ENCODING` (Task 6).
- Produces: `stress_year(cfg, year) -> dict` with keys `year, n_times, max_tau, path`. It writes `cfg.stress_path(year)` containing `TAUX, TAUY` (float32, N/m², east/north), `TCMASK`, `ntc`, and the same coordinates.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_stress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.stress'`

- [ ] **Step 3: Implement `src/paleotc/stress.py`**

```python
"""Stage 2: boxed winds and density -> TC wind stress on the LR ocean grid."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .config import Config
from .extract import TIME_ENCODING, write_atomic
from .flux import wind_stress


def stress_year(cfg: Config, year: int) -> dict:
    with xr.open_dataset(cfg.boxed_path(year)) as ds:
        ds = ds.load()
    taux, tauy = wind_stress(ds.UBOT.values, ds.VBOT.values, ds.U10.values, ds.RHO.values)
    inside = ds.TCMASK.values.astype(bool)
    taux = np.where(inside, taux, 0.0).astype("f4")
    tauy = np.where(inside, tauy, 0.0).astype("f4")

    dims = ("time", "nlat", "nlon")
    attrs = {"units": "N/m2", "comment": "stress on the ocean, positive in the wind direction"}
    out = xr.Dataset(
        {
            "TAUX": (dims, taux, attrs | {"long_name": "TC wind stress, eastward"}),
            "TAUY": (dims, tauy, attrs | {"long_name": "TC wind stress, northward"}),
            "TCMASK": ds.TCMASK,
            "ntc": ds.ntc,
        },
        coords=ds.coords,
        attrs=ds.attrs
        | {"method": "rho*cdn(U10)*U10^2 along (UBOT,VBOT); cdn from iCESM1.3.1 shr_flux_mod"},
    )
    encoding = {v: {"zlib": True, "complevel": 1} for v in ("TAUX", "TAUY", "TCMASK")}
    encoding["time"] = TIME_ENCODING
    path = cfg.stress_path(year)
    write_atomic(out, path, encoding)
    max_tau = float(np.hypot(taux, tauy).max()) if taux.size else 0.0
    return {"year": year, "n_times": int(ds.sizes["time"]), "max_tau": max_tau, "path": str(path)}
```

- [ ] **Step 4: Run it and confirm it passes**

Run: `pytest tests/test_stress.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/stress.py tests/test_stress.py
git commit -m "Add stress stage"
```

---

### Task 8: POP export stage

**Files:**
- Create: `src/paleotc/export.py`, `tests/test_export.py`

**Interfaces:**
- Consumes: `cfg.stress_path(year)` (Task 7); `write_atomic`, `TIME_ENCODING` (Task 6).
- Produces: `pop_filename(prefix, model_year, day_of_year, hour) -> str`; `export_year(cfg, year) -> dict` with keys `year, n_files`. Files are written to `cfg.export_dir`.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_export.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.export'`

- [ ] **Step 3: Implement `src/paleotc/export.py`**

```python
"""Stage 3: one file per TC time step, in the form check_tc_ws (POP SourceMod) reads."""
from __future__ import annotations

import xarray as xr

from .config import Config
from .extract import TIME_ENCODING, write_atomic

EXPORT_VARS = ("TAUX", "TAUY", "TCMASK")


def pop_filename(prefix: str, model_year: int, day_of_year: int, hour: int) -> str:
    """Mirror POP forcing_tools.F90:get_forcing_filename (root.YYYY.DDD.HH) plus '.nc'."""
    return f"{prefix}.{model_year:04d}.{day_of_year:03d}.{hour:02d}.nc"


def export_year(cfg: Config, year: int) -> dict:
    with xr.open_dataset(cfg.stress_path(year)) as ds:
        ds = ds.load()
    n_files = 0
    for k, t in enumerate(ds.time.values):
        if t.hour not in cfg.export.hours:
            continue
        snap = xr.Dataset(
            {v: (("nlat", "nlon"), ds[v].values[k].astype("f4"), ds[v].attrs) for v in EXPORT_VARS},
            coords={"time": t},
            attrs={"hr_time": t.isoformat(), "lr_year_offset": cfg.export.lr_year_offset} | ds.attrs,
        )
        name = pop_filename(cfg.export.prefix, t.year + cfg.export.lr_year_offset, t.dayofyr, t.hour)
        encoding = {v: {"_FillValue": None} for v in EXPORT_VARS} | {"time": TIME_ENCODING}
        write_atomic(snap, cfg.export_dir / name, encoding, format="NETCDF3_64BIT")
        n_files += 1
    return {"year": year, "n_files": n_files}
```

- [ ] **Step 4: Run them and confirm they pass**

Run: `pytest tests/test_export.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/paleotc/export.py tests/test_export.py
git commit -m "Add POP export stage"
```

---

### Task 9: Pipeline, Casper cluster, CLI

**Files:**
- Create: `src/paleotc/pipeline.py`, `src/paleotc/cluster.py`, `src/paleotc/cli.py`, `scripts/submit_driver.pbs`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `extract_year`, `stress_year`, `export_year`, `load_config`.
- Produces: `STAGES = ("extract", "stress", "export")`; `run_year(config_path: str, year: int, stages: tuple[str, ...]) -> dict`; `make_client(cfg, local_workers: int | None = None, n_jobs: int | None = None) -> distributed.Client`; `cli.main(argv=None) -> int` (installed as `paleotc-run`).

- [ ] **Step 1: Write the failing test**

```python
import json

from paleotc.cli import main
from paleotc.config import load_config


def test_cli_runs_all_stages_locally(fake_case):
    assert main([str(fake_case), "--local", "1"]) == 0
    cfg = load_config(fake_case)
    assert len(list(cfg.export_dir.glob("*.nc"))) == 2
    (log,) = (cfg.output_root / "logs").glob("run_*.json")
    report = json.loads(log.read_text())
    assert report["failures"] == [] and report["results"][0]["year"] == 31


def test_cli_reports_failure(fake_case):
    # year 45 has no HR files -> FileNotFoundError captured, exit code 1
    assert main([str(fake_case), "--local", "1", "--years", "45", "45"]) == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paleotc.cli'`

- [ ] **Step 3: Implement `src/paleotc/pipeline.py`**

```python
"""Run the stages for one HR year (one dask task)."""
from __future__ import annotations

import logging

from .config import load_config
from .export import export_year
from .extract import extract_year
from .stress import stress_year

STAGES = ("extract", "stress", "export")
_RUNNERS = {"extract": extract_year, "stress": stress_year, "export": export_year}


def run_year(config_path: str, year: int, stages: tuple[str, ...] = STAGES) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(config_path)
    result = {"year": year}
    for stage in STAGES:
        if stage in stages:
            result[stage] = _RUNNERS[stage](cfg, year)
    return result
```

- [ ] **Step 4: Implement `src/paleotc/cluster.py`**

```python
"""Dask cluster: PBS jobs on Casper, or a LocalCluster for tests and debugging."""
from __future__ import annotations

from distributed import Client, LocalCluster

from .config import Config


def make_client(cfg: Config, local_workers: int | None = None, n_jobs: int | None = None) -> Client:
    if local_workers:
        return Client(LocalCluster(n_workers=local_workers, threads_per_worker=1))
    from dask_jobqueue import PBSCluster

    c = cfg.cluster
    if c is None:
        raise ValueError("config has no 'cluster' section; use --local N")
    cluster = PBSCluster(
        cores=c.cores,
        processes=1,
        memory=c.memory,
        account=c.account,
        queue=c.queue,
        walltime=c.walltime,
        resource_spec=f"select=1:ncpus={c.cores}:mem={c.memory}",
        interface=c.interface,
        local_directory=c.local_directory,
        log_directory=c.log_directory,
    )
    cluster.scale(jobs=min(c.n_workers, n_jobs or c.n_workers))
    return Client(cluster)
```

- [ ] **Step 5: Implement `src/paleotc/cli.py`**

```python
"""paleotc-run: run pipeline stages for a range of HR years, one dask task per year."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from distributed import as_completed

from .cluster import make_client
from .config import load_config
from .pipeline import STAGES, run_year


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"))
    p.add_argument("--local", type=int, metavar="N", help="use a LocalCluster with N workers")
    a = p.parse_args(argv)

    config_path = str(Path(a.config).resolve())
    cfg = load_config(config_path)
    years = list(range(a.years[0], a.years[1] + 1)) if a.years else cfg.years

    client = make_client(cfg, a.local, n_jobs=len(years))
    try:
        futures = {
            client.submit(run_year, config_path, y, tuple(a.stages), key=f"year-{y}", retries=1): y
            for y in years
        }
        results, failures = [], []
        for fut in as_completed(futures):
            year = futures[fut]
            try:
                results.append(fut.result())
                print(f"year {year}: done", flush=True)
            except Exception as exc:  # report every failed year, keep going
                failures.append({"year": year, "error": repr(exc)})
                print(f"year {year}: FAILED {exc!r}", flush=True)
    finally:
        client.close()

    log_dir = cfg.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stages": a.stages,
        "results": sorted(results, key=lambda r: r["year"]),
        "failures": sorted(failures, key=lambda r: r["year"]),
    }
    (log_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write `scripts/submit_driver.pbs`**

```bash
#!/bin/bash
#PBS -N paleotc-driver
#PBS -A P93300014
#PBS -q casper
#PBS -l select=1:ncpus=1:mem=8GB
#PBS -l walltime=06:00:00
#PBS -j oe
# Runs the dask scheduler/driver; workers are separate PBS jobs started by PBSCluster.
# Usage: qsub -v PALEOTC_ARGS="--years 31 31 --stages extract stress" scripts/submit_driver.pbs
set -eo pipefail  # no -u: conda activate scripts reference unset variables
module load conda
conda activate /glade/work/huili7/miniconda3/envs/paleotc
cd "$PBS_O_WORKDIR"
paleotc-run config/eocene_03x.yaml ${PALEOTC_ARGS:-}
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 8: Run the full suite**

Run: `pytest -v`
Expected: all pass (GLADE-marked tests pass on Casper)

- [ ] **Step 9: Commit**

```bash
git add src/paleotc/pipeline.py src/paleotc/cluster.py src/paleotc/cli.py scripts/submit_driver.pbs tests/test_cli.py
git commit -m "Add per-year pipeline, Casper PBSCluster and paleotc-run CLI"
```

---

### Task 10: Real-data run for one year, plus QC

**Files:**
- Create: `scripts/qc_year.py`

**Interfaces:**
- Consumes: `load_config`, `read_tracks`, and the stress files from Task 7.

- [ ] **Step 1: Write `scripts/qc_year.py`**

```python
"""Quick-look QC for one processed HR year: summary stats and the strongest snapshot."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from paleotc.config import load_config
from paleotc.tracks import read_tracks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("year", type=int)
    p.add_argument("--out", default=None, help="PNG path (default: <output_root>/logs/qc_YYYY.png)")
    a = p.parse_args()
    cfg = load_config(a.config)

    s = xr.open_dataset(cfg.stress_path(a.year))
    tau = np.hypot(s.TAUX.values, s.TAUY.values)
    inside = s.TCMASK.values == 1
    print(f"TC time steps: {s.sizes['time']}")
    print(f"storms per step: mean {float(s.ntc.mean()):.1f}, max {int(s.ntc.max())}")
    print(f"|tau| inside boxes: median {np.median(tau[inside]):.3f}, "
          f"p99.9 {np.percentile(tau[inside], 99.9):.2f}, max {tau.max():.2f} N/m2")
    print(f"non-finite values: {int((~np.isfinite(tau)).sum())}")

    k = int(tau.reshape(tau.shape[0], -1).max(axis=1).argmax())
    t = s.time.values[k]
    m = inside[k]
    tr = read_tracks(cfg.track_file)
    now = tr[(tr.year == t.year) & (tr.month == t.month) & (tr.day == t.day) & (tr.hour == t.hour)]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sc = ax.scatter(s.TLONG.values[m], s.TLAT.values[m], c=tau[k][m], s=3, cmap="viridis")
    ax.plot(now.lon, now.lat, "rx", ms=8, label="track centers")
    fig.colorbar(sc, ax=ax, label="|tau| (N/m²)")
    ax.set(xlim=(0, 360), ylim=(-60, 60), xlabel="longitude", ylabel="latitude",
           title=f"TC wind stress, strongest snapshot of HR year {a.year}: {t}")
    ax.legend(loc="lower left")
    out = a.out or cfg.output_root / "logs" / f"qc_{a.year:04d}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run year 31 on Casper with one PBS worker**

```bash
qsub -v PALEOTC_ARGS="--years 31 31" scripts/submit_driver.pbs
```

Expected: the job finishes with `year 31: done`. The newest `/glade/derecho/scratch/huili7/PaleoTC/logs/run_*.json` shows `failures: []`, `extract.n_missing: 0`, `extract.n_times` between 500 and 1460, and `export.n_files == extract.n_times`.

- [ ] **Step 3: Run the QC script and inspect the output**

Run: `python scripts/qc_year.py config/eocene_03x.yaml 31`
Expected: `non-finite values: 0`; max |tau| between about 2 and 15 N/m² (U10 = 50 m/s gives about 11 N/m²); and in the PNG, filled TC boxes centered on the red crosses, with none on land.

- [ ] **Step 4: Commit**

```bash
git add scripts/qc_year.py
git commit -m "Add QC script for processed years"
```

- [ ] **Step 5: Run all 30 years**

```bash
qsub scripts/submit_driver.pbs
```

Expected: the run JSON lists 30 results and no failures. Copy `pop_forcing/` to `/glade/work` or campaign storage (scratch is purged after 180 days).

---

### Task 11: Review notes for `get_tau_dask.ipynb` (tasks 2b and 2c) and README

**Files:**
- Create: `docs/notes/get_tau_review.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/notes/get_tau_review.md`**

```markdown
# Review of scripts_Li2023/get_tau_dask.ipynb

Reference: iCESM1.3.1 `cime/src/share/util/shr_flux_mod.F90` (`shr_flux_atmOcn`) and
`shr_const_mod.F90`. New implementation: `src/paleotc/flux.py`.

## (b) Does it reproduce the model's wind stress?

| # | Notebook | Model (shr_flux_atmOcn) | Effect | New code |
|---|---|---|---|---|
| 1 | `cdn1`: `if umps.any() > 12.5` compares a bool to 12.5, so it is always False and the low-wind formula is always used | `cdn = 0.0027/U + 0.000142 + 0.0000764·U` only, with no high-wind branch | Correct by accident; the high-wind branch must not be "fixed" in | `flux.cdn`, low-wind formula only |
| 2 | `ustar = sqrt(cdn(U10))·U10` | `ustar = rd·vmag` after a two-pass stability iteration from UBOT at zbot | CAM U10 is `Sx_u10 = sqrt(duu10n)` from this routine (`atm_import_export.F90:90`), so the shortcut is consistent. Measured offline: within 0.9% (slightly high) for 10–70 m/s, up to 7% only below 10 m/s, where the stress is small | Kept |
| 3 | `rho_air = 1.2` | `rbot` from PBOT, TBOT, QBOT | Fixed 1.2 is +6% over tropical ocean (Eocene air is warm and moist) and about +12% in 950 hPa TC cores | `PBOT/(Rd·TREFHT·(1+zvir·QREFHT))` with exact PBOT; about −0.2% |
| 4 | Direction from (UBOT, VBOT); currents ignored | Direction from (UBOT−us, VBOT−vs) | Unbiased; ±6% scatter for 0.5 m/s currents | Kept, documented |
| 5 | Attribute says "grid-x direction" | Coupler stress is east/north; POP rotates it by ANGLET (`forcing_coupled.F90`) | Values are right but the label is wrong | Labelled eastward/northward |
| 6 | Units N/m² | POP multiplies by `momentum_factor` | Correct | Same |
| 7 | `remap` attribute says "Higher-order Patch" | The weight file says `map_method = Bilinear remapping` | Misleading metadata | Records `map_file` |
| 8 | `alz`, `qsat`, `psimhu`, `SHR_CONST_G` defined but unused | — | Dead code | Removed |

Grid note for the Eocene runs: the LR ocean grid `gx1eocene` has the gx1v6 dimensions but a
different geometry, so the old `ne120_to_gx1v6` weights cannot be reused. The pipeline uses
`map_ne120np4_TO_gx1eocene_blin.260410.nc`.

## (c) Parallelization and robustness

1. **Barriers.** Batches of 20 `dask.delayed` tasks with a `dask.compute` per batch leave
   workers idle while the slowest file in each batch finishes. Now there is one task per
   year (`client.submit`), with results collected through `as_completed`, so there are no barriers.
2. **Task granularity.** One task per ~1 MB file means scheduler overhead and file-open
   latency on GLADE dominate. Now each task is vectorized numpy over (time, nlat, nlon) for a whole year.
3. **Too many small files.** Intermediates are now one file per year. Per-time-step files are
   only produced at the final export, because that is what the POP SourceMod reads.
4. **Fragile names and times.** `date_str[0:2]` and `'00' + yyy` break for years ≥ 100.
   Now uses cftime `.year`, `.dayofyr` and `.hour`, plus a configurable LR year offset.
5. **Partial files.** `os.remove` followed by three `to_netcdf` calls (one write, two appends)
   leaves broken files if a task dies. Now there is one write to `*.tmp`, then `os.replace`.
6. **Cluster setup.** `project=` is deprecated in dask-jobqueue (now `account=`), and
   20 processes × 6 GB on one node gives no isolation for I/O-bound work. Now there is one
   worker per PBS job on Casper (16 GB each), scaled to the number of years.
7. **Logging.** `print` per file is replaced by `logging` and a JSON run report listing
   each year's results and failures.
```

- [ ] **Step 2: Replace `README.md`**

```markdown
# PaleoTC

Role of tropical cyclones in equable (Eocene) climates. TC winds are extracted from the
HR Eocene simulation (ne120/t12), converted to surface wind stress on the LR ocean grid
(gx1eocene), and prescribed in LR-TC through POP SourceMods (Li et al. 2023 method).

## Setup (Casper)

    module load conda
    conda env create -f environment.yml -p /glade/work/huili7/miniconda3/envs/paleotc
    conda activate /glade/work/huili7/miniconda3/envs/paleotc
    pip install -e .
    pytest

## Run

All paths and parameters are in `config/eocene_03x.yaml`.

    qsub -v PALEOTC_ARGS="--years 31 31" scripts/submit_driver.pbs   # one year
    qsub scripts/submit_driver.pbs                                     # all years
    python scripts/qc_year.py config/eocene_03x.yaml 31                # QC

Stages (each per HR year, one dask worker per year): `extract` → `stress` → `export`.
Outputs go under `output_root`: `boxed/`, `stress/`, `pop_forcing/`, `logs/`.

## Method notes

See `docs/superpowers/plans/2026-10-15-tc-wind-stress-pipeline.md` (Design decisions)
and `docs/notes/get_tau_review.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/notes/get_tau_review.md README.md
git commit -m "Add get_tau notebook review and README"
```

---

### Task 12: Port the POP SourceMods to the Eocene LR case, with replace semantics

**Needs your confirmation before it runs.** This task changes model code, and it can only be checked by building and running an LR-TC case.

**Files:**
- Create: `sourcemods/src.pop/forcing_coupled.F90` (copy of the LR case's file plus the TC hook)
- Create: `sourcemods/src.pop/forcing_ws.F90` (iCESM1.3.1 source plus the TC module, replacing the Li et al. 2023 version)

Why a port is needed: the Li et al. 2023 `forcing_coupled.F90` is based on a 2011 POP revision. The Eocene LR case already has its own `forcing_coupled.F90` SourceMod (2017 base, with estuary and water-isotope changes), so it can't simply be dropped in. `forcing_ws.F90` from Li et al. 2023 differs from the iCESM1.3.1 version only in the TC blocks.

- [ ] **Step 1: Copy the base files**

```bash
mkdir -p sourcemods/src.pop
cp /glade/u/home/jiangzhu/cesm/cases/b.e13.B1850C5.ne30_g16.icesm13_ihesp.eocene_03x.004/SourceMods/src.pop/forcing_coupled.F90 sourcemods/src.pop/
cp /glade/campaign/cesm/development/palwg/cesm_tags/iCESM1.3.1/components/pop/source/forcing_ws.F90 sourcemods/src.pop/
git add sourcemods && git commit -m "Add unmodified base files for LR-TC POP SourceMods"
```

- [ ] **Step 2: In `forcing_coupled.F90`, change the `forcing_ws` import and add the hook**

Replace `   use forcing_ws, only: ws_data_type` with:

```fortran
!----- PaleoTC mod: TC wind stress
   use forcing_ws, only: ws_data_type, check_tc_ws
!----- PaleoTC mod end
```

In `rotate_wind_stress`, directly before the `SMFT(:,:,1,:) = (WORK1(:,:,:)*cos(ANGLET(:,:,:)) +  &` line, insert:

```fortran
!----- PaleoTC mod: replace coupler stress inside TC boxes
   call check_tc_ws(WORK1,WORK2)
!----- PaleoTC mod end
```

- [ ] **Step 3: In `forcing_ws.F90`, add the TC module code**

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
      '/glade/derecho/scratch/huili7/PaleoTC/pop_forcing/b.e13.B1850C5.ne120_t12.icesm13_ihesp.eocene_03x.002.tctau'
!----- PaleoTC mod end
```

Directly before `end module forcing_ws`, add the subroutine:

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

   real (r8), dimension(nx_block,ny_block,max_blocks_clinic), intent(inout) :: &
      WORK1, WORK2        ! taux and tauy from coupler

   integer (int_kind) :: iblock
   character (char_len) :: forcing_filename
   type (datafile) :: forcing_file
   type (io_field_desc) :: io_taux, io_tauy, io_mask
   type (io_dim) :: i_dim, j_dim
   logical (log_kind) :: iexist

   call get_forcing_filename(forcing_filename, tc_filename, thour00, c1)
   forcing_filename = trim(forcing_filename)//'.nc'
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
      write(stdout,'(a22,a)') ' TC file read: ', trim(forcing_filename)
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

- [ ] **Step 4: Review the diff against the base files**

Run: `git diff -- sourcemods/`
Expected: only the `PaleoTC mod` blocks have changed.

- [ ] **Step 5: Commit**

```bash
git add sourcemods/src.pop
git commit -m "Port TC wind-stress SourceMods to Eocene LR with replace semantics"
```

- [ ] **Step 6 (you run this; needs the LR-TC case): build, and run a 5-day smoke test**

Clone LR-cntl into LR-TC. Copy `sourcemods/src.pop/*.F90` into `SourceMods/src.pop/`, overwriting `forcing_coupled.F90`, and run `./case.build`. Expected: it compiles cleanly. Then run 5 days starting on a date with TCs. Expected: `ocn.log` shows a `TC file read:` line on every day that has a 00Z TC file, and the run completes.

---

## Self-review notes

- **Coverage against the spec:** task 2a (Python rewrite plus parallelization) is Tasks 2–9; task 2b is Task 5 plus the review notes in Task 11; task 2c is the review notes in Task 11 plus Task 9. "Organize the output" is Tasks 1, 6–8 and the output layout above. Prescribing stress in LR-TC is Task 12.
- **Open items for you:** `export.lr_year_offset` (which LR-TC model year HR year 31 maps to), and confirmation of Task 12's replace semantics. The Li et al. 2023 code *adds* the stress, which double-counts the background stress inside the box.
