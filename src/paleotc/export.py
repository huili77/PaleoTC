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
