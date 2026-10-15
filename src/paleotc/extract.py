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
