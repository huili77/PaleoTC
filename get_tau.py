"""Compute TC wind stress from the extract_tcbot.py output, one year at a time.

Replaces scripts_Li2023/get_tau_dask.ipynb. The wind stress follows CESM shr_flux_atmOcn
(iCESM1.3.1 cime/src/share/util/shr_flux_mod.F90):
    tau        = rho * cdn(U10) * U10**2     U10 = neutral 10 m wind (CAM U10 = sqrt(duu10n))
    TAUX, TAUY = tau * (UBOT, VBOT) / |(UBOT, VBOT)|     eastward / northward, N/m2

Changes from the notebook (details in docs/notes/get_tau_review.md):
  - air density from extract_tcbot.py (RHO) instead of a constant 1.2 kg/m3
  - only the low-wind cdn formula, as in iCESM1.3.1 (the notebook's high-wind branch never ran)
  - file names from the model time (year, day of year, hour), valid for any year
  - TCMASK is written too, so the POP SourceMod can replace the coupler stress inside the boxes

Writes the files the POP SourceMod reads: <out_tau_dir>/<tau_prefix>.YYYY.DDD.HH.nc
with YYYY = HR year + lr_year_offset (POP model year of LR-TC), DDD = day of year (1-365).

Usage: python get_tau.py YEAR [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

UMIN = 0.5  # minimum wind speed (m/s), as in shr_flux_atmOcn


def cdn(u):
    """Neutral 10 m drag coefficient (shr_flux_mod.F90)."""
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def wind_stress(ubot, vbot, u10, rho):
    """Eastward and northward wind stress on the ocean (N/m2)."""
    u10 = np.maximum(u10, UMIN)                     # avoids cdn(0) = inf
    tau = rho * cdn(u10) * u10**2                   # = rho * ustar**2
    vmag = np.maximum(np.hypot(ubot, vbot), UMIN)
    return tau * ubot / vmag, tau * vbot / vmag


def pop_filename(prefix, year, doy, hour):
    """File name POP looks for (forcing_tools.F90:get_forcing_filename), plus '.nc'."""
    return f"{prefix}.{year:04d}.{doy:03d}.{hour:02d}.nc"


def get_tau_year(year, cfg):
    """Write one tau file per tcbot file of this year. Returns the number of files written."""
    out_dir = Path(cfg["out_tau_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted((Path(cfg["out_bot_dir"]) / f"{year:04d}").glob("tcbot.*.nc")):
        ds = xr.open_dataset(f)
        t = ds["time"].item()
        if t.hour not in cfg["tau_hours"]:
            ds.close()
            continue

        taux, tauy = wind_stress(ds["UBOT"].values, ds["VBOT"].values, ds["U10"].values, ds["RHO"].values)
        mask = ds["TCMASK"].values == 1
        ds.close()

        out = xr.Dataset({
            "TAUX": (("nlat", "nlon"), np.where(mask, taux, 0.0).astype("f4")),
            "TAUY": (("nlat", "nlon"), np.where(mask, tauy, 0.0).astype("f4")),
            "TCMASK": (("nlat", "nlon"), mask.astype("f4")),
        })
        out["TAUX"].attrs = {"units": "N/m2", "long_name": "TC wind stress, eastward"}
        out["TAUY"].attrs = {"units": "N/m2", "long_name": "TC wind stress, northward"}
        out["TCMASK"].attrs = {"long_name": "1 where TC stress replaces the coupler stress"}
        out.attrs = {"hr_time": t.isoformat(), "lr_year_offset": cfg["lr_year_offset"]}

        # POP reads netCDF-3; no _FillValue, so the field is used exactly as written
        name = pop_filename(cfg["tau_prefix"], t.year + cfg["lr_year_offset"], t.dayofyr, t.hour)
        encoding = {var: {"_FillValue": None} for var in out.data_vars}
        out.to_netcdf(out_dir / name, format="NETCDF3_64BIT", encoding=encoding)
        n += 1

    print(f"year {year}: {n} tau files written")
    return n


if __name__ == "__main__":
    year = int(sys.argv[1])
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    cfg = yaml.safe_load(open(config_file))
    get_tau_year(year, cfg)
