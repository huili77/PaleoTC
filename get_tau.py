"""Compute daily-mean TC wind stress from the extract_tcbot.py output, one year at a time.

POP couples the ocean once a day (OCN_NCPL=1) and reads only the 00Z file of each day.
So for each day with TCs this writes ONE file, named for 00Z, that holds the mean of the
day's 6-hourly stresses. The snapshot version (00Z only) is get_tau_00z.py, which also has
the stress formula (CESM shr_flux_atmOcn) used here.

For each ocean point and day, over the day's TC time steps (hours in tau_hours):
    TAUX, TAUY = mean over the time steps at which the point is inside a TC box (TCMASK=1)
    TCMASK     = 1 if the point is inside a box at any of those time steps
So a moving storm becomes a swath along the day's track. A day whose storms appear only
after 00Z still gets a file.

Writes <out_tau_dir>/<tau_prefix>.YYYY.DDD.00.nc, with YYYY = HR year + lr_year_offset and
DDD = day of year (1-365).

Usage: python get_tau.py YEAR [config.yaml]
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from get_tau_00z import pop_filename, wind_stress


def get_tau_year(year, cfg):
    """Write one daily-mean tau file per day with TCs. Returns the number of files written."""
    out_dir = Path(cfg["out_tau_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # group the tcbot files by day: tcbot.YYYY-MM-DD-HH.nc
    days = defaultdict(list)
    for f in sorted((Path(cfg["out_bot_dir"]) / f"{year:04d}").glob("tcbot.*.nc")):
        days[f.name.split(".")[1][:10]].append(f)

    n = 0
    for day_files in days.values():
        sum_x = sum_y = count = 0
        times = []
        for f in day_files:
            ds = xr.open_dataset(f)
            t = ds["time"].item()
            if t.hour not in cfg["tau_hours"]:
                ds.close()
                continue
            taux, tauy = wind_stress(ds["UBOT"].values, ds["VBOT"].values, ds["U10"].values, ds["RHO"].values)
            mask = ds["TCMASK"].values == 1
            ds.close()

            sum_x = sum_x + np.where(mask, taux, 0.0)
            sum_y = sum_y + np.where(mask, tauy, 0.0)
            count = count + mask
            times.append(t)
        if not times:
            continue

        inside = count > 0
        n_in = np.maximum(count, 1)
        out = xr.Dataset({
            "TAUX": (("nlat", "nlon"), np.where(inside, sum_x / n_in, 0.0).astype("f4")),
            "TAUY": (("nlat", "nlon"), np.where(inside, sum_y / n_in, 0.0).astype("f4")),
            "TCMASK": (("nlat", "nlon"), inside.astype("f4")),
        })
        out["TAUX"].attrs = {"units": "N/m2", "long_name": "TC wind stress, eastward, daily mean"}
        out["TAUY"].attrs = {"units": "N/m2", "long_name": "TC wind stress, northward, daily mean"}
        out["TCMASK"].attrs = {"long_name": "1 where TC stress replaces the coupler stress"}
        out.attrs = {"hr_times": ",".join(t.isoformat() for t in times),
                     "averaging": "mean over the listed times at which the point is inside a TC box",
                     "lr_year_offset": cfg["lr_year_offset"]}

        # POP reads netCDF-3; no _FillValue, so the field is used exactly as written
        t = times[0]
        name = pop_filename(cfg["tau_prefix"], t.year + cfg["lr_year_offset"], t.dayofyr, 0)
        encoding = {var: {"_FillValue": None} for var in out.data_vars}
        out.to_netcdf(out_dir / name, format="NETCDF3_64BIT", encoding=encoding)
        n += 1

    print(f"year {year}: {n} daily tau files written")
    return n


if __name__ == "__main__":
    year = int(sys.argv[1])
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    cfg = yaml.safe_load(open(config_file))
    get_tau_year(year, cfg)
