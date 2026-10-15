"""Extract TC winds and air density from the HR run, one year at a time.

Python version of scripts_Li2023/extract_tcbot_batch.ncl. Same steps as the NCL:
  1. read the TC track file (TempestExtremes) and keep this year's TC points
  2. read the 6-hourly HR fields UBOT, VBOT, U10 (+ PS, TREFHT, QREFHT for air density)
  3. regrid them from ne120 to the LR ocean grid with ESMF weights
  4. at each time with TCs, keep a 20x20 deg box around every TC center; zero elsewhere
  5. write one file per TC time step: <out_bot_dir>/tcbot.YYYY-MM-DD-HH.nc

Changes from the NCL, for the Eocene runs:
  - reads the instantaneous h2 files (h4 holds 6-hourly means)
  - LR ocean grid is gx1eocene (not standard gx1v6), with its own ESMF bilinear weights
  - the box wraps around 0/360 longitude, and land points are set to 0
  - adds air density RHO = PBOT / (Rd * TREFHT * (1 + zvir * QREFHT)) for the wind stress

Usage: python extract_tcbot.py YEAR [config.yaml]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
import yaml

RDAIR = 6.02214e26 * 1.38065e-23 / 28.966  # dry-air gas constant (J/kg/K), CESM shr_const_mod
ZVIR = 28.966 / 18.016 - 1.0               # Rv/Rd - 1, CESM shr_const_mod
CHUNK = 120                                # time steps read at once (~2 GB of ne120 fields)
HR_VARS = ["UBOT", "VBOT", "U10", "PS", "TREFHT", "QREFHT"]
TIME_ENC = {"units": "days since 0001-01-01 00:00:00", "calendar": "noleap"}


def read_tracks(track_file):
    """Read a TempestExtremes track file.

    Each track starts with a line 'start <npts> <yyyy> <mm> <dd> <hh>', followed by one line
    per point: '<i> <lon> <lat> <psl> <wind> <..> <..> <yyyy> <mm> <dd> <hh>'.
    Returns {(year, month, day, hour): [(lon, lat), ...]} with lon in [0, 360).
    """
    cols = ["i", "lon", "lat", "psl", "wind", "c5", "c6", "year", "month", "day", "hour"]
    df = pd.read_csv(track_file, sep=r"\s+", header=None, names=cols)
    df = df[df["i"] != "start"]  # drop the header line of each track
    df = df.astype({"lon": float, "lat": float, "year": int, "month": int, "day": int, "hour": int})
    df["lon"] = df["lon"] % 360.0

    centers = {}
    for (year, month, day, hour), points in df.groupby(["year", "month", "day", "hour"]):
        centers[(year, month, day, hour)] = list(zip(points["lon"], points["lat"]))
    return centers


def read_weights(map_file):
    """Read an ESMF weight file as a sparse matrix W, so that dst = W @ src.

    Returns W (n_ocean_points x n_ne120_columns) and the ocean grid shape (nlat, nlon).
    """
    w = xr.open_dataset(map_file)
    nlon, nlat = w["dst_grid_dims"].values  # ESMF stores (nlon, nlat)
    W = scipy.sparse.csr_matrix(
        (w["S"].values, (w["row"].values - 1, w["col"].values - 1)),  # 1-based -> 0-based
        shape=(w.sizes["n_b"], w.sizes["n_a"]),
    )
    w.close()
    return W, (int(nlat), int(nlon))


def regrid(W, shape, data):
    """Regrid data (ntime, ncol) on ne120 -> (ntime, nlat, nlon) on the ocean grid."""
    return (W @ data.T).T.reshape(data.shape[0], *shape)


def box_mask(tlat, tlon, centers, half_width):
    """True on grid points within [-half_width, half_width) of any TC center, in both
    latitude and longitude. Longitude differences wrap around 0/360."""
    mask = np.zeros(tlat.shape, dtype=bool)
    for lon, lat in centers:
        dlon = (tlon - lon + 180.0) % 360.0 - 180.0  # in [-180, 180)
        dlat = tlat - lat
        mask |= (dlon >= -half_width) & (dlon < half_width) & (dlat >= -half_width) & (dlat < half_width)
    return mask


def hr_file(cfg, var, year):
    """Path of the HR h2 time-series file of one variable for one year."""
    pattern = f"{cfg['case']}.cam.h2.{var}.{year:04d}01*.nc"
    files = sorted(Path(cfg["hr_dir"]).glob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"{pattern} in {cfg['hr_dir']}: found {len(files)} files")
    return files[0]


def extract_year(year, cfg):
    """Write one tcbot file per TC time step of this year. Returns the number of files."""
    # 1. TC centers at each time of this year
    tracks = {key: pts for key, pts in read_tracks(cfg["track_file"]).items() if key[0] == year}

    # regridding weights, and the ocean grid (lat, lon, land/ocean mask)
    W, shape = read_weights(cfg["map_file"])
    dom = xr.open_dataset(cfg["ocn_domain_file"])
    tlat, tlon, ocean = dom["yc"].values, dom["xc"].values % 360.0, dom["mask"].values == 1
    dom.close()

    # 2. HR fields (opened lazily; read below, one chunk of time steps at a time)
    hr = {var: xr.open_dataset(hr_file(cfg, var, year)) for var in HR_VARS}

    # bottom-level pressure from the hybrid coefficients: PBOT = hyam*P0 + hybm*PS
    ps = hr["PS"]
    hyam_p0 = float(ps["hyam"][-1] * ps["P0"])
    hybm = float(ps["hybm"][-1])

    # time steps that have TCs
    times = ps["time"].values
    keys = [(t.year, t.month, t.day, t.hour) for t in times]
    tc_steps = [i for i, key in enumerate(keys) if key in tracks]
    for key in sorted(set(tracks) - set(keys)):
        print(f"warning: track time {key} not in HR output, skipped")

    out_dir = Path(cfg["out_bot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(tc_steps), CHUNK):
        steps = tc_steps[start:start + CHUNK]
        f = {var: hr[var][var].isel(time=steps).values.astype("f8") for var in HR_VARS}

        # air density on the ne120 grid
        pbot = hyam_p0 + hybm * f["PS"]
        f["RHO"] = pbot / (RDAIR * f["TREFHT"] * (1.0 + ZVIR * f["QREFHT"]))

        # 3. regrid to the ocean grid
        ocn = {var: regrid(W, shape, f[var]) for var in ["UBOT", "VBOT", "U10", "RHO"]}

        # 4-5. keep the TC boxes (over ocean) and write one file per time step
        for n, i in enumerate(steps):
            t = times[i]
            mask = box_mask(tlat, tlon, tracks[keys[i]], cfg["box_half_width"]) & ocean
            ds = xr.Dataset(
                {var: (("nlat", "nlon"), np.where(mask, ocn[var][n], 0.0).astype("f4")) for var in ocn},
                coords={"time": t},
                attrs={"ntc": len(tracks[keys[i]]), "source": cfg["case"], "map_file": cfg["map_file"]},
            )
            ds["TCMASK"] = (("nlat", "nlon"), mask.astype("i1"))
            ds["UBOT"].attrs = ds["VBOT"].attrs = ds["U10"].attrs = {"units": "m/s"}
            ds["RHO"].attrs = {"units": "kg/m3", "long_name": "air density at lowest model level"}
            ds["TCMASK"].attrs = {"long_name": "1 inside TC boxes over ocean"}

            encoding = {var: {"zlib": True, "complevel": 1} for var in ds.data_vars}
            encoding["time"] = TIME_ENC
            name = f"tcbot.{t.year:04d}-{t.month:02d}-{t.day:02d}-{t.hour:02d}.nc"
            ds.to_netcdf(out_dir / name, encoding=encoding)

        print(f"year {year}: {start + len(steps)}/{len(tc_steps)} TC time steps written")

    for ds in hr.values():
        ds.close()
    return len(tc_steps)


if __name__ == "__main__":
    year = int(sys.argv[1])
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    cfg = yaml.safe_load(open(config_file))
    extract_year(year, cfg)
