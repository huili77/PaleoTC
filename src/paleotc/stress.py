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
