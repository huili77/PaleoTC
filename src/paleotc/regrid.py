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
