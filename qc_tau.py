"""Quick check of one year of tau files: summary numbers and a plot of the strongest snapshot.

Usage: python qc_tau.py YEAR [config.yaml]    -> prints stats, writes qc_tau_YYYY.png
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
config_file = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
cfg = yaml.safe_load(open(config_file))

model_year = year + cfg["lr_year_offset"]
files = sorted(Path(cfg["out_tau_dir"]).glob(f"{cfg['tau_prefix']}.{model_year:04d}.*.nc"))
grid = xr.open_dataset(cfg["ocn_domain_file"])

# go through all files: count bad values, find the strongest snapshot
peak, strongest, n_bad = 0.0, None, 0
for f in files:
    ds = xr.open_dataset(f)
    tau = np.hypot(ds["TAUX"].values, ds["TAUY"].values)
    n_bad += int((~np.isfinite(tau)).sum())
    if tau.max() > peak:
        peak = float(tau.max())
        strongest = (tau, ds["TCMASK"].values == 1, ds.attrs["hr_time"])
    ds.close()
print(f"{len(files)} tau files, max |tau| = {peak:.2f} N/m2, non-finite values = {n_bad}")

# plot the strongest snapshot with the track centers at that time
tau, mask, hr_time = strongest
key = (int(hr_time[0:4]), int(hr_time[5:7]), int(hr_time[8:10]), int(hr_time[11:13]))
centers = read_tracks(cfg["track_file"]).get(key, [])
lon, lat = grid["xc"].values % 360, grid["yc"].values
land = grid["mask"].values == 0

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(lon[land], lat[land], s=0.3, c="0.8")
order = np.argsort(tau[mask])  # draw the strongest points last so they are visible
sc = ax.scatter(lon[mask][order], lat[mask][order], c=tau[mask][order], s=6, cmap="magma_r",
                vmin=0, vmax=max(peak, 0.1))
ax.plot([c[0] for c in centers], [c[1] for c in centers], "cx", ms=8, mew=2, label="track centers")
fig.colorbar(sc, label="|tau| (N/m2)")
ax.set(xlim=(0, 360), ylim=(-60, 60), xlabel="lon", ylabel="lat",
       title=f"Strongest TC wind stress snapshot of HR year {year}: {hr_time}")
ax.legend(loc="lower left")
out = f"qc_tau_{year:04d}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
