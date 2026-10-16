"""Compare the 00Z TC stress snapshot (what POP reads now) with the mean of the
four 6-hourly snapshots, and the wind-stress curl of each, for one storm-day.

Mean = average over the times each point is inside a box (TCMASK=1);
union mask = inside at any of the 4 times (same rule as get_tau.py).
Reads the 6-hourly files of get_tau_00z.py (out_tau_00z_dir).

Usage: python docs/figs/plot_daily_mean.py OUT.png
"""
import sys
sys.path.insert(0, "/glade/work/huili7/PaleoTC")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from extract_tcbot import read_tracks

YEAR, MONTH, DAY = 60, 8, 21
OUT = sys.argv[1]
R = 6.371e6

cfg = yaml.safe_load(open("/glade/work/huili7/PaleoTC/config.yaml"))
doy = (np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])[MONTH - 1] + DAY)
dom = xr.open_dataset(cfg["ocn_domain_file"])
lat, lon, ocean = dom["yc"].values, dom["xc"].values % 360.0, dom["mask"].values == 1

tx, ty, mk = [], [], []
for hh in (0, 6, 12, 18):
    ds = xr.open_dataset(f"{cfg['out_tau_00z_dir']}/{cfg['tau_prefix']}.{YEAR:04d}.{doy:03d}.{hh:02d}.nc")
    tx.append(ds["TAUX"].values.astype("f8"))
    ty.append(ds["TAUY"].values.astype("f8"))
    mk.append(ds["TCMASK"].values > 0.5)
tx, ty, mk = np.array(tx), np.array(ty), np.array(mk)

count = mk.sum(0)
union = count > 0
with np.errstate(invalid="ignore", divide="ignore"):
    mx = np.where(union, (tx * mk).sum(0) / count, 0.0)
    my = np.where(union, (ty * mk).sum(0) / count, 0.0)

# follow this storm through the day (other storms elsewhere share the time stamps)
STORM0 = (162.54, 20.64)  # 00Z center of the chosen storm (track file)
alltc = read_tracks(cfg["track_file"])
track, prev = [], STORM0
for hh in (0, 6, 12, 18):
    pts = alltc[(YEAR, MONTH, DAY, hh)]
    prev = min(pts, key=lambda p: ((p[0] - prev[0] + 180) % 360 - 180) ** 2 + (p[1] - prev[1]) ** 2)
    track.append(prev)
clon0, clat0 = track[0]

# index window around the storm; roll in i so the window never crosses the grid seam
j0, i0 = np.unravel_index(np.argmin((lat - clat0) ** 2 + ((lon - clon0 + 180) % 360 - 180) ** 2), lat.shape)
shift = lat.shape[1] // 2 - i0
roll = lambda a: np.roll(a, shift, axis=-1)
sj = slice(max(j0 - 30, 0), j0 + 30)
si = slice(lat.shape[1] // 2 - 30, lat.shape[1] // 2 + 30)
cut = lambda a: roll(a)[..., sj, si]

LAT, LON, OCN = cut(lat), cut(lon), cut(ocean)
LON = np.degrees(np.unwrap(np.radians(LON), axis=1))
LON = np.where(LON - clon0 > 180, LON - 360, LON)
LON = np.where(LON - clon0 < -180, LON + 360, LON)


def curl(taux, tauy):
    """Vertical curl (N/m3) of east/north stress on the curvilinear grid (chain rule)."""
    phi, lam = np.radians(LAT), np.radians(LON)
    x_j, x_i = np.gradient(R * np.cos(phi) * lam)  # local tangent-plane x, y
    y_j, y_i = np.gradient(R * phi)
    det = x_i * y_j - x_j * y_i
    d = lambda f: ((np.gradient(f, axis=1) * y_j - np.gradient(f, axis=0) * y_i) / det,
                   (np.gradient(f, axis=0) * x_i - np.gradient(f, axis=1) * x_j) / det)
    dty_dx, _ = d(tauy)
    _, dtx_dy = d(taux)
    return dty_dx - dtx_dy + taux * np.tan(phi) / R


def erode(m):
    """Drop the outer ring of the mask: the curl there sees the box edge (stress -> 0)."""
    e = m.copy()
    for ax in (0, 1):
        for s in (1, -1):
            e &= np.roll(m, s, axis=ax)
    return e


cases = {
    "00Z snapshot (what POP reads now)": (cut(tx[0]), cut(ty[0]), cut(mk[0])),
    "Mean of 00/06/12/18Z": (cut(mx), cut(my), cut(union)),
}

tau_max = max(np.hypot(a, b).max() for a, b, _ in cases.values())
curls = {k: np.where(erode(m), curl(a, b), np.nan) for k, (a, b, m) in cases.items()}
cmax = np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in curls.values()])), 99.5)
SC = 1e5  # plot curl in 1e-5 N m-3

fig, axs = plt.subplots(2, 2, figsize=(11, 9.2), constrained_layout=True, sharex=True, sharey=True)
for col, (name, (a, b, m)) in enumerate(cases.items()):
    mag = np.where(m, np.hypot(a, b), np.nan)
    ax = axs[0, col]
    ax.pcolormesh(LON, LAT, np.where(OCN, np.nan, 1), cmap="Greys", vmin=0, vmax=2.5, shading="auto")
    pm = ax.pcolormesh(LON, LAT, mag, cmap="Blues", vmin=0, vmax=tau_max, shading="auto")
    q = slice(None, None, 1)
    ax.quiver(LON[q, q], LAT[q, q], np.where(m, a, np.nan)[q, q], np.where(m, b, np.nan)[q, q],
              color="0.15", scale=120, width=0.003)
    ax.set_title(f"|τ|  {name}\nmax {np.nanmax(mag):.1f} N m⁻²", fontsize=11)

    ax = axs[1, col]
    ax.pcolormesh(LON, LAT, np.where(OCN, np.nan, 1), cmap="Greys", vmin=0, vmax=2.5, shading="auto")
    pc = ax.pcolormesh(LON, LAT, curls[name] * SC, cmap="RdBu_r", vmin=-cmax * SC, vmax=cmax * SC, shading="auto")
    c = curls[name]
    ax.set_title(f"curl τ  {name}\nmax {np.nanmax(c) * SC:.1f}, min {np.nanmin(c) * SC:.1f} ×10⁻⁵ N m⁻³", fontsize=11)

for ax in axs.flat:
    for hh, (clon, clat) in zip((0, 6, 12, 18), track):
        clon = clon if abs(clon - clon0) < 180 else clon - 360 * np.sign(clon - clon0)
        ax.plot(clon, clat, "o", ms=6, mfc="white", mec="black", mew=1.2, zorder=5)
        ax.annotate(f"{hh:02d}Z", (clon, clat), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlim(clon0 - 9, clon0 + 6)
    ax.set_ylim(clat0 - 5.5, clat0 + 7)
    ax.set_aspect(1 / np.cos(np.radians(clat0)))
    ax.grid(color="0.85", lw=0.5)
for ax in axs[1]:
    ax.set_xlabel("longitude (°E)")
for ax in axs[:, 0]:
    ax.set_ylabel("latitude (°N)")

fig.colorbar(pm, ax=axs[0], shrink=0.85, label="|τ| (N m⁻²)")
fig.colorbar(pc, ax=axs[1], shrink=0.85, label="curl τ (×10⁻⁵ N m⁻³)")
fig.suptitle(f"TC wind stress on gx1eocene, year {YEAR} {MONTH:02d}-{DAY:02d} (day {doy}); "
             "circles = track centers", fontsize=12)
fig.savefig(OUT, dpi=130)

# numbers for the write-up
for name, (a, b, m) in cases.items():
    mag = np.hypot(a, b)
    c = curls[name]
    cell = np.cos(np.radians(LAT)) * (np.pi / 180 * R) ** 2  # rough area weight for ratios only
    print(f"{name:36s} max|tau|={mag[m].max():5.2f}  boxcells={m.sum():4d}  "
          f"sum(curl+*A)={np.nansum(np.where(c > 0, c, 0) * cell):.3e}  "
          f"sum(curl*A)={np.nansum(c * cell):.3e}  sum(|tau|^1.5*A)={np.nansum(np.where(m, mag, 0) ** 1.5 * cell):.3e}")
mag4 = np.hypot(tx, ty)
print("per-snapshot max|tau|:", [f"{mag4[k][mk[k]].max():.2f}" for k in range(4)])
