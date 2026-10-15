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
