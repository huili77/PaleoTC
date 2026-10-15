"""Box masks around TC centers on a curvilinear grid."""
from __future__ import annotations

import numpy as np


def box_mask(tlat, tlon, centers, half_width):
    """True where a grid point lies in [-hw, hw) of any center in both lat and wrapped lon."""
    mask = np.zeros(tlat.shape, dtype=bool)
    for lon, lat in centers:
        dlon = (tlon - lon + 180.0) % 360.0 - 180.0
        dlat = tlat - lat
        mask |= (dlon >= -half_width) & (dlon < half_width) & (dlat >= -half_width) & (dlat < half_width)
    return mask
