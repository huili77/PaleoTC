"""Wind stress following CESM shr_flux_atmOcn (iCESM1.3.1).

Source: cime/src/share/util/shr_flux_mod.F90 and shr_const_mod.F90.
ustar is taken from the neutral 10 m wind, ustar = sqrt(cdn(u10n)) * u10n, where
u10n is CAM's U10 (= sqrt(duu10n) computed by the same routine in the HR run).
"""
from __future__ import annotations

import numpy as np

RGAS = 6.02214e26 * 1.38065e-23  # SHR_CONST_AVOGAD * SHR_CONST_BOLTZ
RDAIR = RGAS / 28.966            # SHR_CONST_RDAIR
RWV = RGAS / 18.016              # SHR_CONST_RWV
ZVIR = RWV / RDAIR - 1.0         # SHR_CONST_ZVIR
UMIN = 0.5                       # shr_flux_atmOcn minimum wind speed (m/s)


def cdn(u):
    """Neutral 10 m drag coefficient. iCESM1.3.1 has no high-wind branch."""
    return 0.0027 / u + 0.000142 + 0.0000764 * u


def air_density(ps, trefht, qrefht, hyam_bot, hybm_bot, p0):
    """Lowest-level density: exact PBOT from hybrid coefficients, 2 m T and q as proxies."""
    pbot = hyam_bot * p0 + hybm_bot * ps
    return pbot / (RDAIR * trefht * (1.0 + ZVIR * qrefht))


def wind_stress(ubot, vbot, u10n, rho):
    """Eastward/northward stress on the ocean (N/m^2), positive in the wind direction."""
    u10n = np.maximum(u10n, UMIN)
    tau = rho * cdn(u10n) * u10n**2
    vmag = np.maximum(np.hypot(ubot, vbot), UMIN)
    return tau * ubot / vmag, tau * vbot / vmag
