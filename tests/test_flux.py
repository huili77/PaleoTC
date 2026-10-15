import numpy as np
import pytest

from paleotc.flux import RDAIR, ZVIR, air_density, cdn, wind_stress


def test_constants_match_shr_const():
    assert RDAIR == pytest.approx(287.04, abs=0.01)
    assert ZVIR == pytest.approx(0.6078, abs=1e-4)


def test_cdn_low_wind_formula_only():
    assert cdn(10.0) == pytest.approx(0.0027 / 10 + 0.000142 + 0.0000764 * 10)
    assert cdn(40.0) == pytest.approx(0.0027 / 40 + 0.000142 + 0.0000764 * 40)  # no high-wind cap


def test_air_density_uses_pbot():
    rho = air_density(1.0e5, 300.0, 0.02, 0.0, 0.99, 1.0e5)
    assert rho == pytest.approx(0.99e5 / (RDAIR * 300.0 * (1 + ZVIR * 0.02)))


def test_wind_stress_magnitude_and_direction():
    taux, tauy = wind_stress(np.array(30.0), np.array(40.0), np.array(40.0), np.array(1.1))
    tau = 1.1 * cdn(40.0) * 40.0**2
    assert taux == pytest.approx(0.6 * tau)
    assert tauy == pytest.approx(0.8 * tau)


def test_wind_stress_calm_is_finite():
    z = np.zeros(3)
    taux, tauy = wind_stress(z, z, z, np.full(3, 1.1))
    assert np.isfinite(taux).all() and np.isfinite(tauy).all()
    np.testing.assert_array_equal(taux, 0.0)
