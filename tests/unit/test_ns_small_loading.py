import numpy as np
import pytest

from financepy.market.curves.ns_discount_curve import NSDiscountCurve
from financepy.market.curves.nss_discount_curve import NSSDiscountCurve
from financepy.utils.date import Date
from financepy.market.curves.curve_fits import (
    CurveFitNelsonSiegel,
    CurveFitNelsonSiegelSvensson,
)


@pytest.mark.parametrize("tau", [1e12, 1e308])
@pytest.mark.parametrize("svensson", [False, True])
def test_large_finite_tau_recovers_constant_rate_limit(tau, svensson):
    anchor = Date(2, 1, 2026)
    if svensson:
        curve = NSSDiscountCurve(anchor, 0.03, -0.02, 0.01, 0.005, tau, tau)
        rates = curve.nss_cc_zero_rate(np.array([0.0, 1e-6, 1.0]))
    else:
        curve = NSDiscountCurve(anchor, 0.03, -0.02, 0.01, tau)
        rates = curve.ns_cc_zero_rate(np.array([0.0, 1e-6, 1.0]))
    # At these tau values the loading correction is below 1e-13.
    np.testing.assert_allclose(rates, 0.01, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("tau", [1e12, 1e308])
@pytest.mark.parametrize("svensson", [False, True])
def test_curve_fit_large_tau_recovers_constant_rate_limit(tau, svensson):
    if svensson:
        fit = CurveFitNelsonSiegelSvensson(0.03, -0.02, 0.01, 0.005, tau, tau)
    else:
        fit = CurveFitNelsonSiegel(0.03, -0.02, 0.01, tau)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rates = fit.interp_rate(np.array([0.0, 1e-6, 1.0]))
    np.testing.assert_allclose(rates, 0.01, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("svensson", [False, True])
def test_regular_tenors_match_independent_loading_formula(svensson):
    times = np.array([0.25, 1.0, 5.0, 30.0])
    theta = times / 2.0
    loading = (1.0 - np.exp(-theta)) / theta
    expected = 0.03 - 0.02 * loading + 0.01 * (loading - np.exp(-theta))
    anchor = Date(2, 1, 2026)
    if svensson:
        theta2 = times / 5.0
        expected += 0.005 * ((1.0 - np.exp(-theta2)) / theta2 - np.exp(-theta2))
        fit = CurveFitNelsonSiegelSvensson(0.03, -0.02, 0.01, 0.005, 2.0, 5.0)
        curve = NSSDiscountCurve(anchor, 0.03, -0.02, 0.01, 0.005, 2.0, 5.0)
        actual = curve.nss_cc_zero_rate(times)
    else:
        fit = CurveFitNelsonSiegel(0.03, -0.02, 0.01, 2.0)
        curve = NSDiscountCurve(anchor, 0.03, -0.02, 0.01, 2.0)
        actual = curve.ns_cc_zero_rate(times)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(fit.interp_rate(times), expected, rtol=1e-14, atol=1e-15)
