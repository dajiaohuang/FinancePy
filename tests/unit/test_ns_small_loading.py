import numpy as np
import pytest

from financepy.market.curves.ns_discount_curve import NSDiscountCurve
from financepy.market.curves.nss_discount_curve import NSSDiscountCurve
from financepy.utils.date import Date


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
