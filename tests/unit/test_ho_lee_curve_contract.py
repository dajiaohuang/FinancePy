import numpy as np
import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.models.rates_ho_lee import ModelRatesHoLee
from financepy.utils.date import Date


@pytest.mark.parametrize("sigma,time,maturity,short_rate", [
    (0.0, 0.0, 3.0, 0.04),
    (0.0, 1.0, 3.0, 0.04),
    (0.01, 1.0, 3.0, 0.05),
    (0.01, 2.0, 2.0, 0.05),
])
def test_zero_coupon_uses_time_curve_and_instantaneous_forward(
    sigma, time, maturity, short_rate
):
    rate = 0.04
    curve = FlatDiscountCurve(Date(2, 1, 2026), rate)
    horizon = maturity - time
    # Ho-Lee bond price for a flat initial curve and instantaneous short rate.
    expected = np.exp(-short_rate * horizon - sigma**2 * time * horizon**2 / 2)
    actual = ModelRatesHoLee(sigma).zcb(short_rate, time, maturity, curve)
    assert actual == pytest.approx(expected, abs=2e-8)


@pytest.mark.parametrize("sigma,expiry,maturity", [
    (0.0, 1.0, 3.0), (0.01, 0.0, 3.0), (0.01, 2.0, 2.0)
])
@pytest.mark.parametrize("strike", [80.0, 110.0])
def test_zero_bond_option_deterministic_limit(sigma, expiry, maturity, strike):
    times = np.array([0.0, 1.0, 2.0, 3.0])
    dfs = np.exp(-0.04 * times)
    underlying_pv = 100.0 * np.exp(-0.04 * maturity)
    strike_pv = strike * np.exp(-0.04 * expiry)
    with np.errstate(divide="raise", invalid="raise"):
        actual = ModelRatesHoLee(sigma).option_on_zcb(
            expiry, maturity, strike, 100.0, times, dfs
        )
    assert actual["call"] == pytest.approx(max(underlying_pv - strike_pv, 0.0))
    assert actual["put"] == pytest.approx(max(strike_pv - underlying_pv, 0.0))
