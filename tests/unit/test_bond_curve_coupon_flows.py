import math
import numpy as np
import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.products.bonds.bond import Bond, CouponType
from financepy.utils.calendar import CalendarTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes


@pytest.mark.parametrize("custom_flows", [False, True])
@pytest.mark.parametrize("method", ["discount", "oas", "survival", "asset_swap"])
def test_curve_prices_use_actual_coupon_cashflows(custom_flows, method):
    anchor = Date(2, 1, 2026)
    bond = Bond(anchor, Date(2, 1, 2029), 0.04, FrequencyTypes.ANNUAL,
                DayCountTypes.ACT_360, cal_type=CalendarTypes.NONE,
                cpn_type=CouponType.ACCRUED)
    payment_dates = [Date(2, 1, 2027), Date(2, 1, 2028), Date(2, 1, 2029)]
    amounts = [0.04 * 365.0 / 360.0, 0.04 * 365.0 / 360.0, 0.04 * 366.0 / 360.0]
    if custom_flows:
        amounts = [0.02, 0.05, 0.03]
        bond.reset_flows([anchor] + payment_dates, [anchor] + payment_dates,
                         np.array([0.0] + amounts))
    curve = FlatDiscountCurve(anchor, 0.03)
    # Independent cashflow sum, including principal at the final payment.
    expected = sum(100.0 * (amount + (i == 2)) * math.exp(-0.03 * (dt - anchor) / 365.0)
                   for i, (dt, amount) in enumerate(zip(payment_dates, amounts)))
    if method == "discount":
        actual = bond.dirty_price_from_discount_curve(anchor, curve)
    elif method == "oas":
        actual = bond.dirty_price_from_oas(anchor, curve, 0.0)
    elif method == "survival":
        survival = FlatDiscountCurve(anchor, 0.0)
        actual = bond.dirty_price_from_survival_curve(anchor, curve, survival, 0.4)
    else:
        assert bond.asset_swap_spread(anchor, expected, curve) == pytest.approx(0.0, abs=1e-10)
        return
    assert actual == pytest.approx(expected, abs=1e-10)
