import numpy as np
import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.products.bonds.bond import Bond
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes


@pytest.mark.parametrize("settlement_lag", [0, 30])
def test_zero_tent_risk_preserves_curve_metadata_and_forward_settlement(settlement_lag):
    anchor = Date(2, 1, 2026)
    settle = anchor.add_days(settlement_lag)
    bond = Bond(Date(2, 1, 2025), Date(2, 1, 2029), 0.04,
                FrequencyTypes.ANNUAL, DayCountTypes.ACT_ACT_ICMA)
    curve = FlatDiscountCurve(anchor, 0.03)
    bump = 1e-4
    _, durations = bond.key_rate_durations_zero_tent(settle, curve, [1.0, 5.0], bump)
    # The two tents partition unity. Independent cashflow sensitivity uses
    # settlement-relative ACT/365F times and normalized settlement discounts.
    payment_dates = [Date(4, 1, 2027), Date(3, 1, 2028), Date(2, 1, 2029)]
    times = np.array([(dt - settle) / 365.0 for dt in payment_dates])
    flows = np.array([4.0, 4.0, 104.0])
    pv = flows * np.exp(-0.03 * times)
    expected = np.sum(pv * times) / np.sum(pv)
    assert sum(durations) == pytest.approx(expected, rel=1e-7)


def test_z_spread_preserves_act360_curve_day_count():
    anchor = Date(2, 1, 2026)
    bond = Bond(Date(2, 1, 2025), Date(2, 1, 2029), 0.04,
                FrequencyTypes.ANNUAL, DayCountTypes.ACT_ACT_ICMA)
    curve = FlatDiscountCurve(anchor, 0.03, curve_dc_type=DayCountTypes.ACT_360)
    spread = 0.005
    combined = FlatDiscountCurve(anchor, 0.03 + spread, curve_dc_type=DayCountTypes.ACT_360)
    price = bond.clean_price_from_discount_curve(anchor, combined)
    assert bond.z_spread(anchor, price, curve) == pytest.approx(spread, abs=1e-10)
