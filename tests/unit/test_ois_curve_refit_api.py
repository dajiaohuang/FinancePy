import pytest

from financepy.market.curves.ois_curve import OISCurve
from financepy.products.rates.ois_curve import OISCurve as LegacyOISCurve
from financepy.products.rates.ois import OIS
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.global_types import SwapTypes


@pytest.mark.parametrize("curve_type", [OISCurve, LegacyOISCurve])
@pytest.mark.parametrize("direction", [SwapTypes.PAY, SwapTypes.RECEIVE])
def test_constructor_refit_check_uses_current_ois_value_api(curve_type, direction):
    anchor = Date(2, 1, 2026)
    swap = OIS(anchor, "2Y", direction, 0.03,
               FrequencyTypes.ANNUAL, DayCountTypes.ACT_365F)
    curve = curve_type(anchor, [], [], [swap], check_refit_flag=True)
    assert abs(swap.value(anchor, curve, None) / swap.fixed_leg.notional) < 1e-8
