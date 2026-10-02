import numpy as np
import pytest
from financepy.market.curves import ibor_curve_risk_engine as current
from financepy.products.rates import ibor_curve_risk_engine as legacy
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.products.rates.ibor_swap import IborSwap
from financepy.utils.calendar import CalendarTypes, BusDayAdjustTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.global_types import SwapTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes


@pytest.mark.parametrize("engine", [current, legacy])
def test_forward_risk_retains_nondefault_curve_day_count(engine):
    anchor = Date(2, 1, 2026)
    maturity = Date(2, 1, 2027)
    curve = FlatDiscountCurve(anchor, 0.03, curve_dc_type=DayCountTypes.ACT_360)
    trade = IborSwap(anchor, maturity, SwapTypes.PAY, 0.04,
                     FrequencyTypes.ANNUAL, DayCountTypes.ACT_360,
                     cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    base_values, report, carry = engine.forward_rate_risk_report_custom_grid(curve, [anchor, maturity], [trade])
    assert len(report) == 1
    assert "trade_000" in base_values
    bumped = FlatDiscountCurve(anchor, 0.0301, curve_dc_type=DayCountTypes.ACT_360)
    expected_delta = trade.value(anchor, bumped) - trade.value(anchor, curve)
    assert report.loc[0, "DV01:trade_000"] == pytest.approx(expected_delta, abs=1e-7)


@pytest.mark.parametrize("engine", [current, legacy])
def test_parallel_shift_retains_nondefault_curve_day_count(engine):
    anchor = Date(2, 1, 2026)
    maturity = Date(2, 1, 2027)
    curve = FlatDiscountCurve(anchor, 0.03, curve_dc_type=DayCountTypes.ACT_360)
    trade = IborSwap(anchor, maturity, SwapTypes.PAY, 0.04,
                     FrequencyTypes.ANNUAL, DayCountTypes.ACT_360,
                     cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    _, report = engine.parallel_shift_ladder_report(curve, np.array([0.0001]), [trade])
    bumped = FlatDiscountCurve(anchor, 0.0301, curve_dc_type=DayCountTypes.ACT_360)
    assert report.loc[0, "PV:trade_000"] == pytest.approx(trade.value(anchor, bumped), abs=1e-7)
