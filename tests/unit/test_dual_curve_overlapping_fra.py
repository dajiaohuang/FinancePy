import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.market.curves.ibor_dual_curve import IborDualCurve
from financepy.market.curves.interpolator import InterpTypes
from financepy.products.rates.ibor_dual_curve import IborDualCurve as LegacyIborDualCurve
from financepy.products.rates.dual_curve import IborDualCurve as OlderIborDualCurve
from financepy.products.rates.ibor_deposit import IborDeposit
from financepy.products.rates.ibor_fra import IborFRA
from financepy.utils.calendar import CalendarTypes, BusDayAdjustTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.market.curves.ibor_single_curve import IborSingleCurve
from financepy.products.rates.ibor_single_curve import IborSingleCurve as LegacyIborSingleCurve
from financepy.market.curves.ois_curve import OISCurve
from financepy.products.rates.ois_curve import OISCurve as LegacyOISCurve


@pytest.mark.parametrize("curve_class", [IborDualCurve, LegacyIborDualCurve, OlderIborDualCurve])
@pytest.mark.parametrize("interp_type", [InterpTypes.FLAT_FWD_RATES, InterpTypes.NATCUBIC_LOG_DISCOUNT, InterpTypes.PCHIP_LOG_DISCOUNT])
def test_final_overlapping_fra_is_included_in_fitted_index_curve(curve_class, interp_type):
    value_dt = Date(2, 1, 2026)
    discount_curve = FlatDiscountCurve(value_dt, 0.03)
    deposit = IborDeposit(value_dt, "3M", 0.02, DayCountTypes.ACT_360,
                          cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    fra = IborFRA(value_dt.add_tenor("1M"), "5M", 0.04, DayCountTypes.ACT_360,
                  cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    curve = curve_class(value_dt, discount_curve, [deposit], [fra], [], interp_type=interp_type, check_refit_flag=True)
    alpha = (fra.maturity_dt - fra.start_dt) / 360.0
    expected_df = curve.df(fra.start_dt) / (1.0 + alpha * fra.fra_rate)
    assert curve.df(fra.maturity_dt) == pytest.approx(expected_df, rel=1e-10)
    assert fra.value(value_dt, discount_curve, curve) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("curve_class", [IborSingleCurve, LegacyIborSingleCurve, OISCurve, LegacyOISCurve])
@pytest.mark.parametrize("interp_type", [InterpTypes.FLAT_FWD_RATES, InterpTypes.LINEAR_ONFWD_RATES])
def test_final_overlapping_fra_is_fitted_in_single_and_ois_curves(curve_class, interp_type):
    value_dt = Date(2, 1, 2026)
    deposit = IborDeposit(value_dt, "3M", 0.02, DayCountTypes.ACT_360,
                          cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    fra = IborFRA(value_dt.add_tenor("1M"), "5M", 0.04, DayCountTypes.ACT_360,
                  cal_type=CalendarTypes.NONE, bd_type=BusDayAdjustTypes.NONE)
    curve = curve_class(value_dt, [deposit], [fra], [], interp_type=interp_type)
    alpha = (fra.maturity_dt - fra.start_dt) / 360.0
    expected_df = curve.df(fra.start_dt) / (1.0 + alpha * fra.fra_rate)
    assert curve.df(fra.maturity_dt) == pytest.approx(expected_df, rel=1e-7)
    assert abs(fra.value(value_dt, curve) / fra.notional) < 1e-7
