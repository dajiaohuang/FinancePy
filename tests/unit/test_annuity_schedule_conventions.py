import pytest

from financepy.products.bonds.bond_annuity import BondAnnuity
from financepy.utils.calendar import CalendarTypes, BusDayAdjustTypes, DateGenRuleTypes
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes


def test_annuity_uses_requested_forward_generation_rule():
    annuity = BondAnnuity(Date(20, 9, 2026), 0.04, FrequencyTypes.QUARTERLY,
                          cal_type=CalendarTypes.NONE, dg_type=DateGenRuleTypes.FORWARD)
    annuity.calculate_payments(Date(2, 1, 2026), 100.0)
    assert annuity.cpn_dts[1] == Date(2, 4, 2026)
    assert annuity.flow_amounts[1] == pytest.approx(100.0 * 0.04 * 90.0 / 360.0)


def test_annuity_uses_requested_preceding_business_day_rule():
    # Choose a maturity grid whose first unadjusted coupon is Saturday 4 April.
    weekend = BondAnnuity(Date(4, 1, 2027), 0.04, FrequencyTypes.QUARTERLY,
                          bd_type=BusDayAdjustTypes.PRECEDING)
    weekend.calculate_payments(Date(4, 1, 2026), 100.0)
    assert weekend.cpn_dts[1] == Date(3, 4, 2026)
