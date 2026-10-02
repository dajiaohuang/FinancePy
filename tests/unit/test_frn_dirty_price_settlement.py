import pytest

from financepy.products.bonds.bond_frn import BondFRN
from financepy.utils.calendar import CalendarTypes
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes


@pytest.mark.parametrize("prime_cache", [False, True])
@pytest.mark.parametrize("settle", [Date(2, 1, 2026), Date(2, 7, 2026)])
def test_dirty_frn_price_initializes_and_refreshes_coupon_dates(prime_cache, settle):
    bond = BondFRN(Date(2, 1, 2025), Date(2, 1, 2028), 0.005,
                   FrequencyTypes.ANNUAL, DayCountTypes.ACT_360, CalendarTypes.NONE)
    if prime_cache:
        bond.clean_price_from_dm(Date(2, 7, 2025), 0.035, 0.03, 0.03, 0.005)
    actual = bond.dirty_price_from_dm(settle, 0.035, 0.03, 0.03, 0.005)
    # At the next reset future coupons and margin equal their discount rate,
    # hence continuation value is par; add the known full next coupon.
    remaining = (Date(2, 1, 2027) - settle) / 360.0
    expected = 100.0 * (1.0 + 0.035 * 365.0 / 360.0) / (1.0 + 0.035 * remaining)
    assert actual == pytest.approx(expected, abs=1e-10)
