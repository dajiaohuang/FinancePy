import numpy as np
import pytest

from financepy.products.bonds.bond_zero import BondZero
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.error import FinError


@pytest.mark.parametrize("as_list", [False, True])
def test_zero_bond_vector_yields_match_analytic_prices(as_list):
    issue = Date(2, 1, 2026)
    bond = BondZero(issue, Date(2, 1, 2028), 95.0)
    rates = np.array([-0.01, 0.0, 0.03])
    inputs = rates.tolist() if as_list else rates
    years = (Date(2, 1, 2028) - issue) / 365.0
    np.testing.assert_allclose(bond.dirty_price_from_ytm(issue, inputs), 100.0 / (1.0 + rates) ** years)


@pytest.mark.parametrize("day_count,denominator", [(DayCountTypes.ACT_365F, 365.0), (DayCountTypes.ACT_360, 360.0)])
def test_current_yield_uses_configured_day_count(day_count, denominator):
    issue, maturity = Date(2, 1, 2026), Date(2, 1, 2028)
    bond = BondZero(issue, maturity, 95.0, day_count)
    expected = 5.0 / ((maturity - issue) / denominator) / 96.0
    assert bond.current_yield(96.0) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("rates", [[0.03, -0.21], [2.01, 0.03]])
def test_vector_yield_validation_checks_every_element(rates):
    bond = BondZero(Date(2, 1, 2026), Date(2, 1, 2028), 95.0)
    with pytest.raises(FinError, match="YTM must be"):
        bond.dirty_price_from_ytm(Date(2, 1, 2026), rates)
