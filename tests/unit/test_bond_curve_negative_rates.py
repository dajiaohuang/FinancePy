import numpy as np
import pytest

from financepy.market.curves import bond_bootstrap_discount_curve as bootstrap
from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.market.curves.interpolator import InterpTypes
from financepy.products.bonds.bond import Bond
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("rate", [-0.01, 0.02])
@pytest.mark.parametrize("interp", [InterpTypes.NATCUBIC_LOG_DISCOUNT, InterpTypes.PCHIP_LOG_DISCOUNT])
def test_global_bond_curve_refits_negative_and_positive_rates(monkeypatch, fast, rate, interp):
    monkeypatch.setattr(bootstrap, "FAST", fast)
    anchor = Date(2, 1, 2026)
    flat = FlatDiscountCurve(anchor, rate)
    bonds = [
        Bond(Date(2, 1, 2025), anchor.add_years(years), 0.025,
             FrequencyTypes.ANNUAL, DayCountTypes.ACT_ACT_ICMA)
        for years in [1, 2, 3, 5]
    ]
    prices = [bond.clean_price_from_discount_curve(anchor, flat) for bond in bonds]
    curve = bootstrap.BondBootstrapDiscountCurve(anchor, bonds, prices, interp)
    fitted = [bond.clean_price_from_discount_curve(anchor, curve) for bond in bonds]
    np.testing.assert_allclose(fitted, prices, rtol=0.0, atol=1e-6)
    assert np.all(curve._dfs > 0.0)
    if rate < 0.0:
        assert np.all(curve._dfs[1:] > 1.0)
