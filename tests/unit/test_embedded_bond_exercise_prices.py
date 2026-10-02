import numpy as np
import pytest

from financepy.market.curves.flat_discount_curve import FlatDiscountCurve
from financepy.models.bdt_tree import BDTTree
from financepy.models.bk_tree import BKTree
from financepy.models.hw_tree import HWTree
from financepy.products.bonds.bond_embedded_option import BondEmbeddedOption
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes


@pytest.mark.parametrize("model_type", [HWTree, BKTree, BDTTree])
@pytest.mark.parametrize("put", [False, True])
def test_past_exercise_dates_do_not_shift_remaining_exercise_prices(model_type, put):
    settle = Date(2, 1, 2027)
    curve = FlatDiscountCurve(settle, 0.03)
    past, future = Date(2, 7, 2026), Date(2, 1, 2028)
    prices = [120.0, 100.0] if put else [80.0, 100.0]

    def bond(include_past):
        dates = [past, future] if include_past else [future]
        px = np.array(prices if include_past else prices[1:])
        return BondEmbeddedOption(Date(2, 1, 2026), Date(2, 1, 2029), 0.04,
                                   FrequencyTypes.ANNUAL, DayCountTypes.ACT_ACT_ICMA,
                                   [] if put else dates, np.array([]) if put else px,
                                   dates if put else [], px if put else np.array([]))

    def model():
        if model_type is BDTTree:
            return BDTTree(0.1, 40)
        return model_type(0.01, 0.1, 40)

    actual = bond(True).value(settle, curve, model())
    expected = bond(False).value(settle, curve, model())
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)
