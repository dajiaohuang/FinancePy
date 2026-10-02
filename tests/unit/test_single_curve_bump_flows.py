import numpy as np
import pytest

from financepy.market.curves.ibor_single_curve import IborSingleCurve
from financepy.products.rates.ibor_swap import IborSwap
from financepy.utils.date import Date
from financepy.utils.day_count import DayCountTypes
from financepy.utils.frequency import FrequencyTypes
from financepy.utils.global_types import SwapTypes


@pytest.mark.parametrize("coupon", [0.0, 0.03])
@pytest.mark.parametrize("bump", [-0.001, 0.001])
def test_parallel_quote_bump_recalibrates_cached_fixed_leg_flows(coupon, bump):
    anchor = Date(2, 1, 2026)
    swap = IborSwap(anchor, "3Y", SwapTypes.PAY, coupon,
                    FrequencyTypes.ANNUAL, DayCountTypes.THIRTY_E_360)
    base = IborSingleCurve(anchor, [], [], [swap])
    original_dfs = base._dfs.copy()
    bumped = base.bump_parallel(bump)
    benchmark = bumped.used_swaps[0]
    assert benchmark.fixed_leg.cpn == pytest.approx(coupon + bump)
    # The curve's root solver stops at a discount-factor tolerance of 1e-6.
    assert abs(benchmark.value(anchor, bumped) / benchmark.fixed_leg.notional) < 1e-7
    assert benchmark.swap_rate(anchor, bumped) == pytest.approx(coupon + bump, abs=1e-7)
    assert not np.allclose(bumped._dfs, original_dfs)
    np.testing.assert_array_equal(base._dfs, original_dfs)
    assert base.used_swaps[0].fixed_leg.cpn == coupon
