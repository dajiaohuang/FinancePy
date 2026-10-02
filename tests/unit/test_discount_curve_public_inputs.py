import numpy as np
import pytest

from financepy.market.curves.discount_curve import DiscountCurve
from financepy.utils.date import Date
from financepy.utils.frequency import FrequencyTypes


def make_curve():
    anchor = Date(2, 1, 2026)
    times = np.array([1.0, 2.0, 3.0])
    return DiscountCurve(anchor, [anchor.add_days(int(t * 365)) for t in times],
                         np.exp(-0.03 * times))


@pytest.mark.parametrize("frequency", [FrequencyTypes.CONTINUOUS, FrequencyTypes.ANNUAL])
def test_zero_rate_accepts_documented_list_time_input(frequency):
    curve = make_curve()
    expected = 0.03 if frequency == FrequencyTypes.CONTINUOUS else np.expm1(0.03)
    np.testing.assert_allclose(curve.zero_rate_t([1.0, 2.0, 3.0], frequency), expected,
                               rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("bump", [-0.001, 0.001])
def test_bucket_bump_constructs_curve_and_shifts_only_overlap(bump):
    curve = make_curve()
    original = curve._dfs.copy()
    bumped = curve.bump_bucket(0.5, 1.5, bump)
    overlaps = np.array([0.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(bumped._dfs, original * np.exp(-bump * overlaps))
    np.testing.assert_array_equal(curve._dfs, original)
    assert bumped.anchor_dt == curve.anchor_dt
    assert bumped.curve_dc_type == curve.curve_dc_type
