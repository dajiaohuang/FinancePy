import numpy as np
import pytest

from financepy.models.finite_difference import option_payoff
from financepy.utils.global_types import OptionTypes


@pytest.mark.parametrize("strike", [70.0, 100.0])
@pytest.mark.parametrize("option_type", [OptionTypes.EUROPEAN_CALL, OptionTypes.EUROPEAN_PUT])
def test_smoothed_payoff_preserves_endpoint_cashflows(strike, option_type):
    spots = np.array([80.0, 100.0, 120.0])
    sign = 1 if option_type == OptionTypes.EUROPEAN_CALL else -1
    result = option_payoff(spots, strike, True, False, option_type.value)[0]
    assert np.all(result >= 0)
    np.testing.assert_allclose(result[[0, -1]], np.maximum(sign*(spots[[0, -1]]-strike), 0))
