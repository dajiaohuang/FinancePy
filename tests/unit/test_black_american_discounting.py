import numpy as np
import pytest

from financepy.models.black import Black, implied_volatility
from financepy.utils.global_types import BlackTypes, OptionTypes


def stopping_value(volatility, rate, option_type, steps):
    """Independent risk-neutral stopping recursion for a futures martingale."""
    dt = 1.0 / steps
    up = np.exp(volatility * np.sqrt(dt))
    probability = 1.0 / (1.0 + up)
    sign = 1 if option_type == OptionTypes.AMERICAN_CALL else -1
    values = np.maximum(sign * (100 * up ** (2*np.arange(steps+1)-steps) - 100), 0)
    for level in range(steps-1, -1, -1):
        continuation = np.exp(-rate*dt) * (
            probability*values[1:] + (1-probability)*values[:-1]
        )
        exercise = np.maximum(sign * (100*up**(2*np.arange(level+1)-level)-100), 0)
        values = np.maximum(continuation, exercise)
    return values[0]


@pytest.mark.parametrize("option_type", [OptionTypes.AMERICAN_CALL, OptionTypes.AMERICAN_PUT])
def test_positive_discounting_reduces_at_the_money_american_value(option_type):
    model = Black(0.2, BlackTypes.CRR_TREE, 200)
    undiscounted = model.value(100.0, 100.0, 1.0, 1.0, option_type)
    discounted = model.value(100.0, 100.0, 1.0, np.exp(-0.08), option_type)
    assert 0 < discounted < undiscounted
    reference = (stopping_value(0.2, 0.08, option_type, 200)
                 + stopping_value(0.2, 0.08, option_type, 201)) / 2
    assert discounted == pytest.approx(reference, abs=1e-9)


@pytest.mark.parametrize("option_type", [OptionTypes.AMERICAN_CALL, OptionTypes.AMERICAN_PUT])
def test_american_implied_volatility_uses_discounted_stopping_value(option_type):
    reference = (stopping_value(0.2, 0.08, option_type, 200)
                 + stopping_value(0.2, 0.08, option_type, 201)) / 2
    actual = implied_volatility(100.0, 1.0, 0.08, 100.0, reference, option_type)
    assert actual == pytest.approx(0.2, abs=1e-6)
