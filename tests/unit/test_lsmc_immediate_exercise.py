import numpy as np
import pytest

from financepy.models.black_scholes import BlackScholes
from financepy.utils.global_types import BlackScholesTypes, OptionTypes


@pytest.mark.parametrize("volatility", [0.01, 0.2])
@pytest.mark.parametrize("steps", [1, 4])
def test_american_lsmc_does_not_forget_exercise_on_the_valuation_date(volatility, steps):
    model = BlackScholes(volatility, BlackScholesTypes.LSMC,
                         num_steps_per_year=steps, num_paths=1000, seed=42)
    result = model.value(100.0, 1.0, 1000.0, 0.05, 0.0, OptionTypes.AMERICAN_PUT)
    assert result == pytest.approx(900.0)


def test_european_lsmc_continues_to_discount_to_maturity():
    model = BlackScholes(0.01, BlackScholesTypes.LSMC,
                         num_steps_per_year=1, num_paths=1000, seed=42)
    result = model.value(100.0, 1.0, 1000.0, 0.05, 0.0, OptionTypes.EUROPEAN_PUT)
    # All sampled terminal prices are below the strike; the model's exact
    # sample-forward correction gives this discounted linear payoff.
    assert result == pytest.approx(1000*np.exp(-0.05)-100)
