import math

import pytest

from financepy.models.vasicek_mc import variancer, zero_price_mc


@pytest.mark.parametrize("rate", [0.04, -0.02])
@pytest.mark.parametrize("maturity", [0.0, 0.05, 0.25, 0.3, 1.05])
def test_monte_carlo_discounts_the_complete_constant_rate_horizon(rate, maturity):
    actual = zero_price_mc(rate, 0.0, 0.03, 0.0, maturity, 0.1, 8, 123)
    assert actual == pytest.approx(math.exp(-rate * maturity), abs=1e-13)


@pytest.mark.parametrize("mean_reversion", [0.0, 1e-16, 1e-12])
def test_rate_variance_has_brownian_small_mean_reversion_limit(mean_reversion):
    assert variancer(mean_reversion, 0.01, 2.0) == pytest.approx(0.0002, rel=3e-12)
