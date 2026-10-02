import numpy as np
import pytest

from financepy.models.finite_difference import black_scholes_fd
from financepy.models.finite_difference_psor import black_scholes_fd_psor
from financepy.utils.global_types import OptionTypes


@pytest.mark.parametrize("solver", [black_scholes_fd, black_scholes_fd_psor])
@pytest.mark.parametrize("option_type", [OptionTypes.EUROPEAN_CALL, OptionTypes.EUROPEAN_PUT,
                                        OptionTypes.AMERICAN_CALL, OptionTypes.AMERICAN_PUT])
def test_zero_volatility_uses_the_deterministic_cashflow(solver, option_type):
    call = option_type in [OptionTypes.EUROPEAN_CALL, OptionTypes.AMERICAN_CALL]
    spot, strike = (150.0, 100.0) if call else (100.0, 200.0)
    expected = max(spot-strike*np.exp(-0.05), 0) if call else strike*np.exp(-0.05)-spot
    if option_type == OptionTypes.AMERICAN_PUT:
        expected = strike-spot
    actual = solver(spot, 0.0, 1.0, strike, 0.05, 0.0, option_type)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("solver", [black_scholes_fd, black_scholes_fd_psor])
def test_zero_expiry_is_the_cashflow(solver):
    assert solver(80.0, 0.2, 0.0, 100.0, 0.05, 0.0,
                  OptionTypes.AMERICAN_PUT) == 20.0


@pytest.mark.parametrize("solver", [black_scholes_fd, black_scholes_fd_psor])
def test_deterministic_american_exercise_can_be_between_endpoints(solver):
    times = np.linspace(0, 30, 100001)
    cashflows = 100*np.exp(-0.02*times)-95*np.exp(-0.05*times)
    expected = np.max(cashflows)
    assert expected > max(cashflows[0], cashflows[-1])
    actual = solver(100.0, 0.0, 30.0, 95.0, 0.05, 0.02, OptionTypes.AMERICAN_CALL)
    assert actual == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("solver", [black_scholes_fd, black_scholes_fd_psor])
@pytest.mark.parametrize("option_type,spot,strike,rate,dividend", [
    (OptionTypes.AMERICAN_CALL, 100.0, 120.0, 0.05, 0.0),
    (OptionTypes.AMERICAN_PUT, 120.0, 100.0, 0.02, 0.07),
])
def test_deterministic_digital_exercise_at_the_crossing(solver, option_type, spot, strike,
                                                       rate, dividend):
    crossing = np.log(strike/spot)/(rate-dividend)
    expected = np.exp(-rate*crossing)
    actual = solver(spot, 0.0, 10.0, strike, rate, dividend, option_type, digital=True)
    assert actual == pytest.approx(expected, rel=1e-10)
