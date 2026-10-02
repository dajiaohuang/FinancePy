import numpy as np
import pytest

from financepy.utils import polyfit


@pytest.fixture
def accelerated_fit(monkeypatch):
    monkeypatch.setattr(polyfit, "_USE_NUMBA_DEFAULT", True)
    monkeypatch.setattr(polyfit, "_fit_poly_impl", None)
    monkeypatch.setattr(polyfit, "_eval_poly_impl", None)
    return polyfit.fit_poly


def test_accelerated_constant_samples_have_minimum_norm_coefficients(accelerated_fit):
    actual = accelerated_fit(np.ones(3), np.full(3, 2.0), 1)
    assert hasattr(polyfit._fit_poly_impl, "signatures")
    np.testing.assert_allclose(actual, [1.0, 1.0], atol=1e-12)


def test_accelerated_underdetermined_fit_matches_minimum_norm_solution(accelerated_fit):
    actual = accelerated_fit(np.array([0.0, 1.0]), np.array([1.0, 5.0]), 3)
    np.testing.assert_allclose(actual, [4/3, 4/3, 4/3, 1.0], atol=1e-12)


def test_accelerated_fit_preserves_close_but_distinct_sample_information(accelerated_fit):
    x = 1.0 + np.arange(5)*1e-7
    actual = accelerated_fit(x, 2.0+3.0*x, 1)
    np.testing.assert_allclose(actual, [3.0, 2.0], atol=1e-7, rtol=0)
