import pytest
import numpy as np
from task import pearson_corr_ad_spend_x_sales


def test_perfect_positive_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert pytest.approx(r, 0.0001) == 1.0


def test_perfect_negative_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 8, 6, 4, 2])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert pytest.approx(r, 0.0001) == -1.0

def test_zero_correlation():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([7, 2, 9, 1, 6])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert -1.0 <= r <= 1.0


def test_same_values():
    x = np.array([5, 5, 5, 5])
    y = np.array([10, 10, 10, 10])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(x, y)


def test_different_lengths():
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(x, y)


def test_non_numpy_input():
    x = [1, 2, 3]
    y = [1, 2, 3]
    with pytest.raises(ValueError):
        pearson_correlation_numpy(x, y)

def test_multidimensional_arrays():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(x, y)

def test_too_short_arrays():
    x = np.array([1])
    y = np.array([2])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(x, y)
      

def test_arrays_with_zeros():
    x = np.array([0, 0, 5, 0, 10])
    y = np.array([1, 2, 3, 4, 5])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert -1.0 <= r <= 1.0


def test_arrays_with_negative_numbers():
    x = np.array([-1, -2, -3, -4, -5])
    y = np.array([-10, -8, -6, -4, -2])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert pytest.approx(r, 0.0001) == 1.0


def test_arrays_mixed_positive_negative():
    x = np.array([-1, 0, 1])
    y = np.array([1, 0, -1])
    r = pearson_corr_ad_spend_x_sales(x, y)
    assert pytest.approx(r, 0.0001) == -1.0
