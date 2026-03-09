import pytest
import numpy as np
from task import pearson_corr_ad_spend_x_sales


def test_perfect_positive_correlation():
    ad_spend = np.array([1, 2, 3, 4, 5])
    sales = np.array([2, 4, 6, 8, 10])
    assert pearson_corr_ad_spend_x_sales(ad_spend, sales) == pytest.approx(1.0)


def test_perfect_negative_correlation():
    ad_spend = np.array([1, 2, 3, 4, 5])
    sales = np.array([10, 8, 6, 4, 2])
    assert pearson_corr_ad_spend_x_sales(ad_spend, sales) == pytest.approx(-1.0)


def test_realistic_marketing_data():
    ad_spend = np.array([100, 200, 300, 400, 500])
    sales = np.array([10, 20, 25, 40, 50])
    r = pearson_corr_ad_spend_x_sales(ad_spend, sales)
    assert r > 0.9   # сильная положительная корреляция


def test_with_negative_values():
    ad_spend = np.array([-1, -2, -3, -4, -5])
    sales = np.array([-10, -8, -6, -4, -2])
    assert pearson_corr_ad_spend_x_sales(ad_spend, sales) == pytest.approx(-1.0)


def test_different_lengths():
    ad_spend = np.array([1, 2, 3])
    sales = np.array([1, 2])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(ad_spend, sales)


def test_not_numpy_array():
    ad_spend = [1, 2, 3]
    sales = [4, 5, 6]
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(ad_spend, sales)


def test_multidimensional_array():
    ad_spend = np.array([[1, 2], [3, 4]])
    sales = np.array([[5, 6], [7, 8]])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(ad_spend, sales)


def test_constant_array():
    ad_spend = np.array([5, 5, 5, 5])
    sales = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        pearson_corr_ad_spend_x_sales(ad_spend, sales)
