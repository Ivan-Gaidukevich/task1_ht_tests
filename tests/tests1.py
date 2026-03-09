import pytest
import numpy as np
from task import trim_mean_dive


def test_championships_basic():
    scores = np.array([8, 9, 7, 10, 6])
    # отбрасываем 6 и 10, среднее (7+8+9)/3 = 8
    assert trim_mean_dive(scores) == pytest.approx(8.0)


def test_championships_equal_scores():
    scores = np.array([5, 5, 5, 5, 5])
    assert trim_mean_dive(scores) == pytest.approx(5.0)


def test_championships_sorted_desc():
    scores = np.array([10, 9, 8, 7, 6])
    assert trim_mean_dive(scores) == pytest.approx(8.0)


def test_olympics_basic():
    scores = np.array([8, 9, 7, 10, 6, 8, 9])
    assert trim_mean_dive(scores) == pytest.approx(29.52)


def test_olympics_hard_coef():
    scores = np.array([6, 7, 8, 9, 10, 7, 8])
    res1 = trim_mean_dive(scores, hard_coef=1.5)
    res2 = trim_mean_dive(scores, hard_coef=2.0)
    assert res2 > res1


def test_olympics_all_equal():
    scores = np.array([7, 7, 7, 7, 7, 7, 7])
    # усечённое [7,7,7,7,7], сумма=35, итог=35*1.2*0.6=25.2
    assert trim_mean_dive(scores) == pytest.approx(25.2)


def test_olympics_sorted_desc():
    scores = np.array([10, 9, 8, 7, 6, 5, 4])
    # после сортировки [4,5,6,7,8,9,10], усечённое [5,6,7,8,9], сумма=35, итог=35*1.2*0.6=25.2
    assert trim_mean_dive(scores) == pytest.approx(25.2)


def test_invalid_judges_count_too_few():
    scores = np.array([8, 9, 7])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_invalid_judges_count_too_many():
    scores = np.array([8, 9, 7, 10, 6, 5, 8, 9])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_scores_out_of_range_negative():
    scores = np.array([8, 9, -1, 10, 6])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_scores_out_of_range_too_high():
    scores = np.array([8, 9, 11, 10, 6])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_non_numeric_scores():
    scores = np.array([8, 'a', 7, 10, 6])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_scores_not_numpy_array():
    scores = [8, 9, 7, 10, 6]
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_scores_multidimensional_array():
    scores = np.array([[8, 9, 7, 10, 6]])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)
        

def test_scores_with_zeros():
    scores = np.array([0, 0, 5, 0, 10])
    # усечённое [0,0,5], среднее = 1.6667
    assert trim_mean_dive(scores) == pytest.approx(5/3)


def test_scores_with_max_values():
    scores = np.array([10, 10, 10, 10, 10])
    # усечённое [10,10,10], среднее = 10
    assert trim_mean_dive(scores) == pytest.approx(10)
