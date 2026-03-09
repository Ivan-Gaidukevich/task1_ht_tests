import pytest
import numpy as np
from task import trim_mean_dive


def test_trim_mean_championships():
    scores = np.array([8, 9, 7, 10, 6])
    result = trim_mean_dive(scores)
    assert result == pytest.approx(8.0)  # (7+8+9)/3


def test_trim_mean_olympics():
    scores = np.array([8, 9, 7, 10, 6, 8, 9])
    result = trim_mean_dive(scores, hard_coef=2.0)
    assert result == pytest.approx(49.2)


def test_invalid_judges_count():
    scores = np.array([8, 9, 7])
    with pytest.raises(ValueError):
        trim_mean_dive(scores)


def test_scores_out_of_range():
    scores = np.array([8, 11, 7, -1, 6])
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


def test_hard_coef_effect():
    scores = np.array([6, 7, 8, 9, 10, 7, 8])
    result1 = trim_mean_dive(scores, hard_coef=1.5)
    result2 = trim_mean_dive(scores, hard_coef=2.0)
    assert result2 > result1
