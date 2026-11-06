import pytest

from files.windows import rolling_max, windows


def test_window_count():
    assert len(list(windows([1, 2, 3, 4], 2))) == 3


def test_last_window_is_included():
    assert list(windows([1, 2, 3, 4], 2))[-1] == [3, 4]


def test_full_length_window():
    assert list(windows([1, 2, 3], 3)) == [[1, 2, 3]]


def test_window_larger_than_input():
    assert list(windows([1, 2], 5)) == []


def test_rolling_max():
    assert rolling_max([1, 5, 2, 4], 2) == [5, 5, 4]


def test_rejects_zero_size():
    with pytest.raises(ValueError):
        list(windows([1, 2], 0))
