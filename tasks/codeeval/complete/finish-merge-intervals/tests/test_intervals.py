from files.intervals import merge_intervals


def test_empty():
    assert merge_intervals([]) == []


def test_single():
    assert merge_intervals([(3, 1)]) == [(1, 3)]


def test_disjoint_are_sorted_not_merged():
    assert merge_intervals([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]


def test_touching_merge():
    assert merge_intervals([(1, 3), (3, 5)]) == [(1, 5)]


def test_adjacent_but_not_touching_stay_apart():
    assert merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]


def test_nested_interval_is_absorbed():
    assert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]


def test_chain():
    merged = merge_intervals([(1, 4), (2, 6), (8, 9), (5, 7)])
    assert merged == [(1, 7), (8, 9)]
