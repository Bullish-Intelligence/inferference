import inspect

import pytest

from files import validators
from files.validators import (
    check_batch_size,
    check_retries,
    check_timeout_s,
    check_workers,
)


@pytest.mark.parametrize(
    ("fn", "low", "high", "name"),
    [
        (check_workers, 1, 64, "workers"),
        (check_retries, 0, 10, "retries"),
        (check_timeout_s, 1, 3600, "timeout_s"),
        (check_batch_size, 1, 4096, "batch_size"),
    ],
)
def test_bounds_and_messages_are_unchanged(fn, low, high, name):
    assert fn(low) == low
    assert fn(high) == high
    with pytest.raises(ValueError) as under:
        fn(low - 1)
    assert str(under.value) == f"{name} must be >= {low}, got {low - 1}"
    with pytest.raises(ValueError) as over:
        fn(high + 1)
    assert str(over.value) == f"{name} must be <= {high}, got {high + 1}"
    with pytest.raises(TypeError) as wrong:
        fn("x")
    assert str(wrong.value) == f"{name} must be an int"
    with pytest.raises(TypeError):
        fn(True)


def test_the_duplication_is_actually_gone():
    """The point of the task: one shared helper, not four copies."""
    source = inspect.getsource(validators)
    assert source.count("must be an int") == 1, "the type check is still duplicated"
    assert source.count("must be >=") == 1, "the lower-bound check is still duplicated"
    assert source.count("must be <=") == 1, "the upper-bound check is still duplicated"
