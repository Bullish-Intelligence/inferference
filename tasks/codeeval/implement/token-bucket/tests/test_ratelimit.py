import pytest

from files.ratelimit import TokenBucket


def test_starts_full():
    b = TokenBucket(capacity=10, rate=1)
    assert b.tokens(0.0) == 10


def test_consume_deducts_and_refuses_when_empty():
    b = TokenBucket(capacity=3, rate=1)
    assert b.consume(0.0, 3) is True
    assert b.consume(0.0, 1) is False
    assert b.tokens(0.0) == 0


def test_refill_is_linear_and_capped():
    b = TokenBucket(capacity=10, rate=2)
    b.consume(0.0, 10)
    assert b.tokens(1.0) == pytest.approx(2.0)
    assert b.tokens(100.0) == 10


def test_time_never_goes_backwards():
    b = TokenBucket(capacity=10, rate=2)
    b.consume(5.0, 10)
    assert b.tokens(1.0) == pytest.approx(0.0)


def test_failed_consume_deducts_nothing():
    b = TokenBucket(capacity=5, rate=1)
    b.consume(0.0, 4)
    assert b.consume(0.0, 3) is False
    assert b.tokens(0.0) == pytest.approx(1.0)


def test_wait_time():
    b = TokenBucket(capacity=10, rate=4)
    b.consume(0.0, 10)
    assert b.wait_time(0.0, 2) == pytest.approx(0.5)
    assert b.wait_time(10.0, 1) == 0.0
    with pytest.raises(ValueError):
        b.wait_time(0.0, 11)


def test_rejects_bad_construction_and_amounts():
    with pytest.raises(ValueError):
        TokenBucket(capacity=0, rate=1)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, rate=0)
    with pytest.raises(ValueError):
        TokenBucket(capacity=1, rate=1).consume(0.0, 0)
