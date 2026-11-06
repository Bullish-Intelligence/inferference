import pytest

from files.duration import parse_duration


@pytest.mark.parametrize(
    "text,expected",
    [
        ("90s", 90),
        ("1h", 3600),
        ("1h30m", 5400),
        ("2m30s", 150),
        ("  45s  ", 45),
        ("0s", 0),
        ("30m1h", 5400),
        ("1h2m3s", 3723),
    ],
)
def test_parses(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize(
    "text", ["", "   ", "10", "10x", "h", "1h1h", "1h 30m", "-5s", "1.5h", "1hx"]
)
def test_rejects(text):
    with pytest.raises(ValueError):
        parse_duration(text)
