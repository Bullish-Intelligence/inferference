import pytest

from files.semver import parse_version


def test_plain_version():
    assert parse_version("1.2.3") == (1, 2, 3, None)


def test_zero_is_allowed():
    assert parse_version("0.0.0") == (0, 0, 0, None)


def test_prerelease():
    assert parse_version("1.0.0-rc.1") == (1, 0, 0, "rc.1")
    assert parse_version("1.0.0-alpha-2") == (1, 0, 0, "alpha-2")


@pytest.mark.parametrize("bad", ["", "1.2", "1.2.3.4", "1..3", "a.2.3", "1.2.-3"])
def test_malformed_core_is_rejected(bad):
    with pytest.raises(ValueError):
        parse_version(bad)


def test_leading_zero_is_rejected():
    with pytest.raises(ValueError):
        parse_version("01.2.3")


def test_empty_prerelease_is_rejected():
    with pytest.raises(ValueError):
        parse_version("1.2.3-")


def test_illegal_prerelease_character_is_rejected():
    with pytest.raises(ValueError):
        parse_version("1.2.3-rc_1")


def test_non_string_is_rejected():
    with pytest.raises(ValueError):
        parse_version(None)
