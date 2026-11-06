import inspect

import pytest

from files import stats


def test_behaviour_unchanged():
    assert stats.mean([1, 2, 3]) == 2
    assert stats.clamp(5, 0, 3) == 3
    assert stats.bucket([0, 1, 5, 6], 5) == {0: 2, 1: 2}
    assert stats.top_n({1: 5, 2: 5, 3: 1}, 2) == [(1, 5), (2, 5)]
    with pytest.raises(ValueError):
        stats.mean([])


@pytest.mark.parametrize("name", ["mean", "clamp", "bucket", "top_n"])
def test_every_parameter_and_return_is_annotated(name):
    sig = inspect.signature(getattr(stats, name))
    assert sig.return_annotation is not inspect.Signature.empty, name
    for param in sig.parameters.values():
        assert param.annotation is not inspect.Parameter.empty, (name, param.name)
