import pytest

from files.client import BASE_DELAY, MAX_ATTEMPTS, Client
from files.errors import TransientError


class Recorder:
    def __init__(self, fail_times, result="ok"):
        self.fail_times = fail_times
        self.result = result
        self.calls = []
        self.slept = []

    def transport(self, *args):
        self.calls.append(args)
        if len(self.calls) <= self.fail_times:
            raise TransientError("boom")
        return self.result

    def sleep(self, seconds):
        self.slept.append(seconds)


@pytest.mark.parametrize(
    "method,args,expected",
    [
        ("get", ("/a",), ("GET", "/a")),
        ("put", ("/a", {"k": 1}), ("PUT", "/a", {"k": 1})),
        ("delete", ("/a",), ("DELETE", "/a")),
    ],
)
def test_passes_arguments_through(method, args, expected):
    r = Recorder(fail_times=0)
    c = Client(r.transport, r.sleep)
    assert getattr(c, method)(*args) == "ok"
    assert r.calls == [expected]
    assert r.slept == []


@pytest.mark.parametrize("method,args", [("get", ("/a",)), ("delete", ("/a",))])
def test_retries_then_succeeds_with_exponential_backoff(method, args):
    r = Recorder(fail_times=2)
    c = Client(r.transport, r.sleep)
    assert getattr(c, method)(*args) == "ok"
    assert len(r.calls) == 3
    assert r.slept == [BASE_DELAY, BASE_DELAY * 2]


def test_raises_after_the_last_attempt_and_does_not_sleep_again():
    r = Recorder(fail_times=99)
    c = Client(r.transport, r.sleep)
    with pytest.raises(TransientError):
        c.put("/a", None)
    assert len(r.calls) == MAX_ATTEMPTS
    assert len(r.slept) == MAX_ATTEMPTS - 1


def test_attempt_counter_is_shared():
    r = Recorder(fail_times=1)
    c = Client(r.transport, r.sleep)
    c.get("/a")
    c.get("/b")
    assert c.attempts == 3


def test_the_retry_loop_exists_once():
    """A behaviour-preserving change needs a structural check as well.

    Without this, returning the input unchanged would pass every test above.
    """
    import inspect

    from files import client

    source = inspect.getsource(client)
    assert source.count("for attempt in range") == 1
    assert source.count("except TransientError") == 1
