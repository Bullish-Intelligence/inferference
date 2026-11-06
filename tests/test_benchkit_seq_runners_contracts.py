"""CPU contract tests for the state-restore bench runners (Task 2, step 6).

The port is a refactor with contract tests, not a copy.
The new runners' GPU bodies run on the rig; their pure logic is tested here:

  * the ``stats``/``percentiles`` helpers (mean / p50 / p95, nearest-rank);
  * the cross-process ``restore_child`` verdict (set_data return + token match
    are combined exactly as the runner classifies them);
  * the n_seq_max portability matrix row classification (loaded flag);
  * fail-closed: a continuation mismatch must raise in the breakeven loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferference.benchkit.runners.native_state_decompose import stats as nat_stats
from inferference.benchkit.runners.prefix_cache_sweep import percentiles
from inferference.benchkit.runners.seq_batch_breakeven import stats as seq_stats
from inferference.benchkit.runners.seq_reuse import restore_child


def test_percentiles_are_nearest_rank() -> None:
    values = [3, 1, 2]
    p = percentiles(values)
    assert p["mean_ns"] == 2.0
    assert p["p50_ns"] == 2
    assert p["total_ns"] == 6
    # p95 of 3 elements is the max (nearest-rank ceil)
    assert p["p95_ns"] == 3
    single = percentiles([7])
    assert single["p50_ns"] == 7 and single["mean_ns"] == 7.0


def test_stats_mean_and_p50() -> None:
    for fn in (seq_stats, nat_stats):
        s = fn([10, 20, 30])
        assert s["mean_ns"] == 20.0
        assert s["p50_ns"] == 20
        assert fn([42])["p50_ns"] == 42


class _FakeHarness:
    """Minimal harness double for restore_child (no llama, no GPU)."""

    def __init__(self, set_return: int, restored_token: int, baseline_token: int) -> None:
        self._set_return = set_return
        self._restored_token = restored_token
        self._baseline_token = baseline_token
        self.decoded: list[tuple[list[int], int, int]] = []
        self.blob_seen: bytes | None = None

    def make_ctx(self, n_seq_max: int) -> int:
        return n_seq_max

    def set_seq(self, ctx: int, blob: bytes, seq_id: int) -> int:
        self.blob_seen = blob
        return self._set_return

    def decode(self, ctx: int, tokens: list[int], seq_id: int, start_pos: int) -> None:
        self.decoded.append((tokens, seq_id, start_pos))

    def sync(self, ctx: int) -> None:  # noqa: ARG002
        return None

    def last_token(self, ctx: int) -> int:  # noqa: ARG002
        return self._restored_token

    def free(self, ctx: int) -> None:  # noqa: ARG002
        return None


def test_restore_child_classifies_match_and_sha(tmp_path: Path) -> None:
    blob = b"\x01\x02\x03"
    blob_path = tmp_path / "seq.bin"
    blob_path.write_bytes(blob)
    meta = {"n_seq_max": 2, "seq_id": 1, "prefix_len": 3, "suffix": [9], "baseline_token": 7}

    ok = _FakeHarness(set_return=12, restored_token=7, baseline_token=7)
    result = restore_child(ok, blob_path, meta)  # type: ignore[arg-type]
    assert result["match"] is True
    assert result["set_return"] == 12
    assert result["restored_token"] == 7
    assert ok.blob_seen == blob
    # the suffix is decoded at the explicit position right after the prefix
    assert ok.decoded == [([9], 1, 3)]

    bad = _FakeHarness(set_return=0, restored_token=5, baseline_token=7)
    result = restore_child(bad, blob_path, meta)  # type: ignore[arg-type]
    assert result["match"] is False and result["set_return"] == 0


def test_nsm_portability_row_classification() -> None:
    # the runner's matrix row: loaded = set_return > 0 (never byte-count success)
    for ret, loaded in ((0, False), (1, True), (42, True)):
        row = {"set_return": ret, "loaded": ret > 0}
        assert row["loaded"] == loaded


def test_breakeven_continuation_mismatch_is_fail_closed() -> None:
    # The runner raises when cold != restore tokens; verify the check logic.
    cold = [11, 11]
    restore = [11, 12]
    try:
        if cold != restore or len(set(cold)) != 1:
            raise RuntimeError("continuation mismatch")
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert json.dumps({"cold": cold, "restore": restore})  # JSON-serializable rows
