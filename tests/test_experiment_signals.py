"""Test that `run` and `run-pair` stop their servers on SIGTERM and on SIGINT.

Each test starts a child Python process. The child runs the real command path
on the fabricated host from the runner tests, with fake servers. The test
sends the signal to that child only. Nothing here signals the pytest process
or any process that the test did not start.

Each fake server counts the SIGINTs it gets. Two SIGINTs deadlock the real
server, so the count must be exactly one.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from .test_experiment_runner import _wait_for

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHILD_TIMEOUT_S = 60.0

_CHILD_SOURCE = """
import sys
from tests.test_experiment_signals import child_main
sys.exit(child_main(sys.argv[1], sys.argv[2]))
"""


# ---- the child process ---------------------------------------------------


def _write_info(root: Path, artifact_root: Path) -> None:
    """Write `info.json` whole or not at all, so the parent never reads half of it."""
    temporary = root / "info.json.tmp"
    temporary.write_text(json.dumps({"artifact_root": str(artifact_root)}))
    os.replace(temporary, root / "info.json")


def child_main(kind: str, root_text: str) -> int:
    """Run the `run` or `run-pair` command on a fabricated host, then return its status.

    The child writes `info.json` when the host exists, so the parent knows
    where the run artifacts will appear.
    """
    import pytest

    from inferference.experiment import cli
    from tests.test_experiment_cli import _target_args
    from tests.test_experiment_pair import _make_pair_host, _PairLoader
    from tests.test_experiment_runner import _FakeResidentLoader, _make_host

    root = Path(root_text)
    patcher = pytest.MonkeyPatch()
    sigint_files = {card: root / f"sigint-card{card}" for card in (0, 1)}
    if kind == "run-pair":
        with _make_pair_host(root / "host", workloads=("soak_120s",)) as pair:
            host = pair.host
            patcher.setattr(cli, "_host_paths", lambda: host.paths)
            patcher.setattr(cli, "unit_active", lambda: True)
            patcher.setattr(cli, "_fan_lease_factory", host.lease_factory)
            real_run_pair = cli.run_pair
            card_envs = {
                card: {**env, "FAKE_SERVER_SIGINT_COUNT_FILE": str(sigint_files[card])}
                for card, env in pair.card_envs.items()
            }

            def run_pair_with_card_envs(*args: Any, **kwargs: Any) -> Any:
                return real_run_pair(*args, card_envs=card_envs, http_timeout_s=3, rendezvous_timeout_s=20, **kwargs)

            patcher.setattr(cli, "run_pair", run_pair_with_card_envs)
            os.environ["FAKE_SERVER_DRM_ROOT"] = str(host.paths.drm_root)
            os.environ["GPU_LEASE_BUDGET_FILE"] = str(host.lease_budget_file)
            argv = [
                "run-pair",
                str(host.spec_path),
                "--cards",
                "0,1",
                "--ports",
                f"{pair.targets[0].port},{pair.targets[1].port}",
                "--server-bin",
                str(host.target.server_bin),
                "--config-root",
                str(host.target.config_root),
                "--stock-pp-table",
                str(host.target.stock_pp_table),
                "--thermal-guard",
                str(host.target.thermal_guard),
                "--gpu-lease",
                str(host.target.gpu_lease),
                "--expect-pci",
                f"0={pair.targets[0].expect_pci}",
                "--expect-pci",
                f"1={pair.targets[1].expect_pci}",
                "--stall-floor",
                "2=1000",
                "--artifact-root",
                str(host.artifact_root),
            ]
            _write_info(root, host.artifact_root)
            with _PairLoader(pair):
                return cli.main(argv)
    with _make_host(root / "host", workloads=("soak_120s",)) as host:
        patcher.setattr(cli, "_host_paths", lambda: host.paths)
        patcher.setattr(cli, "unit_active", lambda: True)
        patcher.setattr(cli, "_fan_lease_factory", host.lease_factory)
        os.environ["FAKE_SERVER_SIGINT_COUNT_FILE"] = str(sigint_files[0])
        os.environ["FAKE_SERVER_DRM_ROOT"] = str(host.paths.drm_root)
        os.environ["GPU_LEASE_BUDGET_FILE"] = str(host.lease_budget_file)
        _write_info(root, host.artifact_root)
        with _FakeResidentLoader(host):
            return cli.main(["run", *_target_args(host.target, host.spec_path, host.artifact_root)])


# ---- the parent side -----------------------------------------------------


def _start_child(kind: str, root: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD_SOURCE, kind, str(root)],
        cwd=_REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _soak_requests(artifact_root: Path, cards: tuple[int, ...]) -> bool:
    """Has every run sent at least one request? Then the soak runs and the servers are up."""
    dirs = [artifact_root / f"card{card}" for card in cards] if len(cards) > 1 else [artifact_root]
    for directory in dirs:
        if not any(directory.glob("*/requests/*.request.json")):
            return False
    return True


def _finish(child: subprocess.Popen[str]) -> tuple[int, str, str]:
    try:
        out, err = child.communicate(timeout=_CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        child.kill()
        out, err = child.communicate()
        pytest.fail(f"the child did not stop after the signal.\nstdout: {out}\nstderr: {err}")
    return child.returncode, out, err


def _signal_when_serving(child: subprocess.Popen[str], root: Path, cards: tuple[int, ...], *signums: int) -> None:
    assert _wait_for(lambda: (root / "info.json").is_file() or child.poll() is not None, timeout_s=30)
    assert child.poll() is None, "the child exited before it started the run"
    artifact_root = Path(json.loads((root / "info.json").read_text())["artifact_root"])
    assert _wait_for(lambda: _soak_requests(artifact_root, cards) or child.poll() is not None, timeout_s=45)
    assert child.poll() is None, "the child exited before the soak started"
    for signum in signums:
        child.send_signal(signum)
        time.sleep(0.05)


def _outcomes(artifact_root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(artifact_root.glob("**/outcome.json"))]


@pytest.mark.parametrize("signums", [(signal.SIGTERM,), (signal.SIGINT,), (signal.SIGTERM, signal.SIGINT)])
def test_run_pair_stops_both_servers_with_one_sigint_each(tmp_path: Path, signums: tuple[int, ...]) -> None:
    child = _start_child("run-pair", tmp_path)
    try:
        _signal_when_serving(child, tmp_path, (0, 1), *signums)
        code, out, err = _finish(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert code == 1, err
    payload = json.loads(out)
    assert payload["passed"] is False
    runs = payload["pairs"][0]["runs"]
    expected_boundary = f"signal:{signal.Signals(signums[0]).name}"
    for card in ("0", "1"):
        assert runs[card]["state"] == "aborted"
        assert runs[card]["failure_boundary"] == expected_boundary
        assert (tmp_path / f"sigint-card{card}").read_text() == "1"
    artifact_root = Path(json.loads((tmp_path / "info.json").read_text())["artifact_root"])
    outcomes = _outcomes(artifact_root)
    assert [item["failure_boundary"] for item in outcomes] == [expected_boundary] * 2
    for run_dir in artifact_root.glob("card*/*"):
        assert (run_dir / "server.log").is_file()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_run_stops_its_server_with_one_sigint(tmp_path: Path, signum: int) -> None:
    child = _start_child("run", tmp_path)
    try:
        _signal_when_serving(child, tmp_path, (0,), signum)
        code, out, err = _finish(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()

    assert code == 1, err
    payload = json.loads(out)
    assert payload["passed"] is False
    expected_boundary = f"signal:{signal.Signals(signum).name}"
    assert payload["runs"][0]["state"] == "aborted"
    assert payload["runs"][0]["failure_boundary"] == expected_boundary
    assert (tmp_path / "sigint-card0").read_text() == "1"
    artifact_root = Path(json.loads((tmp_path / "info.json").read_text())["artifact_root"])
    assert [item["failure_boundary"] for item in _outcomes(artifact_root)] == [expected_boundary]


def test_a_stop_request_keeps_the_first_reason() -> None:
    from inferference.experiment.stop import StopRequest

    request = StopRequest()
    assert request.reason is None
    assert request.request("SIGTERM") is True
    assert request.request("SIGINT") is False
    assert request.is_set()
    assert request.reason == "SIGTERM"


def test_the_signal_scope_restores_the_old_handlers() -> None:
    from inferference.experiment import cli

    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    with cli._stop_on_signals() as request:
        assert signal.getsignal(signal.SIGTERM) is not before[signal.SIGTERM]
        assert not request.is_set()
    assert {signum: signal.getsignal(signum) for signum in before} == before
