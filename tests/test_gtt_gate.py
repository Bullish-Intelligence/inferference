import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "ci/gates/gtt-check.sh"


def _run(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$1"; {command}', "bash", str(HELPER), *args], capture_output=True, text=True
    )


def test_gtt_counter_requires_a_readable_number(tmp_path: Path) -> None:
    counter = tmp_path / "mem_info_gtt_used"
    assert _run('gtt_read_bytes "$2"', str(counter)).returncode != 0
    counter.write_text("invalid\n")
    assert _run('gtt_read_bytes "$2"', str(counter)).returncode != 0
    counter.write_text("123\n")
    result = _run('gtt_read_bytes "$2"', str(counter))
    assert result.returncode == 0
    assert result.stdout == "123\n"


@pytest.mark.parametrize(("before", "after", "passes"), [(100, 100, True), (100, 99, True), (100, 101, False)])
def test_gtt_counter_must_not_increase(before: int, after: int, passes: bool) -> None:
    result = _run('gtt_did_not_increase "$2" "$3"', str(before), str(after))
    assert (result.returncode == 0) is passes
