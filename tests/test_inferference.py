import tomllib
from pathlib import Path

from inferference import __version__


def test_version() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(project_file.read_text())
    assert __version__ == metadata["project"]["version"]
