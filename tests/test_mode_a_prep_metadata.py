"""The Mode A vendor source tree keeps build inputs and drops management files."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_prune_project_metadata_keeps_build_inputs(tmp_path: Path) -> None:
    source = tmp_path / "llama"
    (source / ".agents" / "skills" / "example").mkdir(parents=True)
    (source / ".agents" / "skills" / "example" / "SKILL.md").write_text("skill\n")
    (source / ".copier-answers.example.yml").write_text("source: template\n")
    (source / "nested" / ".copier-answers.example.yaml").parent.mkdir()
    (source / "nested" / ".copier-answers.example.yaml").write_text("source: template\n")
    (source / "CMakeLists.txt").write_text("project(example)\n")
    (source / "llama.cpp").write_text("int main(void) { return 0; }\n")

    helper = Path(__file__).parents[1] / "ci" / "build" / "prune-project-metadata.sh"
    subprocess.run(["bash", str(helper), str(source)], check=True)

    assert not (source / ".agents").exists()
    assert not list(source.rglob(".copier-answers*.yml"))
    assert not list(source.rglob(".copier-answers*.yaml"))
    assert (source / "CMakeLists.txt").read_text() == "project(example)\n"
    assert (source / "llama.cpp").read_text() == "int main(void) { return 0; }\n"
