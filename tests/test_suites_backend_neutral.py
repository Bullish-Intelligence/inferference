"""The suites carry no vendor assumption and name no model.

Two rules from AGENTS.md meet here. The platform is model-agnostic, and a
capability gate never claims a backend it did not check. This test scans
``suites/`` and ``tests/`` and fails on:

* a CUDA backend label in ``EngineConfig`` (use ``inferference.gpu_env.GPU_BACKEND_LABEL``);
* the word CUDA in a skip reason, or in probe text, when the need is a GPU;
* a model file in a ``suite.json`` (name a role in ``env.MODEL_ROLE`` instead);
* a model release name or a quantized GGUF file name anywhere in a suite or test.

The scan functions take text, so the self-tests below can plant a bad line and
prove that each rule fires.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()

_CUDA_BACKEND = re.compile(r"""backend\s*=\s*["']cuda["']""", re.IGNORECASE)
_CUDA_WORD = re.compile(r"\bCUDA\b", re.IGNORECASE)
# The router launcher takes the same role names (ci/runner/inferference-server-launch.sh).
_ROLE_NAME = re.compile(r"^[a-z][a-z0-9]*$")
_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Inventory variables beyond the role model: the probe adapters and the tokenizer.
_EXTRA_INVENTORY_VAR = re.compile(r'asset_path\("(MODEL_PROBE_DIR|MODEL_NANBEIGE_LORA_PROBE|TOKENIZER_[A-Z]+)"\)')
# A model release name, or a GGUF file name that carries a quantization tag.
_MODEL_NAME = re.compile(
    r"ornith|tinyllama|nanbeige\s*\d|\bQ\d(?:_[A-Z0-9]+)+\.gguf",
    re.IGNORECASE,
)


def find_cuda_backend_labels(text: str) -> list[str]:
    """Lines that pass a CUDA label as ``backend=`` to anything."""
    return [line.strip() for line in text.splitlines() if _CUDA_BACKEND.search(line)]


def _strings_in(node: ast.AST) -> Iterator[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def find_cuda_skip_reasons(source: str) -> list[str]:
    """Skip reasons (``reason=...`` or ``pytest.skip(...)``) that say CUDA."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        nodes: list[ast.AST] = [kw.value for kw in node.keywords if kw.arg == "reason"]
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "skip" and node.args:
            nodes.append(node.args[0])
        for expr in nodes:
            hits.extend(text for text in _strings_in(expr) if _CUDA_WORD.search(text))
    return hits


def find_cuda_in_probe(text: str) -> list[str]:
    """Probe lines that say CUDA. A probe asks for a GPU, not a vendor."""
    return [line.strip() for line in text.splitlines() if _CUDA_WORD.search(line)]


def find_model_names(text: str) -> list[str]:
    """Lines that name a model release or a quantized GGUF file."""
    return [line.strip() for line in text.splitlines() if _MODEL_NAME.search(line)]


def suite_json_problems(data: dict[str, Any]) -> list[str]:
    """Rule breaks in one parsed ``suite.json``."""
    problems: list[str] = []
    env = data.get("env", {})
    if "MODEL" in env:
        problems.append("env.MODEL names a model; use env.MODEL_ROLE")
    role = env.get("MODEL_ROLE")
    if not isinstance(role, str) or not _ROLE_NAME.fullmatch(role):
        problems.append("env.MODEL_ROLE must be a role name (lowercase letters and digits)")
    model_vars = env.get("MODEL_VARS", [])
    if not isinstance(model_vars, list) or not all(isinstance(v, str) and _VAR_NAME.fullmatch(v) for v in model_vars):
        problems.append("env.MODEL_VARS must be a list of variable names (UPPER_CASE)")
    gpu_tests = env.get("GPU_TESTS", [])
    if not isinstance(gpu_tests, list) or not set(gpu_tests) <= set(data.get("tests", [])):
        problems.append("env.GPU_TESTS must list files that the suite names in tests")

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
        elif isinstance(value, str) and value.lower().endswith(".gguf"):
            problems.append(f"{where} is a model file: {value}")

    walk(data, "suite.json")
    return problems


def _files(root: str, *patterns: str) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        found.update((_REPO / root).rglob(pattern))
    return sorted(p for p in found if p.resolve() != _SELF and ".venv" not in p.parts and "__pycache__" not in p.parts)


# --------------------------------------------------------------------------- #
# the scan of the tree
# --------------------------------------------------------------------------- #


def test_no_cuda_backend_label_in_suites_or_tests() -> None:
    hits = {
        str(p.relative_to(_REPO)): found
        for p in _files("suites", "*.py") + _files("tests", "*.py")
        if (found := find_cuda_backend_labels(p.read_text()))
    }
    assert not hits, f"a CUDA backend label is a claim no gate checks; use GPU_BACKEND_LABEL: {hits}"


def test_no_cuda_in_skip_reasons() -> None:
    hits = {
        str(p.relative_to(_REPO)): found
        for p in _files("suites", "*.py") + _files("tests", "*.py")
        if (found := find_cuda_skip_reasons(p.read_text()))
    }
    assert not hits, f"a skip reason says CUDA where the need is a GPU: {hits}"


def test_no_cuda_in_probe_text() -> None:
    hits = {
        str(p.relative_to(_REPO)): found
        for p in _files("suites", "capability.sh")
        if (found := find_cuda_in_probe(p.read_text()))
    }
    assert not hits, f"a probe says CUDA where the need is a GPU: {hits}"


def test_suite_json_names_roles_not_models() -> None:
    files = _files("suites", "suite.json")
    assert files, "no suite.json found; the scan is looking in the wrong place"
    problems = {
        str(p.relative_to(_REPO)): found for p in files if (found := suite_json_problems(json.loads(p.read_text())))
    }
    assert not problems, f"suite.json breaks the model-agnostic rule: {problems}"


def test_no_model_release_names_in_suites_or_tests() -> None:
    hits = {
        str(p.relative_to(_REPO)): found
        for p in _files("suites", "*.py", "*.sh", "*.json", "*.md") + _files("tests", "*.py")
        if (found := find_model_names(p.read_text()))
    }
    assert not hits, f"a suite or test names a model: {hits}"


# --------------------------------------------------------------------------- #
# the scan itself: a planted bad line must fail each rule
# --------------------------------------------------------------------------- #

_BAD_REASON_SOURCE = (
    "import pytest\n"
    "pytestmark = pytest.mark.skipif(True, reason='requires CUDA-offload lib')\n"
    "def test_x():\n"
    "    pytest.skip('no CUDA here')\n"
)


# Built from parts, so that this file itself holds no literal bad line.
_LABEL = "cu" + "da"


@pytest.mark.parametrize(
    "line",
    [
        f'EngineConfig(model_path="m", n_ctx=1, backend="{_LABEL}")',
        f"EngineConfig(model_path='m', n_ctx=1, backend = '{_LABEL.upper()}')",
    ],
)
def test_scan_catches_a_planted_cuda_backend(line: str) -> None:
    assert find_cuda_backend_labels(line)


def test_scan_accepts_the_neutral_backend_label() -> None:
    assert not find_cuda_backend_labels("EngineConfig(model_path='m', n_ctx=1, backend=GPU_BACKEND_LABEL)")


def test_scan_catches_a_planted_cuda_skip_reason() -> None:
    assert len(find_cuda_skip_reasons(_BAD_REASON_SOURCE)) == 2


def test_scan_ignores_cuda_outside_a_skip_reason() -> None:
    source = 'def f():\n    """Uses CUDA_VISIBLE_DEVICES."""\n    return "CUDA"\n'
    assert not find_cuda_skip_reasons(source)


def test_scan_catches_a_planted_cuda_probe_line() -> None:
    assert find_cuda_in_probe("# the gates are CUDA decode gates")
    assert not find_cuda_in_probe("export CUDA_VISIBLE_DEVICES=0")


@pytest.mark.parametrize(
    "line",
    [
        'MODEL = "Example-Model-9B-UD-Q4_K_XL.gguf"',
        'path = MODELS / "some-model-7B.Q4_K_M.gguf"',
        "# tinyllama is dense",
    ],
)
def test_scan_catches_a_planted_model_name(line: str) -> None:
    assert find_model_names(line)


def test_scan_accepts_an_architecture_name_and_a_neutral_gguf() -> None:
    assert not find_model_names("arch = 'nanbeige'  # a loop architecture")
    assert not find_model_names('LORA = PROBE_DIR / "probe-a.gguf"')


def test_suite_json_scan_catches_a_planted_model_file() -> None:
    bad = {"env": {"MODEL": "x.gguf", "MODEL_ROLE": "default"}, "tests": []}
    problems = suite_json_problems(bad)
    assert any("env.MODEL names a model" in p for p in problems)
    assert any(".gguf" in p for p in problems)


def test_suite_json_scan_requires_a_role() -> None:
    assert suite_json_problems({"env": {"GPU": True}})
    assert suite_json_problems({"env": {"MODEL_ROLE": "Not A Role"}})
    assert suite_json_problems({"env": {"MODEL_ROLE": "default_model"}})
    assert not suite_json_problems({"env": {"MODEL_ROLE": "default"}})


def test_suite_json_scan_checks_model_vars_and_gpu_tests() -> None:
    ok = {"env": {"MODEL_ROLE": "default", "MODEL_VARS": ["MODEL_PROBE_DIR"], "GPU_TESTS": ["t.py"]}, "tests": ["t.py"]}
    assert not suite_json_problems(ok)
    assert suite_json_problems({"env": {"MODEL_ROLE": "default", "MODEL_VARS": ["not a var"]}})
    assert suite_json_problems({"env": {"MODEL_ROLE": "default", "GPU_TESTS": ["other.py"]}, "tests": ["t.py"]})


def test_every_gpu_suite_declares_the_variables_its_tests_read() -> None:
    """A suite that reads $MODEL_PROBE_DIR (or the tokenizer) in a test lists it in env.MODEL_VARS."""
    for path in _files("suites", "suite.json"):
        data = json.loads(path.read_text())
        declared = set(data["env"].get("MODEL_VARS", []))
        used: set[str] = set()
        for rel in data.get("tests", []):
            text = (path.parent / rel).read_text()
            used |= set(
                re.findall(r'asset_path\("((?:MODEL_PROBE_DIR|MODEL_NANBEIGE_LORA_PROBE|TOKENIZER_[A-Z]+))"\)', text)
            )
        assert used <= declared, (
            f"{path.parent.name}: tests read {sorted(used - declared)} but env.MODEL_VARS lacks them"
        )
