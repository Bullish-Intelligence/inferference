"""The suites that compare a mixed batch with an isolated run choose their gate by backend.

Rule 1 (fail-closed): a mixed-batch test that does not ask ``gate_mode`` would run one
family on every backend, and an unknown backend would pass or skip by accident. This test
reads the source of each such test. It does not run the GPU path.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

# file -> the test functions that must dispatch on the backend, and the helper that keeps the token-exact path
_DISPATCH = {
    "suites/router/tests/test_router_adapter_isolation.py": (
        ["test_mixed_batch_adapter_isolation"],
        "_token_exact_isolation",
    ),
    "suites/router/tests/test_router_context_pool_gates.py": (
        ["test_router_equivalence_and_distinctiveness"],
        "_token_exact_equivalence",
    ),
    "suites/router/tests/test_router_cached_restore.py": (
        ["test_cached_shared_prefix_restore_is_token_exact", "test_run_cached_api_equivalence_and_shape"],
        "_token_exact_restore",
    ),
    "suites/p2-mixed-batch-lora/tests/test_p2_gates.py": (
        ["test_p2_mixed_batch_verdict_go"],
        "_token_exact_mixed_batch_verdict",
    ),
    "suites/p2-mixed-batch-lora/tests/test_seq_routing_gpu.py": (
        ["test_seq_routed_batch1_is_token_exact_vs_isolated_baseline"],
        "_token_exact_batch1",
    ),
}


def _calls(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("relative", sorted(_DISPATCH))
def test_the_test_asks_the_backend_and_keeps_its_token_exact_path(relative: str) -> None:
    tree = ast.parse((_REPO / relative).read_text())
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    tests, keeper = _DISPATCH[relative]
    for name in tests:
        assert "gate_mode" in _calls(functions[name]), f"{relative}::{name} does not ask gate_mode"
    assert keeper in functions, f"{relative} lost its token-exact path {keeper}"


def test_the_cached_router_runner_asks_the_backend() -> None:
    source = (_REPO / "src/inferference/benchkit/runners/cached_router.py").read_text()
    assert "gate_mode()" in source and "TOKEN_EXACT" in source
