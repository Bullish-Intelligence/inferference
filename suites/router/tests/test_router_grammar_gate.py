"""Grammar-constrained routing gate (run_router_grammar validation half).

The validation half; perf is the runner in
``src/inferference/benchkit/runners/router_grammar.py``. That runner already
carries the ``Route`` schema + ``validate()`` classifier, and this gate reuses
them. xgrammar masks each sequence's logits to a JSON schema so
every routed decision is guaranteed well-formed and schema-valid.

Gate (fail-closed, deterministic greedy):
  constrained validity — EVERY constrained output parses as JSON and validates
  against the routing schema (across base + probe-a); the unconstrained run is
  reported for comparison (what the grammar buys).

GPU-only: skips (never fails) when the env lacks GPU-offload lib / model /
probe adapters / the local default-role tokenizer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.asset_paths import asset_path

from inferference.benchkit.gates import _resolve_tokenizer, load_tokenizer
from inferference.benchkit.runners.router_grammar import INSTR, USER_TASKS, Route, diagnose, validate
from inferference.gpu_env import GPU_BACKEND_LABEL, gpu_env_ready
from inferference.models import EngineConfig
from inferference.router import AdapterSpec, MultiLoRARouter, RouterConfig, RouteRequest

_REPO = Path(__file__).resolve().parents[3]
_MODEL = asset_path("MODEL_DEFAULT")
_PROBE_DIR = asset_path("MODEL_PROBE_DIR")
_LORA_A = _PROBE_DIR / "probe-a.gguf"
_LORA_B = _PROBE_DIR / "probe-b.gguf"
_TOKENIZER_DIR = asset_path("TOKENIZER_DEFAULT")


def _ready() -> bool:
    # Backend-agnostic: a vendor device variable means nothing to a build for another backend,
    # and requiring it here SKIPS this gate on a rig that can run it.
    if not gpu_env_ready():
        return False
    if not (_MODEL.exists() and _LORA_A.exists() and _LORA_B.exists() and _TOKENIZER_DIR.exists()):
        return False
    try:
        import llama_cpp
    except Exception:
        return False
    return bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())


pytestmark = pytest.mark.skipif(
    not _ready(),
    reason="requires GPU-offload lib + MODEL_DEFAULT + probe adapters + TOKENIZER_DEFAULT",
)


def test_grammar_constrained_routing_is_schema_valid() -> None:
    tokenizer = load_tokenizer(_resolve_tokenizer(str(_TOKENIZER_DIR), allow_network=False))
    router = MultiLoRARouter(
        RouterConfig(
            engine=EngineConfig(
                model_path=str(_MODEL), n_ctx=4096, n_gpu_layers=-1, n_batch=128, seed=17018, backend=GPU_BACKEND_LABEL
            ),
            adapters=(
                AdapterSpec(name="probe-a", gguf_path=str(_LORA_A)),
                AdapterSpec(name="probe-b", gguf_path=str(_LORA_B)),
            ),
            n_seq_max=8,
            include_base=True,
            backend="context_pool",
        )
    )
    try:
        router.enable_grammar(tokenizer, Route.model_json_schema())
        requests = [
            RouteRequest(
                request_id=f"{'base' if a is None else a}-{i}", prompt=INSTR + USER_TASKS[i], adapter=a, max_tokens=48
            )
            for a in (None, "probe-a")
            for i in range(len(USER_TASKS))
        ]
        constrained = router.run(requests, constrained=True)
        unconstrained = router.run(requests)

        rows = []
        all_valid = True
        unconstrained_valid = 0
        for cg, ug in zip(constrained, unconstrained, strict=True):
            cv = validate(cg.text or "")
            uv = validate(ug.text or "")
            all_valid = all_valid and cv["schema_ok"]
            unconstrained_valid += 1 if uv["schema_ok"] else 0
            rows.append(
                {
                    "rid": cg.request_id,
                    "schema_ok": cv["schema_ok"],
                    "value": cv["value"],
                    # The check stays as strict as before. The diagnosis only says WHY an output failed.
                    "diagnosis": diagnose(cg.text or "", cg.finish_reason, cg.token_ids, 48),
                }
            )
        assert all_valid, (
            f"constrained validity broken: {[r for r in rows if not r['schema_ok']]}; "
            f"unconstrained valid {unconstrained_valid}/{len(rows)}"
        )
        assert unconstrained_valid <= len(rows)  # informational
    finally:
        router.close()
