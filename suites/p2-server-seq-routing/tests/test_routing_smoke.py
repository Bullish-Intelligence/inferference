"""Routing serves on THIS architecture — the per-model smoke gate.

`15` surveyed 103 of 140 architectures onto the ungated row-reduction pattern
behind R12, and only `llama` was ever run. Auditing the other 102 is a code
reading that goes stale on the next anchor bump. This is the gate instead: point
it at a (model, adapter) pair and it answers by decoding.

The five arms and why the controls are not optional are in
``suites/p2-server-seq-routing/routing_smoke.py``. In short: the three control
arms are what located all three failed builds.

**Adding an architecture is two commands, not a code change.** Synthesize the
adapter sets, then name the model here or in ``ROUTING_SMOKE_MODELS``::

    ci/library/.venv/bin/python suites/p2-server-seq-routing/make_arch_lora.py \\
        --model <model.gguf> --out-dir "$ROUTING_SMOKE_ADAPTERS"

**Fail-closed.** A model or adapter set that is absent is a SKIP carrying that
command. A model that is present and does not serve is a FAILURE.

GPU: each arm launches a server, but this is a CI job, so it takes the CI role's
device (``$LLAMA_CI_GPU``; ``$ROUTING_SMOKE_GPU`` overrides). It asserts the
variable, never a literal index.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SUITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_DIR))

import routing_smoke  # noqa: E402

MODELS = Path(os.environ.get("INFERFERENCE_MODELS_DIR", "~/.cache/inferference/models")).expanduser()
SMOKE_ROOT = Path(os.environ.get("ROUTING_SMOKE_ADAPTERS", str(MODELS / "routing-smoke")))

# (adapter subdirectory, model file). Three architectures, each a different
# graph shape:
#   llama    the arch R12 was found and fixed on
#   nanbeige looped; its row reduction sits outside the layer loop
#   qwen35   hybrid, and irregular: 24 SSM blocks and 8 full-attention blocks at
#            3, 7, ... 31. The gate handles it without a constant because
#            arch_blocks reads the structure out of the GGUF.
DEFAULT_MODELS = [
    ("tinyllama", MODELS / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
    ("nanbeige", Path(os.environ.get("MODEL_NANBEIGE", str(MODELS / "Nanbeige4.2-3B-UD-Q4_K_XL.gguf")))),
    ("ornith", Path(os.environ.get("MODEL_ORNITH", str(MODELS / "Ornith-1.0-9B-UD-Q4_K_XL.gguf")))),
]


def _configured() -> list[tuple[str, Path]]:
    """DEFAULT_MODELS, or ROUTING_SMOKE_MODELS as 'name=path,name=path'."""
    raw = os.environ.get("ROUTING_SMOKE_MODELS")
    if not raw:
        return DEFAULT_MODELS
    out = []
    for item in raw.split(","):
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"ROUTING_SMOKE_MODELS entry {item!r} is not name=path")
        out.append((name.strip(), Path(path.strip())))
    return out


_LIB_DIR = os.environ.get("LLAMA_CPP_LIB_PATH")
_OUT_DIR = Path(_LIB_DIR).parent if _LIB_DIR else None
_SERVER = _OUT_DIR / "bin" / "llama-server" if _OUT_DIR else None

pytestmark = pytest.mark.skipif(
    not (_SERVER and _SERVER.is_file()),
    reason="requires a routed build (LLAMA_CPP_LIB_PATH -> <out>/lib with <out>/bin/llama-server)",
)


@pytest.mark.parametrize(("name", "model"), _configured(), ids=[n for n, _ in _configured()])
def test_routing_serves_on_this_architecture(name: str, model: Path, tmp_path: Path) -> None:
    lora_dir = SMOKE_ROOT / name
    if not model.is_file():
        pytest.skip(f"model absent: {model}")
    missing = [p for p in ("last-ffn", "mid-ffn", "last-attn") if len(sorted(lora_dir.glob(f"{p}-*.gguf"))) != 2]
    if missing:
        pytest.skip(
            f"adapter sets {missing} absent under {lora_dir}. Synthesize them:\n"
            f"  ci/library/.venv/bin/python {SUITE_DIR}/make_arch_lora.py "
            f"--model {model} --out-dir {lora_dir}"
        )

    result = routing_smoke.run_gate(_SERVER, model, lora_dir, tmp_path)
    (tmp_path / "routing-smoke.json").write_text(json.dumps(result, indent=2))

    failed = [k for k, v in result["verdict"].items() if not v]
    per_arm = {a["arm"]: {k: a[k] for k in ("startup", "aborted", "serves")} for a in result["arms"]}
    detail = json.dumps(per_arm, indent=2)

    # The controls are reported first on failure. A control that stops serving
    # means the harness or the build is at fault, and reading it as "the arm
    # under test regressed" is the mistake on record.
    controls = [k for k in failed if k != "routed_serves_after_the_reduction"]
    assert not controls, (
        f"{name} ({result['arch']}): CONTROL arms failed {controls} — suspect the build or this harness, "
        f"not the row-reduction path\n{detail}"
    )
    assert not failed, f"{name} ({result['arch']}): routing does not serve after the row reduction\n{detail}"
