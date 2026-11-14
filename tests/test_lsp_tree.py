"""Tests for the LSP -> context-tree adapter (lsp_tree).

Pure: a deterministic word tokenizer stands in for the model tokenizer.  The
final test wires adapter output through ``capture_tree`` + ``pull_ins_for`` +
``blend_by_redecode`` to prove the path-derived node ids line up end to end.
"""

from __future__ import annotations

from pathlib import Path

from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint
from inferference.gpu_env import GPU_BACKEND_LABEL
from inferference.lsp_tree import (
    DEFINITION,
    REFERENCES,
    TYPE_DEFINITION,
    CrossEdge,
    LspFile,
    LspSymbol,
    build_tree,
    file_node_id,
    pull_ins_for,
    symbol_node_id,
)
from inferference.node_blend_live import PullInSpec, blend_by_redecode
from inferference.node_delta import NodeDeltaIndex, NodeInclusion
from inferference.node_delta_live import capture_tree, reconstruct_chain
from inferference.prefix_cache_live import InMemoryPrefixCache

_VOCAB: dict[str, int] = {}


def _tok(text: str) -> list[int]:
    """Deterministic word tokenizer: stable non-zero id per distinct word."""
    out = []
    for word in text.split():
        out.append(_VOCAB.setdefault(word, len(_VOCAB) + 1))
    return out


def _fingerprint() -> LlamaEngineFingerprint:
    a = ArtifactIdentity(path="/m.gguf", sha256="a" * 64, size_bytes=1, mtime_ns=1, inode=1)
    return LlamaEngineFingerprint(
        model=a,
        tokenizer=a,
        llama_cpp_python_version="0.3.34",
        llama_cpp_commit="b1",
        backend=GPU_BACKEND_LABEL,
        n_ctx=2048,
    )


def _sample_files() -> list[LspFile]:
    return [
        LspFile(
            path="src/models.py",
            preamble_text="module models imports",
            symbols=(
                LspSymbol(
                    "User", "class", "class User header", children=(LspSymbol("save", "method", "def save body"),)
                ),
            ),
        ),
        LspFile(
            path="src/api.py",
            preamble_text="module api imports",
            symbols=(LspSymbol("handler", "function", "def handler body"),),
        ),
        LspFile(path="README.md", preamble_text="readme prose here", symbols=()),
    ]


def test_build_tree_synthesizes_containment_with_stable_ids() -> None:
    specs = build_tree(_sample_files(), _tok, root_name="myrepo")
    by_id = {s.node_id: s for s in specs}

    # repo -> dir:src -> file:src/models.py -> sym User -> sym User.save
    assert "repo:myrepo" in by_id
    assert by_id["dir:src"].parent_node_id == "repo:myrepo"
    assert by_id[file_node_id("src/models.py")].parent_node_id == "dir:src"
    user = by_id[symbol_node_id("src/models.py", "User")]
    assert user.parent_node_id == file_node_id("src/models.py")
    assert by_id[symbol_node_id("src/models.py", "User.save")].parent_node_id == user.node_id
    # The shared "src" directory node is emitted exactly once across two files.
    assert sum(s.node_id == "dir:src" for s in specs) == 1
    # README lives directly under the repo root (no directory component).
    assert by_id[file_node_id("README.md")].parent_node_id == "repo:myrepo"


def test_include_file_hook_drops_file_and_symbols() -> None:
    specs = build_tree(_sample_files(), _tok, include_file=lambda f: not f.path.endswith(".md"))
    ids = {s.node_id for s in specs}
    assert file_node_id("README.md") not in ids
    assert file_node_id("src/api.py") in ids


def test_summarize_file_hook_replaces_body_with_digest() -> None:
    specs = build_tree(
        _sample_files(),
        _tok,
        summarize_file=lambda f: "short summary" if f.path == "src/models.py" else None,
    )
    by_id = {s.node_id: s for s in specs}
    models = by_id[file_node_id("src/models.py")]
    assert models.inclusion is NodeInclusion.SUMMARIZE
    assert models.summary_token_ids == tuple(_tok("short summary"))
    # A summarized file exposes no per-symbol nodes.
    assert symbol_node_id("src/models.py", "User") not in by_id


def test_pull_ins_dedup_by_priority_and_skip_ancestors() -> None:
    src = symbol_node_id("src/api.py", "handler")
    user = symbol_node_id("src/models.py", "User")
    save = symbol_node_id("src/models.py", "User.save")
    edges = [
        CrossEdge(src, user, REFERENCES),  # lower priority, dropped in favor of...
        CrossEdge(src, user, DEFINITION),  # ...this higher-priority edge to same target
        CrossEdge(src, save, TYPE_DEFINITION),
        CrossEdge(src, src, DEFINITION),  # self-edge, skipped
        CrossEdge(src, "dir:src", DEFINITION),  # on the ancestor chain, skipped
        CrossEdge("other", user, DEFINITION),  # different origin, ignored
    ]
    pull_ins = pull_ins_for(src, edges, ancestors=["dir:src", "repo:repo"])
    # definition(user) before type_definition(save); duplicates and ancestors gone.
    assert pull_ins == (PullInSpec(user), PullInSpec(save))


def test_adapter_output_flows_through_capture_and_blend(tmp_path: Path) -> None:
    fp = _fingerprint()
    specs = build_tree(_sample_files(), _tok, root_name="repo")
    bridge, cache, idx = _capture(specs, fp, tmp_path)

    handler = symbol_node_id("src/api.py", "handler")
    user = symbol_node_id("src/models.py", "User")

    # A plain reconstruction at the handler works with the adapter's ids.
    r = reconstruct_chain(
        _fresh(bridge),
        None,
        cache=cache,
        index=idx,
        node_id=handler,
        prompt_token_ids=_tok("what does this do"),
        n_seq_max=2,
        seq_id=1,
    )
    assert r.restored, getattr(r, "reason", r)

    # A go-to-def edge handler -> User becomes a pull-in and blends in cleanly.
    edges = [CrossEdge(handler, user, DEFINITION)]
    pulls = pull_ins_for(handler, edges)
    assert pulls == (PullInSpec(user),)
    blended = blend_by_redecode(
        _fresh(bridge),
        None,
        cache=cache,
        index=idx,
        base_node_id=handler,
        pull_ins=list(pulls),
        prompt_token_ids=_tok("explain"),
        n_seq_max=2,
        seq_id=1,
    )
    assert blended.ok
    assert blended.admitted == (user,)


def _capture(specs, fp, tmp_path):
    from tests.test_node_delta import FakeTreeBridge

    bridge, cache, idx = FakeTreeBridge(), InMemoryPrefixCache(), NodeDeltaIndex(tmp_path)
    capture_tree(
        bridge, None, cache=cache, index=idx, namespace="tree", fingerprint=fp, nodes=specs, n_seq_max=2, seq_id=0
    )
    return bridge, cache, idx


def _fresh(_bridge):
    from tests.test_node_delta import FakeTreeBridge

    return FakeTreeBridge()
