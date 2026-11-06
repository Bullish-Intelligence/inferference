"""Turn LSP output into a codebase context tree and cross-chain pull-ins.

The context-tree engine (``node_delta_live``) consumes two things: a containment
tree of :class:`~node_delta_live.TreeNodeSpec` (repo -> dir -> file -> symbol,
each contributing its own token span) and, per query, a set of
:class:`~node_blend_live.PullInSpec` for the definitions/types a symbol
references.  Both map directly onto standard LSP requests:

  * ``textDocument/documentSymbol`` -> the per-file symbol hierarchy;
  * the filesystem path -> the repo/dir/file containment above each file;
  * ``textDocument/definition`` / ``typeDefinition`` / ``references`` -> the
    cross-edges that become pull-ins.

This module is pure: it takes plain dataclasses (already extracted from whatever
LSP client the caller uses) plus a ``tokenize`` callable, and performs no I/O and
no inference.  Node ids are stable, path-derived strings so an edge discovered in
one request resolves to the same node captured in the tree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .node_blend_live import PullInSpec
from .node_delta_live import NodeInclusion, TreeNodeSpec

Tokenize = Callable[[str], Sequence[int]]

# Kinds of cross-reference an LSP client can surface, in default pull-in priority.
DEFINITION = "definition"
TYPE_DEFINITION = "type_definition"
REFERENCES = "references"
_EDGE_PRIORITY = {DEFINITION: 0, TYPE_DEFINITION: 1, REFERENCES: 2}


@dataclass(frozen=True, slots=True)
class LspSymbol:
    """One ``documentSymbol`` node: the tokens it adds, minus its children's."""

    name: str
    kind: str
    body_text: str
    children: tuple[LspSymbol, ...] = ()


@dataclass(frozen=True, slots=True)
class LspFile:
    """A source file: repo-relative POSIX path, a preamble, and its symbols.

    ``preamble_text`` is whatever precedes the first symbol (module docstring,
    imports) -- the file-level context every symbol in the file depends on.
    """

    path: str
    preamble_text: str
    symbols: tuple[LspSymbol, ...] = ()


@dataclass(frozen=True, slots=True)
class CrossEdge:
    """A resolved LSP reference: ``from_node_id`` uses the def at ``to_node_id``."""

    from_node_id: str
    to_node_id: str
    kind: str = DEFINITION


# --------------------------------------------------------------------------- #
# Stable node-id scheme (path-derived so tree capture and edge lookup agree)
# --------------------------------------------------------------------------- #


def repo_node_id(root_name: str) -> str:
    return f"repo:{root_name}"


def dir_node_id(dir_path: str) -> str:
    return f"dir:{dir_path}"


def file_node_id(file_path: str) -> str:
    return f"file:{file_path}"


def symbol_node_id(file_path: str, dotted_name: str) -> str:
    return f"sym:{file_path}::{dotted_name}"


def _dir_components(path: str) -> list[str]:
    """POSIX-split a repo-relative path into its ancestor directory prefixes."""
    parts = [p for p in path.split("/") if p]
    prefixes: list[str] = []
    acc: list[str] = []
    for part in parts[:-1]:  # exclude the file component itself
        acc.append(part)
        prefixes.append("/".join(acc))
    return prefixes


@dataclass
class _Emit:
    """Mutable accumulator so a directory node is emitted at most once."""

    specs: list[TreeNodeSpec] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)

    def add(self, spec: TreeNodeSpec) -> None:
        if spec.node_id in self.seen:
            return
        self.seen.add(spec.node_id)
        self.specs.append(spec)


def build_tree(
    files: Sequence[LspFile],
    tokenize: Tokenize,
    *,
    root_name: str = "repo",
    include_file: Callable[[LspFile], bool] = lambda _f: True,
    summarize_file: Callable[[LspFile], str | None] = lambda _f: None,
) -> tuple[TreeNodeSpec, ...]:
    """Build the containment tree specs for ``files`` in parent-before-child order.

    Every structural node contributes a minimal marker span (its path segment) so
    no chain is empty and directory context is itself part of the prompt.  Policy
    hooks mirror the user's "filter some files, summarize others" requirement:

      * ``include_file`` returning ``False`` drops the file and its symbols
        entirely (no node, so nothing downstream can pull it in);
      * ``summarize_file`` returning a string replaces the file's raw preamble
        and symbol bodies with that one summary span (``NodeInclusion.SUMMARIZE``),
        so the file still anchors go-to-def but costs only the digest's tokens.
    """
    emit = _Emit()
    root_id = repo_node_id(root_name)
    emit.add(TreeNodeSpec(root_id, None, tuple(tokenize(root_name)), source_path=""))

    for file in files:
        if not include_file(file):
            continue

        # Synthesize the directory chain repo -> a -> a/b for this file's path.
        parent_id = root_id
        for prefix in _dir_components(file.path):
            node_id = dir_node_id(prefix)
            segment = prefix.rsplit("/", 1)[-1]
            emit.add(TreeNodeSpec(node_id, parent_id, tuple(tokenize(segment)), source_path=prefix))
            parent_id = node_id

        f_id = file_node_id(file.path)
        summary = summarize_file(file)
        if summary is not None:
            emit.add(
                TreeNodeSpec(
                    f_id,
                    parent_id,
                    tuple(tokenize(file.preamble_text)),
                    inclusion=NodeInclusion.SUMMARIZE,
                    summary_token_ids=tuple(tokenize(summary)),
                    source_path=file.path,
                )
            )
            continue  # a summarized file exposes no per-symbol nodes

        emit.add(TreeNodeSpec(f_id, parent_id, tuple(tokenize(file.preamble_text)), source_path=file.path))
        _emit_symbols(emit, tokenize, file.path, f_id, prefix="", symbols=file.symbols)

    return tuple(emit.specs)


def _emit_symbols(
    emit: _Emit,
    tokenize: Tokenize,
    file_path: str,
    parent_id: str,
    *,
    prefix: str,
    symbols: Sequence[LspSymbol],
) -> None:
    for sym in symbols:
        dotted = f"{prefix}{sym.name}"
        node_id = symbol_node_id(file_path, dotted)
        emit.add(TreeNodeSpec(node_id, parent_id, tuple(tokenize(sym.body_text)), source_path=file_path))
        if sym.children:
            _emit_symbols(emit, tokenize, file_path, node_id, prefix=f"{dotted}.", symbols=sym.children)


def pull_ins_for(
    from_node_id: str,
    edges: Sequence[CrossEdge],
    *,
    ancestors: Sequence[str] = (),
) -> tuple[PullInSpec, ...]:
    """Select cross-chain pull-ins for a query at ``from_node_id``.

    Filters ``edges`` to those originating at the node, drops any whose target is
    the node itself or already on its ancestor chain (that context is free via
    reconstruct_chain), de-duplicates targets keeping the highest-priority edge
    kind (definition > type > references), and returns them in that priority then
    discovery order -- the exact order ``blend_by_redecode`` admits under budget.
    """
    ancestor_set = {from_node_id, *ancestors}
    best: dict[str, tuple[int, int]] = {}  # to_node_id -> (priority, discovery_index)
    for index, edge in enumerate(edges):
        if edge.from_node_id != from_node_id:
            continue
        if edge.to_node_id in ancestor_set:
            continue
        priority = _EDGE_PRIORITY.get(edge.kind, len(_EDGE_PRIORITY))
        current = best.get(edge.to_node_id)
        if current is None or priority < current[0]:
            best[edge.to_node_id] = (priority, index)

    ordered = sorted(best.items(), key=lambda item: (item[1][0], item[1][1]))
    return tuple(PullInSpec(node_id) for node_id, _ in ordered)
