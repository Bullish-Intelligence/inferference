# Mode A — source rebuild of llama-cpp-python against the fork (what ships)

`release-` tags imply Mode A provenance: llama-cpp-python rebuilt from source
against `llama-infernal@integration`, verified WITHOUT a lib swap (the wheel's
own bundled libs). Mode B (`LLAMA_CPP_LIB_PATH` lib swap) is iteration-only,
same-ABI experiments; a Mode-B-only artifact is never released.

## Mechanism: pinned clone, not a git submodule

llama-cpp-python is **not forked**; stock `v0.3.34` enters as a vendored
checkout whose nested `vendor/llama.cpp` is re-pointed to
`llama-infernal@integration`, so the source build compiles OUR integration lib
into the wheel.

Why a pinned clone instead of a submodule:

1. **gitman canonicity** — inferference's VC manager does not model nested
   submodule repos; a `.gitmodules`/gitlink change risks reconcile churn.
2. **Determinism** — a submodule re-point is not reproducible: the parent pins
   only the submodule HEAD, not its `.gitmodules`, so a fresh recursive clone
   restores the stock `ggml-org/llama.cpp` re-point. A materialized pinned
   clone is deterministic by construction.
3. **Repo bloat** — the full vendored sources (~100 MB) are never committed;
   `vendor/` is gitignored and materialized by `ci/build/mode-a-prep.sh`.

## Recipes

```bash
# 1. materialize vendor/llama-cpp-python (v0.3.34) + vendor/llama.cpp
#    (llama-infernal@integration) as plain trees; emits the build record
#    ci/build/vendor-manifest.json (exact commits).
ci/build/mode-a-prep.sh

# 2. source rebuild + install the wheel into ci/modea/.venv (CPython 3.13 —
#    llama-cpp-python 0.3.34 has no cp314 wheel). Emits the version tuple
#    ci/build/modea-version-tuple.json.
NIXPKGS_ALLOW_UNFREE=1 devenv shell -- ci/build/mode-a-build.sh
```

Prep removes `.agents` trees and Copier answer files from the nested source
tree. These files manage source work. They do not affect the Python or C++ build.
Prep records a hash of the full source tree. The build refuses a tree that
changed after prep.

Set `MODEA_PYTHON` to select a Python 3.13 interpreter. Use one that shares
the build toolchain's C library. The build refuses an interpreter that cannot
import the bundled libraries.

The build recipe: `CMAKE_ARGS=
"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 uv pip install
--no-binary llama-cpp-python <vendor path>`. The CUDA toolkit, gcc, cmake,
ninja, ccache are devenv packages.

## Verification (Q10) — the wheel's own bundled libs, no swap

The wheel bundles the lib set at `<venv>/lib/python3.13/site-packages/llama_cpp/lib`.
"Without `LLAMA_CPP_LIB_PATH`" means: point the gates at THAT lib dir (the
wheel's own pair), never at a separate Mode-B build dir:

```bash
WHEEL_LIB="ci/modea/.venv/lib/python3.13/site-packages/llama_cpp/lib"
export LLAMA_CPP_LIB_PATH="$WHEEL_LIB"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LLAMA_CPP_LIB_PATH:$(dirname $(gcc -print-file-name=libstdc++.so)):$(dirname $(gcc -print-file-name=libgomp.so))"
export CUDA_VISIBLE_DEVICES=0
ci/modea/.venv/bin/python ci/gates/abi_smoke_gate.py --model <any GGUF>           # exit 0
ci/gates/.venv/bin/python ci/gates/cffi_bindgen.py \
  --lib-dir "$WHEEL_LIB" \
  --include-dir <integration build out>/include --profile p2fork --check           # exit 0
```

The wheel does not ship headers, so the bindgen pairs the wheel's lib with the
integration build's include/ (the exact tree that produced the lib). The
version tuple (`llama-cpp-python ref/commit`, anchor, built commit, build
flags, ggml version) is recorded in `ci/build/modea-version-tuple.json` +
`ci/build/vendor-manifest.json`; the `release-` tag names the anchor +
featureset (e.g. `release-b10103-p2`).

## Mode B does not apply to an executable — it cannot

`LLAMA_CPP_LIB_PATH` is read **only by the Python package**, to `dlopen`
`libllama.so` out of a directory (`llama_cpp/llama_cpp.py:36-43`;
`_ctypes_extensions.py:37-40, :102`). It swaps a **shared library**. There is no
mechanism by which it swaps a **binary**.

So `llama-server` has no Mode B. **Every server artefact is Mode A provenance by
construction** — a source rebuild of the fork tree, or nothing. The rule that a
`release-` tag implies Mode A is therefore not a constraint on the server; it is
a description of the only thing the server can be.

(This paragraph belongs here rather than in a separate build-workflow document:
the wheel's provenance has to be recorded next to the recipe that produces it.)

## The NixOS module and the YAML config

```nix
imports = [ /path/to/inferference/nix/nixos-module.nix ];

services.inferference-server = {
  enable  = true;
  user    = "youruser";
  repoDir = "/path/to/inferference";
  roles   = [ "<role>" ];          # ONE role — see AGENTS.md, GPU discipline
};
```

**The module holds no model path, no port, no adapter list and no card.** Those
live in `~/.config/inferference/<role>.yaml` and are read by the unit **at
start**, so changing a model or a pool is an edit plus `systemctl --user restart`
and never a NixOS rebuild. Schema and commentary:
`ci/runner/server-role.example.yaml`.

The card is not in either place: each role resolves its own from that role's
variable (`$LLAMA_GPU_<ROLE>`), so no file holds a literal
index.

`ci/runner/inferference-config.py` turns the YAML into an argv and **fails closed
with EX_CONFIG (78)** on anything it cannot verify: a missing config, an
unreadable model, a **missing adapter**, a bad type, or an unknown key. An
unknown key is refused rather than ignored — a typo silently dropped leaves a
server running a configuration nobody wrote. systemd reports 78 distinctly, so a
misconfiguration never looks like a crashed server.

`routing:` defaults to **false**, which is a measured decision: per-sequence
routing wins 1.6x-2.0x on adapter-mixed traffic and *loses* 0.96x / 0.91x / 0.89x
at 8 / 16 / 32 slots when every request names one adapter. Setting it true also
forces `--cache-ram 0` (R-4), so the operator does not have to know that.

Listing two roles emits a build-time **warning**: the rig policy is one always-on
server on one card, and preemption is not implemented.

## The server artefact

```bash
# build the routed server from a gated commit …
ci/build/build-llamacpp.sh --ref integration --profile p2fork-server
# … then publish it out of the prunable build garage
ci/build/mode-a-server.sh --out ci/build/.llamacpp-builds/out-p2fork-server-<sha>
```

Layout, with `current` pointing at the newest published build:

```
ci/modea/server/<build_id>/
  bin/llama-server
  lib/                       its own lib set: the build is -DBUILD_SHARED_LIBS=ON
  include/
  build-manifest.json        the cross-repo contract, copied byte-for-byte
  modea-provenance.json      what is true of the PUBLICATION, not the build
ci/modea/server/current -> <build_id>
```

**Why publish at all.** `ci/build/.llamacpp-builds/` is gitignored, unbounded,
and pruned by `ci/build/prune-builds.sh`, which keeps only the newest `out-*` per
profile. A binary a consumer runs must not live where a housekeeping script
deletes it. `prune-builds.sh` does not touch `ci/modea/`.

`mode-a-server.sh` refuses, rather than publishes, when: the manifest is missing;
`server_built`, `seq_adapter_routing` or `server_seq_routing` is false;
`ref_built` carries `-dirty`; `ldd` under a cleared `LD_LIBRARY_PATH` finds an
unresolved library **or one resolving back into the build garage**; or the
binary does not answer `--version`. Every one of those runs **after** the copy,
from the published location — a relocation that held in the garage is not
evidence that it holds here.

`--release` additionally requires the wheel and the server to name **one**
commit. Without it the two may differ, and the difference is recorded in
`modea-provenance.json` as `wheel_and_server_same_commit: false` rather than
left implicit.

## The manifest gap, closed

The manifest does not survive the Mode A copy on its own: `find ci/modea -name
build-manifest.json` returns nothing. `build-llamacpp.sh` writes the manifest
beside a lib set **it** built; the wheel is a second path to the same libraries
and had no emitter.

`ci/build/mode-a-manifest.sh` is that emitter. It writes to
`<site-packages>/llama_cpp/build-manifest.json` — the **parent** of the wheel's
`lib/`, which is where `diagnostics.py`'s `_build_manifest` looks first
("How the contract is read" in `build-manifest.md`).

It **fails closed on every key it cannot derive**, and never guesses. The sharp
one is `features`: `ci/runner/run-suite.sh:66` reads `stock = not
manifest.get("features")`, and a `stock` build downgrades a **failed capability
probe from FAIL to SKIP** (`run-suite.sh:92`). An empty `features` on a fork
build therefore converts a lost capability into a silent pass, so the emitter
refuses to write rather than record `[]`.
