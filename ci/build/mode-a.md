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
NIXPKGS_ALLOW_UNFREE=1 MODEA_BACKEND=<cuda|vulkan> devenv shell -- ci/build/mode-a-build.sh
```

Prep removes `.agents` trees and Copier answer files from the nested source
tree. These files manage source work. They do not affect the Python or C++ build.
Prep records a hash of the full source tree. The build refuses a tree that
changed after prep.

Set `MODEA_PYTHON` to select a Python 3.13 interpreter. Use one that shares
the build toolchain's C library. The build refuses an interpreter that cannot
import the bundled libraries.

The build recipe: `CMAKE_ARGS="<backend flags>" FORCE_CMAKE=1 uv pip install
--no-binary llama-cpp-python <vendor path>`. The backend flags name the backend
of the machine that serves the wheel. For a Vulkan build use `-DGGML_VULKAN=ON`.
For a CUDA build use `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>`.
Set `MODEA_BACKEND=vulkan` (or `cuda`), or set `CMAKE_ARGS` yourself. Without
either, `mode-a-build.sh` builds for CUDA. The build checks that the bundled libs
match the flags. The backend SDK (the Vulkan headers and shader compiler, or
the CUDA toolkit), gcc, cmake, ninja and ccache must be on the build path.

## Verification — the wheel's own bundled libs, no swap

The wheel bundles the lib set at `<venv>/lib/python3.13/site-packages/llama_cpp/lib`.
"Without `LLAMA_CPP_LIB_PATH`" means: point the gates at THAT lib dir (the
wheel's own pair), never at a separate Mode-B build dir.

The reference host serves on Vulkan. The commands below are the Vulkan release path.
The older CUDA commands are at the end of this section as history.

### Vulkan release path

Run every command from the repository root, inside the devenv. `<server out>`
is the raw build dir of the server (`ci/build/.llamacpp-builds/out-<profile>-<ref>-residency-<patch12>`).
`<card>` is the card index that the lease grants. Models come from the machine
inventory: `ci/runner/rig-env.sh` exports `MODEL_DEFAULT`. Do not type a model name.

1. Prepare and build the wheel. No GPU. It records the patch hash in
   `ci/build/vendor-manifest.json`, and the real flags in `ci/build/modea-version-tuple.json`.
   Both files are tracked and generated. Never edit them by hand.

   ```bash
   bash ci/build/mode-a-prep.sh
   MODEA_BACKEND=vulkan MODEA_SKIP_SERVER=1 MODEA_PYTHON=<python 3.13> bash ci/build/mode-a-build.sh
   grep build_flags ci/build/modea-version-tuple.json     # must say GGML_VULKAN=ON and GGML_CUDA=OFF
   ```

   `MODEA_PYTHON` must run on the glibc of the build toolchain. The libs need
   the toolchain glibc, and an older interpreter cannot load them ("version
   `GLIBC_2.43' not found"). Compare the glibc of the interpreter and of the
   toolchain before the build:
   `ldd "$MODEA_PYTHON" | grep libc.so` and `ldd "$(gcc -print-file-name=libstdc++.so)" | grep libc.so`.
   The same rule holds for the library venv (`LIBRARY_BASE_PYTHON`) that runs the suites.

   `MODEA_SKIP_SERVER=1` builds only the wheel. Build the server with
   `ci/build/build-llamacpp.sh --ref <fork commit> --profile vulkan-mi25-p2fork-server`.
   To build the fork library alone (no wheel, no server), use
   `ci/build/build-llamacpp.sh --ref <fork commit> --profile vulkan-p2fork`.
   It applies the same residency patch. Its out dir feeds `ci/runner/run-suite.sh --out`
   and the ABI gate (`--profile vulkan-p2fork`).

2. ABI gate. No GPU. It compiles against the headers of `<server out>`.
   A compile failure is ABI drift. Stop (rule 2). Never edit the cdef.

   ```bash
   WHEEL_LIB="$PWD/ci/modea/.venv/lib/python3.13/site-packages/llama_cpp/lib"
   ci/gates/.venv/bin/python ci/gates/cffi_bindgen.py \
     --lib-dir "$WHEEL_LIB" --include-dir <server out>/include \
     --profile vulkan-mi25-p2fork-server --out-dir <scratch dir> --check        # exit 0
   ```

3. ABI smoke on the wheel's libs. GPU. The lease pins one card. The gate then
   reads that card from `LLAMA_CI_GPU`.

   ```bash
   source ci/runner/rig-env.sh
   export LLAMA_CPP_LIB_PATH="$WHEEL_LIB"
   export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LLAMA_CPP_LIB_PATH:$(dirname $(gcc -print-file-name=libstdc++.so)):$(dirname $(gcc -print-file-name=libgomp.so))"
   GPU_LEASE_CARDS=<card> GPU_LEASE_TIMEOUT=10 ci/runner/gpu-lease.sh <MiB> -- \
     ci/modea/.venv/bin/python ci/gates/abi_smoke_gate.py --model "$MODEL_DEFAULT"   # exit 0
   ```

4. Smoke gate on the server build. GPU. Run it once per card. It proves the PCI
   address, the layer assignment, the VRAM rise and a flat GTT.

   ```bash
   GPU_LEASE_CARDS=<card> GPU_LEASE_TIMEOUT=10 ci/runner/gpu-lease.sh <MiB> -- \
     ci/gates/vulkan_mi25_smoke.sh --out <server out> --model "$MODEL_DEFAULT"      # exit 0
   ```

5. Suites. GPU. `ci/runner/run-suite.sh --suite <name> --out <server out>`, under the
   lease. A skip is not a pass. Count every skip.

6. Publish without repointing, then, after review, with `current`.
   Run the script from the lane. Pass `--dest-root` so the build lands in the main
   checkout, because a lane workspace is deleted by `gitman done`.

   ```bash
   ci/build/mode-a-server.sh --out <server out> --no-current --release --dest-root <main checkout>/ci/modea/server
   ci/build/mode-a-server.sh --out <server out> --current   --release --dest-root <main checkout>/ci/modea/server
   ```

   The script refuses to replace the build that `current` names, and a build that
   one of your processes maps or runs. A process of another user is not visible
   to it. Check the router before you publish.
   `modea-provenance.json` records the patch hash and a sha256 of every file
   in `bin/` and `lib/`. A reader then proves the artefact with `sha256sum`.

The wheel does not ship headers, so the bindgen pairs the wheel's lib with the
server build's `include/` (the exact tree that produced the lib). The
version tuple (`llama-cpp-python ref/commit`, anchor, built commit, build
flags, ggml version) is recorded in `ci/build/modea-version-tuple.json` +
`ci/build/vendor-manifest.json`; the `release-` tag names the anchor +
featureset (e.g. `release-b10103-p2`).

### CUDA commands (history)

These ran on the earlier CUDA rig. Keep them for a CUDA machine only.

```bash
WHEEL_LIB="ci/modea/.venv/lib/python3.13/site-packages/llama_cpp/lib"
export LLAMA_CPP_LIB_PATH="$WHEEL_LIB"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LLAMA_CPP_LIB_PATH:$(dirname $(gcc -print-file-name=libstdc++.so)):$(dirname $(gcc -print-file-name=libgomp.so))"
export LLAMA_CI_GPU=0   # the gate pins every backend's device variable from this one
ci/modea/.venv/bin/python ci/gates/abi_smoke_gate.py --model <any GGUF>           # exit 0
ci/gates/.venv/bin/python ci/gates/cffi_bindgen.py \
  --lib-dir "$WHEEL_LIB" \
  --include-dir <integration build out>/include --profile p2fork --check           # exit 0
```

For CUDA, build the wheel with `MODEA_BACKEND=cuda` (the default).

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
live in `<config dir>/<role>.yaml` and are read by the unit **at
start**. The launcher picks the config dir like `inferference-reconcile.py
config-dir`: `$INFERFERENCE_CONFIG_DIR`, then `/etc/inferference` when it exists,
then `~/.config/inferference` with a warning. So changing a model or a pool is an
edit plus `systemctl --user restart` and never a NixOS rebuild. Schema and commentary:
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
ci/build/build-llamacpp.sh --ref integration --profile vulkan-mi25-p2fork-server
# … then publish it out of the prunable build garage
ci/build/mode-a-server.sh --out ci/build/.llamacpp-builds/out-vulkan-mi25-p2fork-server-<sha>[-residency-<hash>]
```

On a CUDA host, use `--profile p2fork-server` and the out dir `out-p2fork-server-<sha>`.

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
