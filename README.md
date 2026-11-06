# Inferference

**A personal, multi-model inference platform: a llama.cpp build you control,
reachable from Python at the level of its internals.**

Most inference libraries hand you `generate()` and hide the machinery. This one
exists to open the machinery: the decode loop, batching, KV and prefix cache,
grammar-constrained sampling, LoRA adapter routing, and per-sequence state are
all things you can drive, intercept and measure directly. It is built for asking
questions about how inference behaves and answering them by experiment.

It is **model-agnostic**. Any GGUF is a candidate. Which models a given machine
holds is inventory, not a property of the platform.

## What you get

- **An owned decode loop** (`decode.py`) built on `llama_decode` and the sampler
  C API — not on a wrapper's `generate()`.
- **Composable decode middleware** (`middleware.py`) — treat inference as a
  workflow you can add to and intercept.
- **Our own continuous/dynamic batching** (`batching.py`), independent of the
  llama.cpp server's.
- **Prefix-cache and KV-state tools** (`prefix_cache.py`, `node_delta*.py`) —
  including replaying a context tree over live KV state with zero prefill.
- **Grammar-constrained decoding** (`grammar.py`) that keeps torch out of the
  hot path.
- **Per-sequence LoRA adapter routing** (`seq_routing.py`, `router.py`) — one
  decode step serving sequences that name *different* adapters.
- **A measurement harness** (`src/inferference/benchkit/`, `bench/`) and capability
  gates (`suites/`).
- **A reproducible build pipeline** with an ABI tripwire, so the low-level
  bindings can never drift from the library they call.

## Install

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/). Every install goes
through uv; there are no private dependencies.

As a dependency of another project:

```bash
uv add "inferference @ git+https://github.com/Bullish-Intelligence/inferference"
uv add "inferference[runtime] @ git+https://github.com/Bullish-Intelligence/inferference"
```

From a clone of this repository:

```bash
uv venv
uv pip install .              # core: contract, models, placement/config readers
uv pip install '.[runtime]'   # + the decode path (llama-cpp-python, xgrammar, numpy)
```

**Read this before installing `[runtime]`.** It pulls stock `llama-cpp-python`,
which links the `llama.cpp` it was built against. The low-level bindings here
are anchored to one `llama.cpp` commit, so set `LLAMA_CPP_LIB_PATH` to a library
set built from that anchor:

```bash
export LLAMA_CPP_LIB_PATH=/path/to/your/llama.cpp/lib
```

A stock binding against a different `llama.cpp` is silent memory corruption,
not an error. There is no `llama-cpp-python` wheel on PyPI for 0.3.34, so the
first install compiles `llama.cpp` and takes several minutes.

## Development environment

`devenv.nix` pins the toolchain for this repository. `uv sync` builds the
environment from `uv.lock`; `--all-extras` adds the runtime extra and the
dev group (pytest, ruff, ty).

```bash
devenv shell
uv sync --all-extras
```

## Verify

Every command runs through the uv-managed environment, so `uv sync --all-extras`
must have run once:

```bash
uv run ruff check . && uv run ruff format --check .  # lint + format
uv run ty check .                                    # types
uv run python -m pytest -m "not runtime"             # the required gate: no GPU, no native stack
uv run python -m pytest                              # with the runtime extra: everything
```

Edit the checks in the `tasks` table of `devenv.nix`. `uv run ruff format src tests`
is the formatter; there is no other fix command. The maintainers run exactly
these commands.

## Version control

The maintainers use a colocated `jj` repository. A plain `git` clone works
normally: branch, commit, open a pull request.

## Getting started

**1. Declare your machine's models.** Roles are per-machine inventory and live
in an untracked file:

```bash
cp ci/runner/rig-local.example.sh ci/runner/rig-local.sh
$EDITOR ci/runner/rig-local.sh
```

Each role names a model and the card it serves:

```bash
inferference_role small 0 "${INFERFERENCE_MODELS_DIR}/some-3B-Q4_K_M.gguf"
```

That exports `MODEL_SMALL` and `LLAMA_GPU_SMALL` — the variables every server
unit, gate and suite resolves a role through. A role whose GGUF is missing is
not declared at all, so a consumer fails loudly rather than serving a model
nobody chose.

**2. Build a llama.cpp library set** for your hardware:

```bash
# NVIDIA (compute capability 8.6)
ci/build/build-llamacpp.sh --ref <git-ref> --profile cuda-3060-server

# AMD Vega / gfx900 via Vulkan
ci/build/build-llamacpp.sh --ref <git-ref> --profile vulkan-mi25-server

# CPU only
ci/build/build-llamacpp.sh --ref <git-ref> --profile cpu-light
```

**3. Point Python at it** (Mode B — a library swap, for same-ABI iteration):

```bash
export LLAMA_CPP_LIB_PATH=<out-dir>/lib
```

Mode A is a source rebuild of llama-cpp-python that bundles the library set;
`release-` tags imply Mode A provenance. See `ci/build/mode-a.md`.

Code evaluation requires load-time residency evidence for local runs. It
currently verifies models placed on one card.

If the reconciler cannot verify placement, it requests an unload of every
declared model. Check the router if an unload request fails.

## Layout

```
src/inferference/          the library — decode loop, batching, cache, grammar, routing
src/inferference/benchkit/ the measurement harness (shipped with the wheel)
bench/                     workload corpora
suites/                    capability gates: does this build carry this capability?
ci/build/                  llama.cpp build pipeline and Mode A packaging
ci/gates/                  tripwires: ABI drift, GPU lease, backend smoke
ci/runner/                 rig environment, GPU lease, router launch, suites executor
tasks/codeeval/            the coding-evaluation corpus (data, not code)
tests/                     the library's test suite
```

## The two rules worth knowing up front

**The ABI anchor.** llama-cpp-python's hand-written ctypes match exactly one
llama.cpp `llama.h`. A struct change is silent memory corruption, so the cffi
bindgen gate compiles against the built headers on every build: **compile
failure means ABI drift, and that is a STOP, never a TODO.**

**Fail-closed everywhere.** A missing capability is a skip or a loud failure,
never a silent pass. Backends in particular fall back to CPU cheerfully — a gate
that only checks for a successful answer will pass on a machine with no GPU at
all, so gates here prove device placement independently.

## Contributing / conventions

Run the four commands under "Verify" before you open a pull request, and keep
them all green — a skipped capability is a skip or a loud failure, never a
silent pass. `AGENTS.md` states the two rules the code depends on: the ABI
anchor and fail-closed behaviour.

## License

MIT. See `LICENSE`.

The low-level binding is stock `llama-cpp-python`, and the `llama.cpp` library
set is expected to come from your own build. Portions of the decode, batching
and routing layer were ported from an earlier MIT-licensed internal codebase.

## Author

Bullish-Intelligence <242310653+Bullish-Intelligence@users.noreply.github.com>
