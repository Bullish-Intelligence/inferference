# AGENTS.md — Inferference

Working notes for anyone changing this repository, human or agent. `README.md`
says what the project is and how to install it; this file says what it is
allowed to be.

## What this project is

**A multi-model inference platform: a llama.cpp build you control, reachable
from Python at the level of its internals.** The point is fine-grained access
to the machinery under a normal inference call — batching, the decode loop,
prefix cache, grammar, adapter routing, per-sequence state — so that questions
about how inference behaves can be answered by experiment rather than by
reading someone else's abstraction.

**It is model-agnostic.** Any GGUF is a candidate. Which models a machine holds,
on which device, at which quant, is *inventory* — declared per machine, never
baked into the platform. If a source file, a default, a gate or a document here
starts to read as being *about* a particular model, that is a defect: fix it.

The same rule covers hardware. Device names, PCI addresses, VRAM totals, fan
channels and power caps belong to the machine that runs the code, not to the
code.

## The four rules

**1. Fail-closed.** A missing capability is a skip or a loud failure, never a
silent pass. Backends in particular fall back to CPU cheerfully, so a gate that
only checks for a successful answer will pass on a machine with no GPU at all.
Gates here prove device placement independently of whether the answer looks
right.

**2. The ABI anchor never moves silently.** llama-cpp-python 0.3.34's
hand-written ctypes match exactly one llama.cpp `llama.h` — the anchor
`0ab9d6fed` (build `b10233`). A struct change is silent memory corruption, not
an exception. The cffi bindgen gate compiles against the built headers on every
build: **compile failure means ABI drift, and that is a STOP, never a TODO.**
Bumping the anchor is the full pipeline, not an edit.

**3. No CPU fallback, ever.** A model is served from the VRAM of the device its
placement names, or it is not served. On AMD this has a second half: RADV will
cheerfully put the overflow in **GTT** (host RAM over PCIe) and report success —
the same layers assigned to the same device, a plausible buffer-size line, and a
throughput collapse nobody told you about. Nothing in llama.cpp's output
distinguishes resident from spilled. Prove placement the hard way: the device's
PCI address matches sysfs for the card the placement named, every layer is
assigned to that device, that card's VRAM rises, **and its GTT does not**. A log
line alone is not proof, and do not gate on `CPU_Mapped model buffer size` — it
is normal for non-offloaded embedding weights.

**4. Own the decode path.** This project owns its decode loop through
llama-cpp-python's low-level bindings. Do not propose delegating it to vLLM,
SGLang or another serving framework; the whole point is access to the machinery.

## Layout

```
src/inferference/        the library — decode loop, batching, cache, grammar,
                         routing, the upward contract, the experiment framework
src/inferference/benchkit/  the measurement harness (shipped with the wheel)
bench/                   workload corpora and the bench runners
suites/                  capability gates: does this build carry this capability?
ci/build/                llama.cpp build pipeline and Mode A packaging
ci/gates/                tripwires: ABI drift, GPU lease, backend smoke
ci/runner/               rig environment, GPU lease, router launch, suites executor
ci/workflows/            Actions-format workflows, run locally (see below)
tasks/codeeval/          the coding-evaluation corpus (data, not code)
tests/                   the library's test suite
```

`suites/` and experiments answer different questions, and the split is
load-bearing:

- **`suites/`** answers "does this build carry this capability", pass or fail. A
  `suite.json` declares a capability probe plus pytest tests, classified
  fail-closed by `ci/runner/run-suite.sh`. It names the model role it needs
  (`env.MODEL_ROLE`), never a model file. The machine inventory fills the role,
  and an unfilled role fails the suite by naming the variable. Other inventory
  variables (probe adapters, tokenizer) go in `env.MODEL_VARS`, and a GPU run
  where every GPU test skipped gives SKIP, never PASS. README.md lists the roles.
- **An experiment** answers "what does this measure", as a quantity with an
  uncertainty. A declarative spec names the fixed factors, the factor under
  test, the workloads and the replication, and is executed by
  `inferference.experiment`.

A measured quantity is never a suite, and a capability gate is never an
experiment. This repository ships the framework for both; it does not ship a
list of the maintainers' own experiment specs.

## Install, verify, integrate

Everything goes through `uv`. There is no bare `pip` step anywhere.

```bash
devenv shell                 # the pinned toolchain (devenv.nix is the only one)
uv sync --all-extras         # project + runtime extra + dev group, from uv.lock

uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv run python -m pytest -m "not runtime"   # the required gate: no GPU, no native stack
uv run python -m pytest                    # with the runtime extra: everything
```

`uv run ruff format src tests` is the formatter. `uv.lock` and `devenv.lock` are
tracked, and the gate is run from the lock: never commit a hand-edited lock.

The maintainers additionally run the same commands through a commit-scoped
verifier whose check definitions are the `tasks` table in `devenv.nix`. If you
change what the gate runs, change that table.

**No AI attribution in commit messages** — no `Co-Authored-By`, no tool
mention, ever.

## The two install modes

- **Mode B** (`LLAMA_CPP_LIB_PATH` pointing at a lib set) is iteration only, and
  only for lib sets with the same ABI.
- **Mode A** is a source rebuild of llama-cpp-python against the fork, producing
  a wheel that bundles our libs. `release-` tags imply Mode A provenance. Never
  ship a Mode-B-only artefact as a release.

A stock `llama-cpp-python` from PyPI is exactly the ABI-incompatible runtime the
anchor rule exists to prevent, which is why `[runtime]` alone is not a supported
serving configuration: it needs `LLAMA_CPP_LIB_PATH` pointing at an anchored
build.

## Serving: the router

One `llama-server` in router mode serves every model on the machine, spawning
one child process per model. The shape is:

- **The resident set is declared, not demand-loaded.** A per-model YAML says
  what a model is; a router-level YAML says which models are resident and on
  which device(s). A model may span several cards; a card may host several
  models. Nothing loads because a request arrived — a request for a model that
  is not resident is an ERROR, never a load.
- **YAML wins.** `/models/load` and `/models/unload` stay enabled and are
  deliberately unused: they are operational controls, not a source of truth. A
  reconcile from the config files overrides whatever is currently on the card,
  without warning.
- **The reconciler refuses to overcommit.** Before loading anything it checks
  the placement against each card's measured free VRAM. If a model would not
  fit, it does not load it and does not evict anything to make room: it fails,
  names the card, and lists what is resident there with each model's footprint,
  so a human decides what comes out.
- **A card can be evicted on demand.** Evicting unloads every model on that card
  and leaves the placement config untouched, so a later reconcile restores it.
  This is the supported way to free a card — not killing the router.
- **`--models-max` must be >= the declared resident count.** With it lower, an
  explicit load returns success and silently evicts the least recently used
  model.
- **Reconcile is three steps:** regenerate the preset file → `GET /models?reload=1`
  (unloads changed/removed presets, does not reload them) → `POST /models/load`
  for each model that should be resident. A crashed child is reported
  `unloaded` with `exit_code` set; a deliberately unloaded one has `exit_code`
  null. That difference is the health signal.
- **`systemctl reload` runs the reconcile.** `ExecReload` of the router unit is
  `ci/runner/inferference-router-reload.sh`. Both units that this repository
  ships carry it: the `systemd --user` unit that `nix/nixos-module.nix`
  generates, and the template `ci/runner/inferference-router.service.in`. A
  machine that defines its own system unit must add the same line. The script
  runs as the unit user, takes a `flock` so that two reloads cannot overlap,
  and `exec`s `inferference-reconcile.py apply`. It exits non-zero on any
  refusal or failed residency proof. A failed reload never stops the router.
  The unit user owns the preset file, so this is the way to refresh it after
  the config changes.
- **A reload has one deadline.** One `--timeout` (default 600 s in the reload
  script) covers every load wait and router call of the whole `apply`. After
  it, `apply` can use at most 240 s more, so a reload ends within 840 s. That
  is under `TimeoutStartSec=900`, which systemd also applies to `ExecReload`.
  The script refuses a deadline that does not fit the unit timeout.
- **A reload that changes nothing touches nothing.** With no model to load or
  unload and a preset that matches, `apply` exits zero at once: no write, no
  verify, no unload. A reload that fails after a change unloads only the models
  that it changed. Models that it did not change stay resident.
- **The GPU lease governs build and gate jobs, not serving.** A lease that finds
  no room waits a bounded time and then refuses, naming the card that is short.
  Free a card explicitly and re-run; do not read a wait as a queue.

A model is served from the VRAM of the card its placement names, or it is not
served. See rule 3.

## Measure, never derive

The numbers that matter here are machine-specific and counter-intuitive, which
is why they are measured rather than computed:

- **A placement's footprint is `model_buf + KV + recurrent_state + compute`, and
  KV behaviour is per-architecture.** Two models on the same card can behave in
  opposite ways: a sliding-window model's KV flattens above its window, so long
  context is nearly free and only lanes cost; a dense model pays linearly for
  both. KV does not track parameter count. `ctx-size` is the total cache, so
  raising `--parallel` at fixed `ctx-size` silently shrinks every lane. Budget
  the requested context, and measure the model's own (ctx, parallel) curve.
- **llama.cpp will not refuse an overcommit.** It reports success. That is why
  the reconciler exists and why residency is proven from sysfs.
- **Speculative decoding is a per-model A/B, never a platform default.** The
  best mode inverts with model size, and a verify batch is wider than a decode
  step, so it also changes the footprint — re-measure before you trust a
  placement.
- **Adapter routing is not free on homogeneous traffic.** Per-token masks cost
  more as the batch widens. Turn it off unless the traffic is genuinely
  adapter-mixed.

Because footprint is a quantity with an error bar, a placement is a claim that
needs evidence: run it, prove residency, and record the number.

## CI

The workflows in `ci/workflows/` are GitHub-Actions format but are written to be
**run locally** by `act` (`.actrc` maps every runner label to the host) — there
is no `.github/` directory and no hosted CI. They name `self-hosted` labels
because the jobs need the real GPU and the model cache. **Do not move or
symlink them into `.github/workflows/`**: on a public repository that would
dispatch fork pull requests onto self-hosted runners.

## License

MIT — see `LICENSE`. Portions of the decode, batching and routing layer are a
port of an earlier MIT-licensed internal codebase; the llama.cpp library set
comes from a fork of `ggml-org/llama.cpp` (MIT) and the bindings from
`abetlen/llama-cpp-python` (MIT). Neither is vendored here.
