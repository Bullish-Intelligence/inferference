# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project uses [Semantic Versioning](https://semver.org/). Before 1.0, a minor
release can break compatibility. Each break has a migration step below.

## 0.4.0

### Added

- Router: `RouteRequest.allow_whitespace=True` lets one request opt in to
  whitespace between JSON tokens. A grammar then allows at most 8 whitespace
  characters at each gap between tokens. Spaces inside strings do not count.
- Router: `seq_routed` refuses an adapter on `token_embd` or on an MoE expert
  tensor. It raises `SeqRoutedAdapterRefused`. Use `context_pool` for such an
  adapter.
- Router: every backend refuses a `token_embd` adapter on a base model with a
  tied output head. It raises `TiedHeadAdapterRefused`.
  `inferference` exports both exceptions. Neither refusal has an override.
- Serving: the `preset-ini` command of `inferference-reconcile.py` prints the
  preset file path. The CLI and the library use one config-dir order:
  `INFERFERENCE_CONFIG_DIR`, then `/etc/inferference`, then
  `~/.config/inferference`.
- Serving: a reload hook. `systemctl reload` runs the reconcile through
  `ci/runner/inferference-router-reload.sh`. The unit template and the NixOS
  module set `ExecReload` and `TimeoutStartSec=900`.
- Serving: the reload script takes a `flock`. It exits with code 75 when
  another reload holds the lock.
- Suites and benchkit: a suite declares its model variables with
  `env.MODEL_VARS` and `env.GPU_TESTS`.
- Suites and benchkit: the suites are backend-neutral. They run on CUDA and on
  Vulkan.
- Build: the `vulkan-p2fork` profile, and a Vulkan wheel build with richer
  server provenance.
- Build: reproducible llama.cpp builds. The lane path does not change the
  bytes of `bin/` and `lib/`.
- Build: `THIRD_PARTY.md`. Each build copies the llama.cpp licence.
- Experiment runner: a SIGTERM stop, duct reports, `fixed.server.extra_args`,
  card gating, rate endpoints in the compare command, CPU state, the `corpus`
  workload and the `agent_turns` workload.
- Experiment runner: `--fan-channel` and `INFERFERENCE_DUCT_FAN_CHANNELS` select
  the duct fan channel.
- Reconcile: a check that refuses a preset that does not match the router
  service config.
- Router venv: the router runs from a venv on a Nix interpreter.

### Changed

- Router: a JSON-schema grammar produces compact JSON by default.
- Router: `check_parity` fails when one output is longer than the other.
- Serving: a reload with no changes exits at once and skips verification.
- Serving: a failed apply unloads only the models that it changed.
- Serving: one deadline covers the whole apply. It stays under the 900 s timeout
  of the unit.
- Serving: the reload script runs python with `exec`.
- Serving: when the config dir is under `/etc`, or the caller cannot write to
  it, the default preset path is in the state dir.
- Serving: the GC root has a name for each interpreter.
- Serving: the launchers expand a leading `~`.
- Suites and benchkit: a GPU suite with no passing GPU test reports SKIP.
- Suites and benchkit: `lib_backend_label` returns `"unknown"` when it finds no
  evidence. It returned `"gpu"` before. It prefers the loaded library.
- Suites and benchkit: `cache_key()` and `baseline_gate` refuse an unknown
  backend.
- Suites and benchkit: `require_gpu0_only` accepts one card. It takes
  `min_model_mib`, checks GTT in absolute mode, and matches the card by PCI
  address or UUID.
- Suites and benchkit: the benchkit pins `HIP_VISIBLE_DEVICES` with the other
  device variables.
- Experiment runner: the server pins `CUDA_VISIBLE_DEVICES` and
  `HIP_VISIBLE_DEVICES` with `GGML_VK_VISIBLE_DEVICES`. Before, it pinned only
  `GGML_VK_VISIBLE_DEVICES`.
- Experiment runner: the stop request ends the long waits.
- Experiment runner: `extra_args` keys are normalised.
- Experiment runner: the `agent_turns` preflight refuses turns that do not fit
  `tokens_per_slot`. A reply with reasoning only aborts the run.
- Build: the residency patch holds `device->mutex` for the read chunk loop. It
  initialises staging lazily. It has a new patch hash. It needs large-BAR
  memory (`DeviceLocal|HostVisible|HostCoherent`).
- Build: a "byte-identical" build needs the same host CPU, the same
  `devenv.lock` and ccache on.
- Repository: the text of the repository no longer names internal projects,
  notes, machines or private models.
- Repository: `.gitignore` ignores `*.jsonl` and `*.pp_table` files outside the
  tracked corpora and fixtures.

### Fixed

- Router: a corrupt GGUF gives the refusal message. It gave `OverflowError`
  before.
- Serving: a failed apply no longer unloads models that it did not change.
- Experiment runner: the stop request no longer waits for the end of a long
  wait.

Known limit: with xgrammar 0.2.1, a free-form `{}` schema in compact mode still
puts a space after each comma.

### Breaking changes and migration

1. **Compact JSON by default.** A JSON-schema grammar no longer allows free
   whitespace. Set `RouteRequest.allow_whitespace=True` on each request that
   needs it. The grammar then allows at most 8 whitespace characters at each
   gap.
2. **Adapter refusals.** `seq_routed` refuses an adapter on `token_embd` or on
   an MoE expert tensor. Every backend refuses a `token_embd` adapter on a base
   with a tied output head. Catch `SeqRoutedAdapterRefused` and
   `TiedHeadAdapterRefused`, or serve such an adapter with `context_pool`
   (`SeqRoutedAdapterRefused` only). No override exists.
3. **NixOS module.** `services.inferference-router.configDir` now defaults to
   `null`. It was `%h/.config/inferference`. The module warns until you set it.
   To keep the old behaviour, set `configDir = "%h/.config/inferference";`.
4. **Existing system units.** Add `ExecReload` and `TimeoutStartSec` of 900 or
   more to each unit. Set `INFERFERENCE_RELOAD_LOCK` in the unit and in the
   environment of any other caller of the reload script. The unit template and
   the NixOS module have these values.
5. **Library venv GC root.** The GC root has a name for each interpreter. The
   old `library-python` root stays until you remove it by hand.
6. **Backend label.** `lib_backend_label` returns `"unknown"` when it finds no
   evidence. `cache_key()` and `baseline_gate` refuse an unknown backend. Point
   the benchkit at a library that shows its backend. Then regenerate any
   baseline that you saved under the label `"gpu"`.
7. **Suites.** Declare the model variables of each suite in `env.MODEL_VARS` and
   `env.GPU_TESTS`. A suite with an unset role variable now fails. Rename a role
   that has an underscore in its name: the loader refuses it.
8. **Experiment specs.** The loader refuses `=` in an `extra_args` key. It also
   refuses the fixed flags and their aliases: `gpu-layers`, `ngl`, `main-gpu`,
   `mg`, `tensor-split`, `ts`, `n-cpu-moe`, `ncmoe`, `cpu-moe`, `cmoe`,
   `override-tensor` and `ot`. Remove them from the spec. The runner sets them.
9. **Duct fan channel.** The default channel is gone. Pass `--fan-channel` or
   set `INFERFERENCE_DUCT_FAN_CHANNELS`.
10. **Vulkan servers.** The launcher refuses a server that carries the old
    residency patch hash. Rebuild the server. The new patch needs large-BAR
    memory.

## 0.3.0

First public release.
