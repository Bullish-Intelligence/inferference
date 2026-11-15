# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The project uses [Semantic Versioning](https://semver.org/). Before 1.0, a minor
release can break compatibility. Each break has a migration step below.

## Unreleased

### Changed

- Hardware: the rig has two AMD Radeon Pro V620 cards (RDNA2, Navi21, gfx1030,
  32 GB each). They replaced the AMD Instinct MI25 cards. Comments and docs
  that named the MI25 now name the V620 or no card.
- Build profiles: the Vulkan profiles lose the card name. `vulkan-mi25` is now
  `vulkan`, `vulkan-mi25-server` is `vulkan-server`, and
  `vulkan-mi25-p2fork-server` is `vulkan-p2fork-server`. The old names stay as
  aliases. `build-llamacpp.sh` and `mode-a-manifest.sh` accept them and print a
  deprecation line. New builds and the Nix `packages.server` use the new
  names, so the `build_id` of `packages.server` changes. The cffi bindgen gate
  accepts both. The experiment preflight treats an old `build_id` prefix as the
  new one, so recorded specs and manifests still resolve
  (`inferference.build_profiles`).
- Gates: `ci/gates/vulkan_mi25_smoke.sh` is now `ci/gates/vulkan_smoke.sh`. It
  names the device from the server log, and it finds builds of both the old and
  the new profile names. The old name stays for one release as a shim that
  prints a deprecation line and runs the new script.

## 0.4.2

### Added

- Packaging: `flake.nix` builds the serving services as a Nix package from
  `uv.lock` (uv2nix, `python313` from nixpkgs). Outputs: `packages.default`
  (the wrapped launchers and tools), `packages.env` (the locked core env) and
  `checks.package` and `checks.lock`.
- Packaging: `packages.server` builds the Mode A `llama-server` with Nix
  (`nix/llama-server.nix`). It uses the llama-infernal fork at one commit, the
  residency patch, and Vulkan. The output has the layout of a published
  artefact, with a `build-manifest.json` and a `modea-provenance.json`. Set
  `INFERFERENCE_SERVER_DIR` to the output. `checks.server-eval` evaluates it
  and does not build it.
- Serving: `inferference-reconcile` and `inferference-balancer` are console
  scripts. The reconcile logic moved to `inferference.reconcile_cli`.
  `ci/runner/inferference-reconcile.py` stays as a thin shim for development.
- Serving: `INFERFERENCE_RESIDENCY_PATCH` names the residency patch that the
  router launcher checks. Without it, the launcher reads the patch beside the
  script, then (development) `ci/build/patches/`.

### Changed

- Serving: the live services and the `llama-server` come from the Nix flake
  (`packages.default`, `packages.server`). The checkout and the library venv are
  for development and gates only. To deploy, pin a release tag (the basix
  input).
- The launchers no longer put the checkout `src/` on the Python path. They use
  `INFERFERENCE_PYTHON`. They use the checkout venv only when it is unset.
- Verify runs `uv lock --check` and evaluates the flake outputs.

## 0.4.1

### Added

- Serving: the replica balancer. The router maps one model name to one child.
  To serve one model from two cards, declare it resident twice under two names.
  Then add a `balancer:` key to `placement.yaml` with `host`, `port` and
  `groups`. Each group gives its members one public name. Omit the key for no
  balancer.
- Serving: `python -m inferference.balancer --config-dir DIR` runs the
  balancer. `ci/runner/inferference-balancer-launch.sh` starts it, and its
  `--check` option validates the config and exits. The balancer reads all
  settings from `placement.yaml`.
- Serving: the balancer sends each request for a group to the loaded member
  with the fewest requests in flight. Equal counts take turns. It polls
  `GET /models` every 2 s.
- Serving: before the first byte, a connection failure, a 5xx answer or a 400
  "model is not loaded" marks the member down until the next poll. The balancer
  then tries the next member once. After the first byte it never retries.
- Serving: the balancer passes streams through without buffering. It removes
  `?autoload` from a client request and never adds it.
- Serving: `/models` and `/v1/models` list the router entries and one entry per
  group. `GET /slots?model=<group>` merges the slot lists of the members.
  `/health` and `/balancer/status` answer for the balancer.
- Serving: `placement.yaml` refuses a bad group. Every member is a resident
  model. A member is in one group at most. A group name equals no resident
  model except its own members. The balancer port differs from the router port
  on one host.
- Library: `inferference.config_paths.read_placement()` reads and validates
  `placement.yaml` only. `BalancerConfig` is the new schema class.

### Changed

- Serving: `resolve_router_url()` returns the balancer URL when `placement.yaml`
  declares a balancer. Otherwise it returns the router URL as before. Clients
  use it. Management tools read `router.port`.
- Serving: the example `placement.example.yaml` puts the router on port 8101
  and the balancer on port 8100.
- Serving: the reconcile CLI and `ci/gates/router_stop_gate.sh` no longer fall
  back to port 8100. They exit with an error when they cannot read
  `router.port`. Pass `--router URL` or set `INFERFERENCE_ROUTER` to override
  the reconcile CLI.
- Serving: the atuin adapter listens on port 8110 by default. It listened on
  port 8100.
- Build: the vendor manifest holds the refreshed source hash. The Mode A
  version tuple holds the patch hash of the current residency patch
  (`d916a61e...`) and the refreshed source hash.

### Breaking changes and migration

1. **Port 8100.** The example placement moves the router to port 8101 and gives
   port 8100 to the balancer. A machine without a `balancer:` key keeps the
   router port that its own `placement.yaml` declares. No code reads 8100 as a
   default now. To add a balancer, set `router.port` to a new port (8101),
   add the `balancer:` key with the old router port, then restart the router
   unit and start the balancer. Point clients at `resolve_router_url()`.
2. **Management tools.** The reconcile CLI and the router stop gate need a
   readable `router.port` in `placement.yaml`, or `--router` /
   `INFERFERENCE_ROUTER`. Fix the config before you run them.
3. **Atuin adapter.** Pass `--listen` or set `endpoint` in the atuin config if
   you relied on port 8100. The new default is 8110.

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
