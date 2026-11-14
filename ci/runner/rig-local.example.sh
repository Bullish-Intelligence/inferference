#!/usr/bin/env bash
# MACHINE-LOCAL rig inventory. Copy to ci/runner/rig-local.sh and edit.
#
#   cp ci/runner/rig-local.example.sh ci/runner/rig-local.sh
#
# rig-local.sh is UNTRACKED on purpose. Which models a machine holds, and which
# card each one serves, is inventory — it changes when you download a GGUF or
# swap a card, and it differs per machine. The platform itself knows nothing
# about any particular model.
#
# ci/runner/rig-env.sh sources this file and gives you `inferference_role`:
#
#   inferference_role <name> <card-index> <model-path> [key=value ...]
#
# It exports LLAMA_GPU_<NAME> and MODEL_<NAME> — the two variables every server
# unit, gate and suite resolves a role through. Extra key=value pairs become
# <KEY>_<NAME>, for role-scoped extras. A role whose model file is missing is
# NOT declared, so a consumer fails loudly on an unset variable instead of
# quietly serving a model nobody chose.
#
# The card is a plain index here and ONLY here. Everywhere else it is named by
# the role variable (AGENTS.md, GPU discipline).

# Where the host keeps GGUFs (rig-env.sh defaults to ~/.cache/inferference/models).
# export INFERFERENCE_MODELS_DIR="$HOME/.cache/inferference/models"

# --- roles --------------------------------------------------------------------
# One line per model you want to serve or test against. Names are yours; they
# become the role name in `inferference-server@<role>` and the suffix of the
# variables above.
#
# inferference_role small 0 "${INFERFERENCE_MODELS_DIR}/some-3B-Q4_K_M.gguf"
# inferference_role large 1 "${INFERFERENCE_MODELS_DIR}/some-9B-Q4_K_M.gguf" \
#     tokenizer="$HOME/.cache/huggingface/hub/models--vendor--Some-9B"
#
# The second line also exports TOKENIZER_LARGE, for grammar work that needs a
# local tokenizer with no network access.

# Which role generic gates and bench use when none is named ($MODEL_DEFAULT).
# Defaults to the first role declared above.
# export INFERFERENCE_DEFAULT_ROLE=large

# --- what the capability suites need ------------------------------------------
# Each suites/<name>/suite.json names a model ROLE in env.MODEL_ROLE, never a
# model file. ci/runner/run-suite.sh resolves the role to $MODEL_<ROLE> and FAILS,
# naming the variable, when it is unset or unreadable. Declare these:
#
#   role "default"   $MODEL_DEFAULT, from INFERFERENCE_DEFAULT_ROLE above.
#                    Used by: router, cached-router, lora-probe, prefix-cache,
#                    p2-mixed-batch-lora. Any model works.
#                    $TOKENIZER_DEFAULT comes from the default role's
#                    `tokenizer=` pair. The router and cached-router suites need it.
#   $MODEL_PROBE_DIR LoRA probe adapters (probe-a.gguf, probe-b.gguf) built for
#                    the DEFAULT role's architecture. Needed by router,
#                    cached-router, lora-probe and p2-mixed-batch-lora. An
#                    adapter that does not match the model is not a valid probe.
#
#   role "nanbeige"  $MODEL_NANBEIGE: a GGUF whose general.architecture is
#                    nanbeige. These suites test that architecture, so no other
#                    model works. Used by: nanbeige-arch, nanbeige-hats,
#                    p2-server-seq-routing.
#   $MODEL_NANBEIGE_LORA_PROBE
#                    nanbeige LoRA probe adapters. Needed by nanbeige-hats and
#                    p2-server-seq-routing.
#
# Optional, for the routing smoke test (one model per architecture):
#   $MODEL_LLAMA     a llama-architecture GGUF.
#   $ROUTING_SMOKE_ADAPTERS  where its adapter sets live (see test_routing_smoke.py).
#
# inferference_role primary 0 "${INFERFERENCE_MODELS_DIR}/<model>.gguf" \
#     tokenizer="<local tokenizer dir>"
# export INFERFERENCE_DEFAULT_ROLE=primary
# export MODEL_PROBE_DIR="${INFERFERENCE_MODELS_DIR}/<default-model>-lora-probe"
# inferference_role nanbeige 1 "${INFERFERENCE_MODELS_DIR}/<nanbeige-model>.gguf"
# export MODEL_NANBEIGE_LORA_PROBE="${INFERFERENCE_MODELS_DIR}/nanbeige-lora-probe"

# --- fan channels ---------------------------------------------------------------
# The fan channels that duct air through the GPU cards, as a comma list. The
# experiment run summary and curve compare tools read it. `--fan-channel` on
# either tool overrides it. With neither, the tools refuse to guess.
# export INFERFERENCE_DUCT_FAN_CHANNELS="1"

# --- optional overrides --------------------------------------------------------
# Bench baselines root. Set this if you already have a baseline store elsewhere;
# moving the root silently orphans every existing baseline.
# export BASELINES_ROOT="$HOME/.cache/inferference/baselines"

# A build kept deliberately as a NEGATIVE CONTROL (a library without some fix,
# so a test can prove a refusal fires rather than assume it would). Export it
# only when the directory really exists — a missing control is a SKIP, never a
# pass.
# _ctrl="/abs/path/to/out-<profile>-<sha>/lib"
# [[ -d "$_ctrl" ]] && export SEQ_ROUTING_UNFIXED_LIB="$_ctrl"
# unset _ctrl
