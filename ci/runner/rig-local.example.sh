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

# Where this machine keeps GGUFs (rig-env.sh defaults to ~/.cache/inferference/models).
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
