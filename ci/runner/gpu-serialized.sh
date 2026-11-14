#!/usr/bin/env bash
# GPU serialization wrapper for CI jobs. GPU jobs must be exclusive:
# serialize with a lock file or a queue job").
#
# Every GPU-bound step in ci/workflows runs through this wrapper. It is now a
# thin wrapper over ci/runner/gpu-lease.sh with a default VRAM budget — P0
# One lock per physical card, so a job on card 1 never waits for card 0.
#
# What changed and why. This script used to take ONE exclusive untimed flock over
# ONE lock file, and pin CUDA_VISIBLE_DEVICES to $LLAMA_CI_GPU. Both cards are
# this project's now, so one lock over one card serialized the wrong
# thing: it blocked a second CI job that had a whole idle card to run on, and it
# guarded nothing on the other card. gpu-lease.sh takes one lock PER CARD and
# places the job on a card that can hold its budget.
#
# Usage is unchanged:
#   ci/runner/gpu-serialized.sh <command> [args...]
#
# The default budget is 7048 MiB — the heaviest CI step measured on this rig
# (the pool-32 probe at -c 16384 --parallel 8). A job that lands is
# therefore guaranteed room for the heaviest observed CI step. Override it per
# job with LLAMA_CI_GPU_BUDGET_MIB when the real footprint is known; the light
# end of the measured range is 2475 MiB (the small ceiling probe).
#
# The child still reads its card from $LLAMA_CI_GPU and the device variables,
# which gpu-lease.sh writes after the placement — so every existing assertion
# against the role variable keeps working, and none of them may become an
# assertion against a literal index (AGENTS.md, GPU discipline).
set -euo pipefail

if [[ -n "${LLAMA_CI_GPU_LOCK:-}" ]]; then
  echo "!! [gpu-serialized] LLAMA_CI_GPU_LOCK is retired: the lock is per card now," >&2
  echo "!! and one lock path cannot express two cards. Set GPU_LEASE_LOCK_DIR" >&2
  echo "!! instead (locks are <dir>/llama-gpu-card<N>.lock). Refusing rather than" >&2
  echo "!! ignoring a serialization override." >&2
  exit 2
fi

exec "$(dirname "$0")/gpu-lease.sh" "${LLAMA_CI_GPU_BUDGET_MIB:-7048}" -- "$@"
