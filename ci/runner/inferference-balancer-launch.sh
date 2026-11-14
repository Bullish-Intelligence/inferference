#!/usr/bin/env bash
# Start the replica BALANCER: one process in front of the llama.cpp router that
# gives a group of resident models one public name.
#
#   inferference-balancer-launch.sh [--check]
#
# --check reads and validates placement.yaml, prints the listen address, the
# router address and the groups, and exits without listening.
#
# The balancer reads its whole configuration from placement.yaml (`balancer:`
# for host, port and groups; `router:` for the upstream). See
# ci/runner/placement.example.yaml and src/inferference/balancer.py.
#
# Environment (install plumbing only):
#   INFERFERENCE_CONFIG_DIR   where placement.yaml lives. Unset: the reconcile
#                             CLI names the dir, as in the router launcher.
#   INFERFERENCE_PYTHON       interpreter. Default the library venv of the
#                             checkout, as the router launcher.
set -euo pipefail

check=0
[[ "${1:-}" == "--check" ]] && check=1

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # ci/runner
repo="$(cd "${here}/../.." && pwd)"

_cfg() { echo "!! [inferference-balancer] $*" >&2; exit 78; }

reconciler="${here}/inferference-reconcile.py"
py="${INFERFERENCE_PYTHON:-${repo}/ci/library/.venv/bin/python}"
[[ -x "$py" ]] || _cfg "no python at ${py}. Set \$INFERFERENCE_PYTHON, or build the router venv: LIBRARY_VENV_SCOPE=router LIBRARY_BASE_PYTHON=<python in /nix/store> ci/runner/library-venv.sh."

if [[ -n "${INFERFERENCE_CONFIG_DIR:-}" ]]; then
  config_dir="$INFERFERENCE_CONFIG_DIR"
else
  config_dir="$("$py" "$reconciler" config-dir)" \
    || _cfg "the reconcile CLI could not name the config dir. Set \$INFERFERENCE_CONFIG_DIR."
  [[ -n "$config_dir" ]] || _cfg "the reconcile CLI printed no config dir. Set \$INFERFERENCE_CONFIG_DIR."
fi
[[ -f "${config_dir}/placement.yaml" ]] || _cfg "no placement at ${config_dir}/placement.yaml. See ci/runner/placement.example.yaml."

# This checkout's source wins over any installed copy of the library.
export PYTHONPATH="${repo}/src${PYTHONPATH:+:${PYTHONPATH}}"

argv=("$py" -m inferference.balancer --config-dir "$config_dir")
if (( check )); then
  exec "${argv[@]}" --check
fi
exec "${argv[@]}"
