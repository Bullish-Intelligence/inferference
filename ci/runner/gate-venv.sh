#!/usr/bin/env bash
# Bootstrap the gate venv for CI jobs (idempotent).
#
# ci/gates/.venv is gitignored (the `.venv/` ignore pattern), so Act jobs must
# create it in-job. The recipe:
#   uv venv --python <python 3.13> ci/gates/.venv
#   uv pip install cffi setuptools llama-cpp-python==0.3.34
#
# CPython 3.13 is pinned because llama-cpp-python 0.3.34 has no cp314 wheel and
# the devenv's default python is 3.14. `uv venv --python 3.13` fetches a
# uv-managed 3.13 deterministically (same version line as the nix-store
# interpreter); llama-cpp-python is a gate consumer
# and is NOT added to pyproject.toml / the lockfile.
#
# Usage: ci/runner/gate-venv.sh   (idempotent; exits 0 if venv already exists)
# GATE_VENV=PATH builds the venv at PATH instead of ci/gates/.venv.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${GATE_VENV:-${here}/gates/.venv}"

if [[ -x "${venv}/bin/python" ]]; then
  echo ">> gate venv present: ${venv} ($("${venv}/bin/python" -V))"
  exit 0
fi

command -v uv >/dev/null || { echo "!! uv not found — run inside 'devenv shell'" >&2; exit 1; }

# devenv shells set UV_PYTHON_DOWNLOADS=never and UV_PYTHON_PREFERENCE=
# only-system. Their default python is 3.14, so the managed 3.13 needs both
# settings lifted (same as ci/runner/library-venv.sh).
export UV_PYTHON_DOWNLOADS=auto
export UV_PYTHON_PREFERENCE=managed

uv venv --python 3.13 "${venv}"
uv pip install --python "${venv}/bin/python" cffi setuptools llama-cpp-python==0.3.34
echo ">> gate venv bootstrapped: ${venv}"
"${venv}/bin/python" -c "import cffi, llama_cpp; print('   cffi', cffi.__version__, '/ llama-cpp-python', llama_cpp.__version__)"
