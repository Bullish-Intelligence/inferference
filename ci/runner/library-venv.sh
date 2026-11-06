#!/usr/bin/env bash
# Bootstrap the library venv for CI jobs and local verification (idempotent).
#
# ci/library/.venv is gitignored (the `.venv/` ignore pattern), so both the
# rig's Act jobs and a fresh checkout must create it in-job. It is the venv
# the Testee gate execs its checks from (see `testee.checks` in devenv.nix): it
# holds the ported library's runtime deps AND the test toolchain
# (pytest / ruff / ty).
#
# CPython 3.13 is pinned (same constraint as the gates venv — llama-cpp-python
# 0.3.34 has no cp314 wheel, and the devenv's default python is 3.14; 10-LIBRARY-
# `uv venv --python 3.13` fetches a
# uv-managed 3.13 deterministically.
#
# Dependency handling:
#   * numpy / pydantic / llama-cpp-python==0.3.34 install normally (clean
#     transitive sets).
#   * xgrammar==0.2.1 is ABI-sensitive and pinned, but its
#     cp313 wheel hard-requires torch / transformers / triton at IMPORT time
#     (xgrammar/__init__.py -> testing.py -> `import torch`;
#     tokenizer_info.py -> `import transformers`). The library's own grammar hot
#     path stays torch-free, but the pinned xgrammar cannot be imported without
#     them, so they enter the venv: CPU-only torch (download.pytorch.org/whl/cpu
#     — no CUDA extras; bench needs torch for CUDA sync anyway,''
#     8.4), transformers (<5), triton, plus apache-tvm-ffi (xgrammar's extension
#     runtime). xgrammar itself installs with --no-deps.
#   * the project itself installs editable with --no-deps (its declared deps are
#     already present from the installs above).
#
# Usage: ci/runner/library-venv.sh   (idempotent; exits 0 if venv already present)
# LIBRARY_VENV=PATH builds the venv at PATH instead of ci/library/.venv (opt:runtime keeps it under .devenv/).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${LIBRARY_VENV:-${here}/library/.venv}"
repo="$(cd "${here}/.." && pwd)"

if [[ -x "${venv}/bin/python" ]]; then
  echo ">> library venv present: ${venv} ($("${venv}/bin/python" -V))"
  exit 0
fi

command -v uv >/dev/null || { echo "!! uv not found — run inside 'devenv shell'" >&2; exit 1; }

# devenv shells set UV_PYTHON_DOWNLOADS=never and UV_PYTHON_PREFERENCE=
# only-system, which would both block the managed 3.13 below. The uv-managed
# interpreter cache is the intended, deterministic source for this venv
# (devenv's own Python is 3.14 and cannot serve cp313).
export UV_PYTHON_DOWNLOADS=auto
export UV_PYTHON_PREFERENCE=managed

uv venv --python 3.13 "${venv}"

# Runtime deps with clean transitive sets.
# httpx and pyyaml are codeeval's: the harness reads the task corpus from YAML
# and streams from an OpenAI-compatible endpoint over HTTP. Without them the
# verification venv cannot even collect the codeeval tests.
uv pip install --python "${venv}/bin/python" \
  "pydantic>=2" "numpy>=2" "llama-cpp-python==0.3.34" "httpx>=0.27" "pyyaml>=6"

# Test toolchain (the gate execs these from this venv's bin — devenv.nix
# `testee.checks`).
# ty is PINNED to 0.0.65: the [[tool.ty.overrides]] in pyproject.toml scope
# reference-inherited strictness findings validated on exactly this version
# (fleet v2 migration note, 2026-10-09). pytest-json-report stays for the
# legacy adapter; Testee 0.5 reads the junitxml the gate now writes.
uv pip install --python "${venv}/bin/python" pytest pytest-json-report ruff "ty==0.0.65"

# CPU-only torch (xgrammar imports it; bench needs it for CUDA sync).
uv pip install --python "${venv}/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu torch

# xgrammar's extension runtime + its import-time app-stack deps (see header).
uv pip install --python "${venv}/bin/python" --no-deps "xgrammar==0.2.1" "apache-tvm-ffi>=0.1.9"
uv pip install --python "${venv}/bin/python" "transformers>=4.38,<5" "triton"

# The project itself (editable, deps already present).
uv pip install --python "${venv}/bin/python" --no-deps -e "${repo}"

echo ">> library venv bootstrapped: ${venv}"
"${venv}/bin/python" -c "
import sys
assert sys.version_info[:2] == (3, 13), sys.version
import numpy, pydantic, llama_cpp
print('   numpy', numpy.__version__, '/ pydantic', pydantic.__version__, '/ llama-cpp-python', llama_cpp.__version__)
import xgrammar as xgr, numpy as np
tok = xgr.TokenizerInfo(['<eos>', '{', '}', '\"', 'ok', ':', 'true', 'false', ',', ' '], xgr.VocabType.RAW, vocab_size=65, stop_token_ids=[0])
compiler = xgr.GrammarCompiler(tok)
matcher = xgr.GrammarMatcher(compiler.compile_json_schema({'type': 'object'}, strict_mode=True))
bm = np.zeros(xgr.get_bitmask_shape(1, 65), dtype=np.int32)
assert matcher.fill_next_token_bitmask(bm) is True
print('   xgrammar 0.2.1 grammar contract OK')
import inferference
print('   inferference import OK, version', inferference.__version__)
"
