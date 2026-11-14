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
# 0.3.34 has no cp314 wheel, and the devenv's default python is 3.14).
# `uv venv --python 3.13` fetches a uv-managed 3.13 deterministically.
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
#
# Environment:
#   LIBRARY_VENV=PATH         build the venv at PATH instead of ci/library/.venv
#                             (opt:runtime keeps it under .devenv/).
#   LIBRARY_VENV_SCOPE=SCOPE  "full" (default) or "router".
#                               full:   everything above (suites, bench, gates).
#                               router: pydantic, pyyaml, httpx and the project.
#                                       The router launcher and the reconciler
#                                       import only these. The scope is small, so
#                                       it builds fast and needs no GPU stack.
#                             A router venv grows to full when you run the script
#                             again with the full scope. A full venv stays full.
#   LIBRARY_BASE_PYTHON=PATH  build the venv on this interpreter, not on a
#                             uv-managed one. The path must resolve into
#                             /nix/store, and the user "other" must be able to
#                             read and run it. The router unit runs as another
#                             user that cannot read /home. Without this variable,
#                             uv fetches a managed 3.13 (the default for dev).
#                             The script also adds a Nix GC root for the
#                             interpreter, so a garbage collection keeps it.
#                             (LIBRARY_PYTHON means the venv python. Other
#                             scripts read it. Do not use it for this.)
#   LIBRARY_GCROOT_DIR=DIR    where the GC root link goes. Default:
#                             ${XDG_STATE_HOME:-$HOME/.local/state}/inferference/gcroots
#                             The link is named library-python-<store entry>, for
#                             example library-python-<hash>-python3-3.13.x. Each
#                             interpreter keeps its own root. An older root named
#                             `library-python` stays until you remove it.
#   LIBRARY_REPO=DIR          the repository that the editable install uses.
#                             Default: the repository that holds this script.
#                             Set it when you run a script copy from a workspace
#                             to build the venv of the main checkout.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${LIBRARY_VENV:-${here}/library/.venv}"
repo="${LIBRARY_REPO:-$(cd "${here}/.." && pwd)}"
scope="${LIBRARY_VENV_SCOPE:-full}"
base_python="${LIBRARY_BASE_PYTHON:-}"
marker="${venv}/.library-scope"

case "${scope}" in
  full|router) ;;
  *) echo "!! LIBRARY_VENV_SCOPE must be 'full' or 'router', got '${scope}'" >&2; exit 1 ;;
esac

# Fail unless a user other than the build user can read and run the interpreter.
# The router unit runs as such a user. A check that only runs the interpreter
# as the build user passes on a path the unit cannot read.
_require_world_access() {
  local path="$1" dir mode
  mode="$(stat -c %a "${path}")"
  (( (8#${mode} & 8#005) == 8#005 )) \
    || { echo "!! ${path} is mode ${mode}. Other users cannot read and run it." >&2; return 1; }
  dir="$(dirname "${path}")"
  while [[ "${dir}" != "/" ]]; do
    mode="$(stat -c %a "${dir}")"
    (( (8#${mode} & 8#001) == 8#001 )) \
      || { echo "!! ${dir} is mode ${mode}. Other users cannot enter it." >&2; return 1; }
    dir="$(dirname "${dir}")"
  done
}

# Resolve LIBRARY_BASE_PYTHON and prove it is usable by the router unit user.
if [[ -n "${base_python}" ]]; then
  base_python="$(readlink -f "${base_python}")" \
    || { echo "!! LIBRARY_BASE_PYTHON does not resolve" >&2; exit 1; }
  [[ -x "${base_python}" ]] || { echo "!! ${base_python} is not an executable file" >&2; exit 1; }
  case "${base_python}" in
    /nix/store/*) ;;
    *) echo "!! ${base_python} is not under /nix/store. The router user cannot read other paths." >&2; exit 1 ;;
  esac
  _require_world_access "${base_python}" || exit 1
  store_root="$(sed -E 's#^(/nix/store/[^/]+).*#\1#' <<<"${base_python}")"
fi

# Keep the interpreter alive across garbage collection (idempotent).
_add_gc_root() {
  command -v nix-store >/dev/null || { echo "!! nix-store not found. Cannot add a GC root for ${store_root}." >&2; exit 1; }
  local dir="${LIBRARY_GCROOT_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/inferference/gcroots}"
  mkdir -p "${dir}"
  # One root per interpreter, named by its store entry (hash and name). A fixed
  # name would make a second venv with another interpreter drop the first root.
  local root="${dir}/library-python-$(basename "${store_root}")"
  nix-store --add-root "${root}" --indirect --realise "${store_root}" >/dev/null
  echo ">> GC root: ${root} -> ${store_root}"
}

if [[ -e "${venv}" || -L "${venv}" ]]; then
  if "${venv}/bin/python" -V >/dev/null 2>&1; then
    have="$(cat "${marker}" 2>/dev/null || echo full)"   # a venv with no marker predates scopes: full
    if [[ "${have}" == "full" || "${scope}" == "router" ]]; then
      [[ -z "${base_python}" ]] || _add_gc_root
      echo ">> library venv present: ${venv} ($("${venv}/bin/python" -V), scope ${have})"
      exit 0
    fi
    echo ">> growing the ${have} venv to ${scope}: ${venv}"
    grow=1
  else
    echo "!! ${venv} exists but its python does not run (a dangling interpreter link)." >&2
    echo "!! Rename it (mv ${venv} ${venv}.broken), then run this script again." >&2
    exit 1
  fi
fi

command -v uv >/dev/null || { echo "!! uv not found — run inside 'devenv shell'" >&2; exit 1; }

# devenv shells set UV_PYTHON_DOWNLOADS=never and UV_PYTHON_PREFERENCE=
# only-system, which would both block the managed 3.13 below. The uv-managed
# interpreter cache is the intended, deterministic source for this venv
# (devenv's own Python is 3.14 and cannot serve cp313).
export UV_PYTHON_DOWNLOADS=auto
export UV_PYTHON_PREFERENCE=managed

if [[ -z "${grow:-}" ]]; then
  if [[ -n "${base_python}" ]]; then
    _add_gc_root
    uv venv --python "${base_python}" "${venv}"
  else
    uv venv --python 3.13 "${venv}"
  fi
fi

if [[ "${scope}" == "router" ]]; then
  # What ci/runner/inferference-router-launch.sh and inferference-reconcile.py
  # import: pyyaml, pydantic, and the project itself (core dependencies only).
  uv pip install --python "${venv}/bin/python" "pydantic>=2" "pyyaml>=6" "httpx>=0.27"
  uv pip install --python "${venv}/bin/python" --no-deps -e "${repo}"
  echo router >"${marker}"
  echo ">> router venv bootstrapped: ${venv}"
  "${venv}/bin/python" -c "
import sys
assert sys.version_info[:2] == (3, 13), sys.version
import pydantic, yaml, inferference
print('   pydantic', pydantic.__version__, '/ inferference import OK, version', inferference.__version__)
"
  exit 0
fi

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

echo full >"${marker}"
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
