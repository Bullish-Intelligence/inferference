#!/usr/bin/env bash
# Mode A build — source-rebuild llama-cpp-python 0.3.34 against the vendored
# llama-infernal@integration tree and install the wheel into a dedicated venv.
#
# Mode A recipe:
#   CMAKE_ARGS="<backend flags>" FORCE_CMAKE=1
#   uv pip install --no-binary llama-cpp-python <vendor path>
#
# Backend. Set MODEA_BACKEND=vulkan for a Vulkan wheel. It selects
#   CMAKE_ARGS="-DGGML_VULKAN=ON -DGGML_CUDA=OFF"
# and builds the server with the vulkan-mi25-p2fork-server profile.
# MODEA_BACKEND=cuda (the default) keeps the historical flags
#   "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86".
# A CMAKE_ARGS from the caller wins over both. After the build the script
# checks that the bundled libs match the flags, and it records the real flags
# in ci/build/modea-version-tuple.json.
#
# The wheel bundles the lib set built from OUR integration tree (Mode A
# provenance); it must be verified WITHOUT an external LLAMA_CPP_LIB_PATH lib
# swap — the smoke gate and bindgen run against the wheel's own bundled
# libs under the installed site-packages.
#
# Prereqs: ci/build/mode-a-prep.sh has run (vendor/ materialized); run inside
# `devenv shell` (gcc, cmake, ninja, ccache, CUDA toolkit are devenv packages).
#
# Outputs (gitignored):
#   ci/modea/.venv/                     the dedicated venv with the Mode A wheel
#   ci/build/modea-version-tuple.json   the full version tuple
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # repo root (script lives in ci/build/)
lcpp="${here}/vendor/llama-cpp-python"
venv="${here}/ci/modea/.venv"
manifest="${here}/ci/build/vendor-manifest.json"

[[ -f "${lcpp}/pyproject.toml" ]] || { echo "!! vendor/llama-cpp-python missing — run ci/build/mode-a-prep.sh first" >&2; exit 1; }
[[ -f "$manifest" ]] || { echo "!! vendor manifest missing — run ci/build/mode-a-prep.sh first" >&2; exit 1; }
recorded_source_sha="$(python3 -c "import json;print(json.load(open('${manifest}'))['source_tree_sha256'])")"
source_sha="$(tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$lcpp" -cf - . | sha256sum | cut -d ' ' -f 1)"
[[ "$source_sha" == "$recorded_source_sha" ]] || { echo "!! vendor source changed after preparation; run ci/build/mode-a-prep.sh again" >&2; exit 1; }

# Reuse the gate venv's proven interpreter (CPython 3.13 — llama-cpp-python
# 0.3.34 has no cp314 wheel) for a fresh dedicated Mode A venv.
# --clear, or this script only ever works ONCE on a machine: `uv venv` refuses an
# existing directory ("A virtual environment already exists"), so every rebuild
# after the first died at rc=2 before compiling anything. Found 2026-08-26, the
# first time the Mode A wheel was rebuilt since it was first created.
#
# Clearing is right rather than merely convenient: a Mode A venv must contain the
# wheel built from THIS commit and nothing left over from a previous one. The
# built wheel is archived under ci/modea/wheels/ separately, so nothing
# unrecoverable is discarded here.
uv venv --clear --python "${MODEA_PYTHON:-3.13}" "$venv"
PY="${venv}/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 13), "Mode A requires Python 3.13"'

backend="${MODEA_BACKEND:-cuda}"
case "$backend" in
  cuda)   default_cmake_args="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"; server_profile="p2fork-server" ;;
  vulkan) default_cmake_args="-DGGML_VULKAN=ON -DGGML_CUDA=OFF";            server_profile="vulkan-mi25-p2fork-server" ;;
  *) echo "!! MODEA_BACKEND must be cuda or vulkan, got '${backend}'" >&2; exit 2 ;;
esac
export CMAKE_ARGS="${CMAKE_ARGS:-$default_cmake_args}"
export FORCE_CMAKE=1
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/llamacpp-ccache}"
mkdir -p "$CCACHE_DIR"

# Path-independent compile, as in build-llamacpp.sh. The wheel compiles the
# vendored llama.cpp in place, and GGML_ASSERT bakes the absolute source path
# (<lane>/vendor/llama-cpp-python/vendor/llama.cpp/...) into .rodata through
# __FILE__. Two lanes then give two wheels with different bytes. cmake reads CFLAGS and
# CXXFLAGS at the first configure, so the map reaches every compile without a change to
# CMAKE_ARGS (which the caller may set, and which scikit-build-core splits on
# whitespace). The RUNPATH needs no step here: the vendored CMakeLists sets
# INSTALL_RPATH "$ORIGIN" with BUILD_WITH_INSTALL_RPATH.
real_lcpp="$(cd "$lcpp" && pwd -P)"
if [[ "$lcpp" == *[[:space:]]* || "$real_lcpp" == *[[:space:]]* ]]; then
  echo "!! path with whitespace cannot go into CFLAGS: '${lcpp}'" >&2; exit 3
fi
wheel_maps="-ffile-prefix-map=${lcpp}=/llama-cpp-python -ffile-prefix-map=${real_lcpp}=/llama-cpp-python"
export CFLAGS="${CFLAGS:+${CFLAGS} }${wheel_maps}"
export CXXFLAGS="${CXXFLAGS:+${CXXFLAGS} }${wheel_maps}"

echo ">> Mode A: building llama-cpp-python from ${lcpp} (CMAKE_ARGS='${CMAKE_ARGS}')"
uv pip install --python "$PY" --no-binary llama-cpp-python --no-cache-dir "$lcpp"

# Check the bundled libraries in this interpreter before recording provenance.
env -u LLAMA_CPP_LIB_PATH "$PY" -c 'import llama_cpp; print("Mode A import:", llama_cpp.__version__)' \
  || { echo "!! Mode A cannot import its bundled libraries in the selected interpreter" >&2; exit 1; }

# --- locate the wheel's bundled libs + record the version tuple -----------------
site="$( "$PY" -c 'import site, sys; print(site.getsitepackages()[0])' )"
libdir="${site}/llama_cpp/lib"
echo ">> wheel bundled libs: ${libdir}"
ls -la "$libdir" | grep -E "libllama|libggml" | head -8

# The libs must match the flags. A Vulkan wheel without libggml-vulkan, or one
# that carries a CUDA backend, would make the version tuple below say something
# the wheel does not do.
if [[ "$CMAKE_ARGS" == *-DGGML_VULKAN=ON* ]] && ! ls "${libdir}"/libggml-vulkan.so* >/dev/null 2>&1; then
  echo "!! CMAKE_ARGS asks for Vulkan but ${libdir} has no libggml-vulkan.so" >&2; exit 3
fi
if [[ "$CMAKE_ARGS" == *-DGGML_CUDA=ON* ]] && ! ls "${libdir}"/libggml-cuda.so* >/dev/null 2>&1; then
  echo "!! CMAKE_ARGS asks for CUDA but ${libdir} has no libggml-cuda.so" >&2; exit 3
fi
if [[ "$CMAKE_ARGS" != *-DGGML_CUDA=ON* ]] && ls "${libdir}"/libggml-cuda.so* >/dev/null 2>&1; then
  echo "!! CMAKE_ARGS does not ask for CUDA but ${libdir} carries a CUDA backend" >&2; exit 3
fi

# Record the versioned soname, not the unversioned one. The wheel bundles
# libggml.so, libggml.so.0 and libggml.so.0.18.0 as three real files, and a glob
# sorts the shortest first — so the old `ls ... | head -1` wrote `libggml.so.0`
# into the tuple. Same selection as build-llamacpp.sh, and it works whether the
# short names are symlinks (build garage) or copies (wheel).
# The `|| true` is load-bearing: `set -euo pipefail` makes the no-match `ls`
# abort the assignment, which would leave the check below unreachable — the same
# dead-fallback defect this change removes.
ggml_lib_path="$(ls "${libdir}"/libggml.so.* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$ggml_lib_path" ]]; then
  echo "!! no libggml.so.* under ${libdir}. The wheel did not bundle its ggml" >&2
  echo "!! libraries; refusing to write a version tuple that cannot name one." >&2
  exit 3
fi

cat > "${here}/ci/build/modea-version-tuple.json" <<JSON
{
  "llama_cpp_python_ref": "v0.3.34",
  "llama_cpp_python_commit": "$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp_python']['commit'])")",
  "llama_cpp_anchor": "$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp_anchor'])")",
  "llama_cpp_built_commit": "$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp']['commit'])")",
  "vulkan_residency_patch_sha256": "$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp']['vulkan_residency_patch_sha256'])")",
  "source_tree_sha256": "${source_sha}",
  "build_flags": "${CMAKE_ARGS}",
  "ggml_version": "$(basename "$ggml_lib_path")",
  "mode": "A"
}
JSON
echo ">> version tuple: ci/build/modea-version-tuple.json"
cat "${here}/ci/build/modea-version-tuple.json"

# --- the wheel's own build manifest ----------------------------------
# "The manifest does not survive the Mode A copy today": build-llamacpp.sh writes
# build-manifest.json beside a lib set it built itself, and the wheel is a
# different path to the same libraries. `find ci/modea -name build-manifest.json`
# returned nothing, so the shipped wheel could not name its own provenance.
#
# It goes in the llama_cpp package dir, NOT the lib dir: diagnostics.py's
# _build_manifest looks in the lib dir's PARENT first (build-manifest.md "How
# the contract is read"), and <site>/llama_cpp is the parent of
# <site>/llama_cpp/lib. mode-a-manifest.sh fails closed rather than guess any key.
built_commit="$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp']['commit'])")"
"${here}/ci/build/mode-a-manifest.sh" \
  --commit "$built_commit" --lib-dir "$libdir" --profile modea-wheel \
  --dest "${site}/llama_cpp" --flags "${CMAKE_ARGS}"

# The wheel bundles llama.cpp and ggml code. Put llama.cpp's MIT licence in the package dir
# beside build-manifest.json, as the notice requires. The licence of llama-cpp-python
# itself already ships in the wheel metadata.
[[ -f "${lcpp}/vendor/llama.cpp/LICENSE" ]] \
  || { echo "!! no LICENSE in ${lcpp}/vendor/llama.cpp. Refusing a wheel with no llama.cpp licence notice." >&2; exit 3; }
cp "${lcpp}/vendor/llama.cpp/LICENSE" "${site}/llama_cpp/LICENSE.llama.cpp"

# --- the server, beside the wheel (stage 7 item 1) ----------------------------
# The wheel is a ctypes binding over shared libraries and bundles no executable,
# so llama-server cannot come out of the pip build. It is built from the SAME
# commit the wheel was built from — that is what "beside the wheel" has to mean
# for a release artefact — and published under ci/modea/server/.
if [[ "${MODEA_SKIP_SERVER:-0}" != "1" ]]; then
  echo ">> Mode A: building llama-server from the same commit (${built_commit})"
  "${here}/ci/build/build-llamacpp.sh" --ref "$built_commit" --profile "$server_profile"
  short="$(git -C "${here}/ci/build/.llamacpp-builds/src-p2fork" rev-parse --short "$built_commit")"
  residency_sha="$(python3 -c "import json; print(json.load(open('${manifest}'))['llama_cpp']['vulkan_residency_patch_sha256'])")"
  "${here}/ci/build/mode-a-server.sh" \
    --out "${here}/ci/build/.llamacpp-builds/out-${server_profile}-${short}-residency-${residency_sha:0:12}" --release
else
  echo ">> MODEA_SKIP_SERVER=1: the wheel is built, the server artefact is NOT."
  echo ">> This wheel is not a release tuple — publish the server with"
  echo ">> ci/build/mode-a-server.sh --out <out-p2fork-server-...> --release."
fi

echo ">> NEXT: run the ABI smoke gate + bindgen against the wheel's own libs (ci/build/mode-a.md)"
echo "   LLAMA_CPP_LIB_PATH=${libdir} (no external lib swap)"
