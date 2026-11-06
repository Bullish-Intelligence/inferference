#!/usr/bin/env bash
# Publish a routed llama-server as a Mode A artefact.
#
# Stage 7 item 1: the server binary is a Mode A artefact at
#   ci/modea/server/<build_id>/{bin,lib,include,build-manifest.json}
# with a `current` symlink. It needs its own lib/ because the
# build is -DBUILD_SHARED_LIBS=ON.
#
#   mode-a-server.sh --out <out-p2fork-server-*> [--current] [--release] [--router]
#
# --router publishes a llama.cpp ROUTER artefact instead of the single-role
# routed-LoRA one: it skips the seq_adapter_routing/server_seq_routing checks
# below, because router mode (`llama-server` with no `-m`, multi-model by
# name) is a stock llama.cpp capability, present in the anchor build with
# features: [] (AGENTS.md, ABI-anchor rule). Those two manifest keys describe
# the P2 fork's per-context mixed-batch LoRA routing, a different and
# unrelated meaning of "routing" that this artefact does not need to have.
#
# Why publish out of the build garage at all. ci/build/.llamacpp-builds/ is
# gitignored, unbounded, and pruned by ci/build/prune-builds.sh, which keeps only
# the newest out-* per profile. An artefact a consumer runs must not live
# somewhere a housekeeping script deletes. ci/modea/ is where Mode A artefacts
# live and prune-builds.sh does not touch it.
#
# Mode B does not apply here and cannot: LLAMA_CPP_LIB_PATH is read only by the
# Python package, to dlopen libllama.so from a directory. It swaps a shared
# library, never an executable. Every server artefact is Mode A provenance by
# construction.
#
# Every check below is fail-closed and re-run AFTER the copy, from the published
# location — a relocation that held in the build garage is not evidence that it
# holds here.
set -euo pipefail

OUT=""; SET_CURRENT=1; RELEASE=0; ROUTER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)        OUT="$2"; shift 2 ;;
    --no-current) SET_CURRENT=0; shift ;;
    --current)    SET_CURRENT=1; shift ;;
    --release)    RELEASE=1; shift ;;
    --router)     ROUTER=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$OUT" ]] || { echo "usage: mode-a-server.sh --out <out-p2fork-server-*> [--no-current] [--release] [--router]" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root
OUT="$(cd "$OUT" && pwd)"
dest_root="${here}/ci/modea/server"

_die() { echo "!! [mode-a-server] $*" >&2; exit 3; }

# --- the source artefact must be complete and honest --------------------------
man="${OUT}/build-manifest.json"
[[ -f "$man" ]] || _die "no build-manifest.json in ${OUT}. An artefact that cannot name its provenance is not a release artefact."
_key() { python3 -c "import json,sys; print(json.load(open('$man')).get('$1',''))"; }

build_id="$(_key build_id)"
ref_built="$(_key ref_built)"
server_built="$(_key server_built)"
server_routing="$(_key server_seq_routing)"
seq_routing="$(_key seq_adapter_routing)"
profile="$(_key profile)"

[[ -n "$build_id" ]] || _die "the manifest has no build_id"
[[ "$ref_built" =~ ^[0-9a-f]{7,40}$ ]] || _die "the manifest has no valid commit"
[[ "$server_built" == "True" ]] || _die "manifest says server_built=${server_built}. There is no server in this build to publish."
[[ -x "${OUT}/bin/llama-server" ]] || _die "manifest claims server_built but ${OUT}/bin/llama-server is missing or not executable"
if (( ! ROUTER )); then
  [[ "$seq_routing" == "True" ]] || _die "manifest says seq_adapter_routing=${seq_routing}. This is the routed-server artefact; a library that cannot route is not it. (Publishing a router-mode artefact instead? Pass --router.)"
  [[ "$server_routing" == "True" ]] || _die "manifest says server_seq_routing=${server_routing}. The binary does not call the routing API, so it is not the stage-7 artefact. (Publishing a router-mode artefact instead? Pass --router.)"
fi

# A dirty tree gets a -dirty ref (build-llamacpp.sh:96-99) because HEAD then
# names a commit whose source is NOT what was compiled. Such a build may be
# measured; it may never be published.
[[ "$ref_built" != *-dirty ]] || _die "ref_built '${ref_built}' names a dirty tree. Commit and rebuild before publishing."

vm="${here}/ci/build/vendor-manifest.json"
wheel_commit=""
[[ -f "$vm" ]] && wheel_commit="$(python3 -c "import json;print(json.load(open('${vm}'))['llama_cpp']['commit'])" 2>/dev/null || true)"
agree=false
[[ -n "$wheel_commit" && "${wheel_commit:0:${#ref_built}}" == "$ref_built" ]] && agree=true
server_patch="$(_key vulkan_residency_patch_sha256)"
wheel_patch="$(python3 -c "import json;print(json.load(open('${vm}'))['llama_cpp']['vulkan_residency_patch_sha256'])" 2>/dev/null || true)"
if (( RELEASE )); then
  [[ "$agree" == true ]] || _die "--release requires the same fork commit in the wheel and server"
  [[ -n "$server_patch" && "$server_patch" == "$wheel_patch" ]] || _die "--release requires the same Vulkan residency patch in the wheel and server"
fi

dest="${dest_root}/${build_id}"
echo ">> publishing ${OUT}"
echo ">>        -> ${dest}   (profile ${profile}, ref ${ref_built})"

# --- copy ---------------------------------------------------------------------
rm -rf "${dest}.tmp"
mkdir -p "${dest}.tmp"
cp -a "${OUT}/bin"     "${dest}.tmp/"
cp -a "${OUT}/lib"     "${dest}.tmp/"
[[ -d "${OUT}/include" ]] && cp -a "${OUT}/include" "${dest}.tmp/"
cp -a "$man"           "${dest}.tmp/build-manifest.json"

# --- prove the artefact is self-contained FROM ITS NEW HOME -------------------
# The binary's RUNPATH is "$ORIGIN/../lib:<the build tree>" (build-llamacpp.sh
# :368-369). $ORIGIN follows the copy, so the published bin/ must resolve into
# the published lib/. Hide the loader's environment to prove it.
missing="$(env -u LD_LIBRARY_PATH ldd "${dest}.tmp/bin/llama-server" 2>/dev/null | grep 'not found' || true)"
[[ -z "$missing" ]] || { echo "$missing" >&2; rm -rf "${dest}.tmp"; _die "the published llama-server has unresolved libraries"; }

leaked="$(env -u LD_LIBRARY_PATH ldd "${dest}.tmp/bin/llama-server" 2>/dev/null | grep '\.llamacpp-builds/' || true)"
if [[ -n "$leaked" ]]; then
  echo "$leaked" >&2
  rm -rf "${dest}.tmp"
  _die "the published llama-server still resolves libraries out of the prunable build garage. It is a pointer, not an artefact."
fi

# It must also actually start. ldd proves the links resolve; only running it
# proves the binary is the thing it claims to be. /run/opengl-driver/lib goes
# FIRST or the CUDA backend silently falls back.
ver="$(LD_LIBRARY_PATH="/run/opengl-driver/lib" "${dest}.tmp/bin/llama-server" --version 2>&1 | head -3 || true)"
grep -qi 'version' <<<"$ver" || { echo "$ver" >&2; rm -rf "${dest}.tmp"; _die "the published llama-server did not answer --version"; }

# --- provenance sidecar -------------------------------------------------------
# The build manifest is a cross-repo contract (ci/build/build-manifest.md) and is
# copied byte-for-byte. What is true of the PUBLICATION rather than the build
# goes beside it, so no reader of the contract sees a new key.
python3 - "$dest.tmp" "$OUT" "$ref_built" "$wheel_commit" "$agree" <<'PY'
import json, sys
dest, src, ref, wheel, agree = sys.argv[1:6]
json.dump({
    "artefact": "llama-server",
    "mode": "A",
    "mode_b_applies": False,
    "mode_b_reason": "LLAMA_CPP_LIB_PATH swaps a shared library, never an executable "
                     "(llama_cpp/llama_cpp.py:36-43, _ctypes_extensions.py:37-40,:102). "
                     "Every server artefact is Mode A provenance by construction.",
    "source_out_dir": src,
    "server_llama_cpp_commit": ref,
    "wheel_llama_cpp_commit": wheel,
    "wheel_and_server_same_commit": agree == "true",
}, open(dest + "/modea-provenance.json", "w"), indent=2)
PY

# --- swap in ------------------------------------------------------------------
rm -rf "$dest"
mv "${dest}.tmp" "$dest"
if (( SET_CURRENT )); then
  ln -sfn "$build_id" "${dest_root}/current.tmp"
  mv -T "${dest_root}/current.tmp" "${dest_root}/current"
  echo ">> current -> ${build_id}"
fi

# --- the release bar ----------------------------------------------------------
# `release-` tags imply Mode A provenance across the WHOLE tuple. The wheel and
# the server are separately built and can name different commits; that is fine
# for an iteration artefact and is not fine for a release.
if [[ "$agree" != true ]]; then
  echo "!! NOTE: the Mode A wheel was built from ${wheel_commit:-<none recorded>}"
  echo "!!       and this server from ${ref_built}. They are different commits."
  echo "!!       Recorded in modea-provenance.json as wheel_and_server_same_commit=false."
  if (( RELEASE )); then
    _die "--release requires the wheel and the server to name one commit. Rebuild the wheel (ci/build/mode-a-prep.sh with FORK_REF=${ref_built}, then ci/build/mode-a-build.sh) or drop --release."
  fi
fi

echo ">> done."
echo ">>   ${dest}/bin/llama-server"
echo ">>   ${dest}/build-manifest.json"
echo ">>   ${dest}/modea-provenance.json"
echo ">> Mode B does not apply to an executable — run this binary directly."
