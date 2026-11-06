#!/usr/bin/env bash
# Emit a build-manifest.json for a Mode A artefact that build-llamacpp.sh did not
# write one for — the wheel.
#
# The manifest does not survive the Mode A copy today, and neither does
# the existing seq_adapter_routing. find ci/modea -name build-manifest.json
# returns nothing." A Mode A artefact that cannot name its provenance is not a
# release artefact.
#
# build-llamacpp.sh emits the manifest beside a lib set it built itself, from the
# git tree it checked out. The wheel is a different path to the same libraries:
# mode-a-prep.sh materializes a PLAIN tree (its .git is stripped, mode-a-prep.sh:49
# and :70), so the wheel's own build has no git tree to derive provenance from.
# This script derives it from a fork clone that knows the commit instead, and the
# commit comes from ci/build/vendor-manifest.json — the record mode-a-prep.sh
# wrote when it materialized that exact tree.
#
#   mode-a-manifest.sh --commit <sha> --lib-dir <dir> --profile <name> --dest <dir>
#                      [--flags <cmake flags>] [--mode A]
#
# FAIL-CLOSED, and this is the load-bearing part. `features` is NOT allowed to
# default to []: ci/runner/run-suite.sh:66 reads `stock = not
# manifest.get("features")`, so an empty list turns every fork-capability suite
# into a SKIP — a silent pass on a fork build. If no clone can answer the
# ancestry question, this script refuses to write a manifest at all.
set -euo pipefail

COMMIT=""; LIB_DIR=""; PROFILE=""; DEST=""; FLAGS=""; MODE="A"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --commit)  COMMIT="$2"; shift 2 ;;
    --lib-dir) LIB_DIR="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --dest)    DEST="$2"; shift 2 ;;
    --flags)   FLAGS="$2"; shift 2 ;;
    --mode)    MODE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$COMMIT" && -n "$LIB_DIR" && -n "$PROFILE" && -n "$DEST" ]] \
  || { echo "usage: mode-a-manifest.sh --commit <sha> --lib-dir <dir> --profile <name> --dest <dir> [--flags <s>]" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # ci/build
work="${here}/.llamacpp-builds"
FORK_REPO="https://github.com/Bullish-Design/llama-infernal"

[[ -d "$LIB_DIR" ]] || { echo "!! --lib-dir ${LIB_DIR} does not exist" >&2; exit 2; }
[[ -d "$DEST" ]]    || { echo "!! --dest ${DEST} does not exist" >&2; exit 2; }

# --- find a fork clone that knows this commit ---------------------------------
src=""
for cand in "${work}"/src-p2fork-maskfix "${work}"/src-p2fork "${work}"/src "${work}"/src-*; do
  [[ -d "${cand}/.git" ]] || continue
  if git -C "$cand" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then src="$cand"; break; fi
done
if [[ -z "$src" ]]; then
  echo "!! no fork clone under ${work} knows commit ${COMMIT}." >&2
  echo "!! features, patch_sha256 and seq_adapter_routing are all derived from" >&2
  echo "!! that tree's history, and none of them may be guessed:" >&2
  echo "!!   run-suite.sh:66 reads an empty 'features' as a STOCK build and turns" >&2
  echo "!!   every fork-capability suite into a skip — a silent pass." >&2
  echo "!! Fetch the commit into ${work}/src-p2fork and re-run." >&2
  exit 3
fi
echo ">> provenance clone: ${src}"
git -C "$src" fetch --tags --force origin >/dev/null 2>&1 || true

short="$(git -C "$src" rev-parse --short "$COMMIT")"

# --- features, by ancestry (same rule as build-llamacpp.sh) -------------------
features=()
while IFS= read -r br; do
  if git -C "$src" merge-base --is-ancestor "$br" "$COMMIT" 2>/dev/null; then
    features+=("\"${br#refs/remotes/origin/feat/}\"")
  fi
done < <(git -C "$src" for-each-ref --format='%(refname)' 'refs/remotes/origin/feat/*' 2>/dev/null | sort)
if [[ ${#features[@]} -eq 0 ]]; then
  echo "!! no feat/* branch is an ancestor of ${short}, so 'features' would be []." >&2
  echo "!! That is how a STOCK build reads, and this is a Mode A fork artefact." >&2
  echo "!! Refusing to write a manifest that would silently skip every" >&2
  echo "!! fork-capability suite (run-suite.sh:66)." >&2
  exit 3
fi
features_json="[$(IFS=,; echo "${features[*]}")]"

# --- the two fail-closed p2fork checks, against the commit, not the profile ---
patch_sha="$(git -C "$src" log --format=%B "$COMMIT" \
  | grep -A1 -m1 'p2-mixed-batch-lora\.patch' \
  | sed -n 's/^sha256 \([0-9a-f]\{64\}\)$/\1/p' | head -1 || true)"
if [[ -z "$patch_sha" ]]; then
  echo "!! no P2 provenance marker in ${short}'s history. Refusing to claim" >&2
  echo "!! seq_adapter_routing for a tree that cannot name the P2 change." >&2
  exit 3
fi
# `git show ... | grep -q` is wrong under `set -o pipefail`: grep exits on the
# first match, git show takes SIGPIPE, and the pipeline reports 141 — a PASS read
# as a FAIL. Ask git itself instead, so there is no pipe.
if ! git -C "$src" grep -q "llama_set_seq_adapters" "$COMMIT" -- include/llama.h; then
  echo "!! ${short} does not declare llama_set_seq_adapters in include/llama.h —" >&2
  echo "!! the routing surface is absent from the tree that produced these libs." >&2
  exit 3
fi

# server_seq_routing: does THIS commit's tools/server/ call the routing API?
# Read from the commit, the same derivation build-llamacpp.sh makes from the
# checked-out tree — never from the profile name.
if git -C "$src" grep -q 'llama_set_seq_adapter' "$COMMIT" -- 'tools/server/' 2>/dev/null; then
  commit_server_routing=true
else
  commit_server_routing=false
fi

# --- ggml version, from the artefact's own libs -------------------------------
ggml_lib_path="$(ls "${LIB_DIR}"/libggml.so.* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$ggml_lib_path" ]]; then
  echo "!! no libggml.so.* under ${LIB_DIR}. Refusing to emit a manifest for a" >&2
  echo "!! lib set with no ggml library." >&2
  exit 3
fi
ggml_ver="$(basename "$ggml_lib_path")"

# The wheel bundles no executable: llama-cpp-python's package is a ctypes binding
# over the shared libraries. server_built is false by construction here, and
# server_seq_routing follows it (build-manifest.md: server_built=false forces
# false), even where the commit's own tools/server/ does route.
server_built=false
server_seq_routing=false

anchor=""
[[ -f "${here}/vendor-manifest.json" ]] && anchor="$(python3 -c "import json;print(json.load(open('${here}/vendor-manifest.json')).get('llama_cpp_anchor',''))" 2>/dev/null || true)"
residency_sha="$(python3 -c "import json;print(json.load(open('${here}/vendor-manifest.json'))['llama_cpp']['vulkan_residency_patch_sha256'])")"
[[ "$residency_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "!! missing Vulkan residency patch provenance" >&2; exit 1; }
source_sha="$(python3 -c "import json;print(json.load(open('${here}/vendor-manifest.json'))['source_tree_sha256'])")"
[[ "$source_sha" =~ ^[0-9a-f]{64}$ ]] || { echo "!! missing vendor source provenance" >&2; exit 1; }

cat > "${DEST}/build-manifest.json" <<JSON
{
  "repo": "${FORK_REPO}",
  "ref_requested": "${COMMIT}",
  "ref_built": "${short}",
  "profile": "${PROFILE}",
  "cmake_flags": "${FLAGS}",
  "ggml_lib": "${ggml_ver}",
  "ggml_version": "${ggml_ver}",
  "llama_cpp_commit": "${short}",
  "build_id": "${PROFILE}-${short}-residency-${residency_sha:0:12}",
  "vulkan_residency_patch_sha256": "${residency_sha}",
  "source_tree_sha256": "${source_sha}",
  "features": ${features_json},
  "patch": "commits on feat/p2-mixed-batch-lora (P2 patch provenance sha256=${patch_sha}; no patch applied)",
  "patch_sha256": "${patch_sha}",
  "seq_adapter_routing": true,
  "server_built": ${server_built},
  "server_seq_routing": ${server_seq_routing},
  "mode": "${MODE}",
  "llama_cpp_anchor": "${anchor}",
  "commit_server_seq_routing": ${commit_server_routing}
}
JSON
echo ">> manifest: ${DEST}/build-manifest.json (build_id ${PROFILE}-${short}-residency-${residency_sha:0:12}, features ${features_json})"
