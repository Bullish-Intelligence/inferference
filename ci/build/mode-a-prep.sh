#!/usr/bin/env bash
# Mode A prep — materialize the vendored llama-cpp-python checkout with its
# nested vendor/llama.cpp re-pointed to llama-infernal@integration.
#
# llama-cpp-python is NOT forked; stock v0.3.34
# enters as a vendored checkout whose nested vendor/llama.cpp is re-pointed to
# the fork's integration branch, so a Mode A source rebuild compiles OUR
# integration lib into the wheel.
#
# Mechanism: PINNED CLONE (prompt-sanctioned fallback, not a git submodule).
# Why not a submodule: (a) gitman is the only VC for inferference and does not
# model nested submodule repos — a .gitmodules/gitlink change risks canonicity
# churn; (b) a submodule re-point is not deterministic: the parent pins only
# the submodule HEAD, not its .gitmodules, so a fresh clone restores the stock
# ggml-org re-point; (c) committing the full vendored sources (~100 MB) bloats
# this repo. The pinned clones are materialized here (deterministic: exact
# commits recorded in vendor-manifest.json), and vendor/ is gitignored.
#
# Outputs (gitignored):
#   vendor/llama-cpp-python/          plain tree @ v0.3.34 (no .git)
#   vendor/llama-cpp-python/vendor/llama.cpp   plain tree @ integration (no .git or agent metadata)
#   ci/build/vendor-manifest.json     the exact commits (the build record)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # repo root (script lives in ci/build/)
vendor="${here}/vendor"
lcpp="${vendor}/llama-cpp-python"
llama="${lcpp}/vendor/llama.cpp"

LCPP_REPO="https://github.com/abetlen/llama-cpp-python"
LCPP_REF="v0.3.34"
FORK_REPO="https://github.com/Bullish-Design/llama-infernal"
FORK_REF="${FORK_REF:-integration}"

mkdir -p "$vendor"

# --- llama-cpp-python @ v0.3.34 ----------------------------------------------
# A leftover materialization is a plain tree (its .git was stripped on the prior
# run) and cannot be re-checked out — re-clone whenever .git is missing so a
# stale tree is never silently reused (fail-closed). A bare .git directory with
# no HEAD (a stray objects dir) is not a clone, so test for HEAD.
if [[ ! -f "${lcpp}/.git/HEAD" ]]; then
  rm -rf "$lcpp"
  git clone --quiet "$LCPP_REPO" "$lcpp"
fi
git -C "$lcpp" checkout --quiet "$LCPP_REF"
git -C "$lcpp" fetch --quiet --tags origin || true
git -C "$lcpp" reset --hard --quiet "$LCPP_REF"
lcpp_sha="$(git -C "$lcpp" rev-parse HEAD)"
rm -rf "${lcpp}/.git"   # plain tree: the parent repo must not see a nested repo

# --- nested vendor/llama.cpp @ fork integration --------------------------------
rm -rf "$llama"
git clone --quiet "$FORK_REPO" "$llama"
git -C "$llama" checkout --quiet "$FORK_REF"
llama_sha="$(git -C "$llama" rev-parse HEAD)"
patch_file="${here}/ci/build/patches/vulkan-residency.patch"
patch_base="ecc0cb0d34e17595ff0546b89c44fa100d076c0d"
[[ "$llama_sha" == "$patch_base" ]] || { echo "!! Vulkan residency patch requires fork ${patch_base}; got ${llama_sha}" >&2; exit 1; }
patch_sha="$(sha256sum "$patch_file" | cut -d ' ' -f 1)"
git -C "$llama" apply --check "$patch_file"
git -C "$llama" apply "$patch_file"

# --- ABI anchor ----------------------------------------------------------------
# The anchor = the fork's base commit (the anchor branch, a strict ancestor of
# every fork integration; llama-cpp-python 0.3.34 pairs with it). Resolved from
# the clone so the record is read from the tree, never a hardcoded sha. A fresh
# clone exposes the branch as origin/<ref> (no local tracking branches).
ANCHOR_REF="${ANCHOR_REF:-anchor/b10233}"
anchor_sha="$(git -C "$llama" merge-base "$FORK_REF" "origin/${ANCHOR_REF}" 2>/dev/null \
  || git -C "$llama" rev-parse "origin/${ANCHOR_REF}" 2>/dev/null \
  || git -C "$llama" rev-parse "$ANCHOR_REF" 2>/dev/null \
  || true)"
[[ -n "${anchor_sha}" ]] || { echo "!! could not resolve the ABI anchor (${ANCHOR_REF}) in the fork clone" >&2; exit 1; }
anchor_short="${anchor_sha:0:9}"

rm -rf "${llama}/.git"

# Agent skills and Copier answers manage source work. They do not enter either build.
"${here}/ci/build/prune-project-metadata.sh" "$llama"
source_sha="$(tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -C "$lcpp" -cf - . | sha256sum | cut -d ' ' -f 1)"

# --- vendor manifest (the build record) ---------------------------------------
mkdir -p "${here}/ci/build"
cat > "${here}/ci/build/vendor-manifest.json" <<JSON
{
  "llama_cpp_python": { "repo": "${LCPP_REPO}", "ref": "${LCPP_REF}", "commit": "${lcpp_sha}" },
  "llama_cpp":        { "repo": "${FORK_REPO}", "ref": "${FORK_REF}", "commit": "${llama_sha}", "vulkan_residency_patch_sha256": "${patch_sha}" },
  "llama_cpp_anchor": "${anchor_short}",
  "source_tree_sha256": "${source_sha}"
}
JSON

echo ">> vendored llama-cpp-python ${LCPP_REF} (${lcpp_sha})"
echo ">> re-pointed vendor/llama.cpp -> ${FORK_REPO} ${FORK_REF} (${llama_sha})"
echo ">> ABI anchor ${ANCHOR_REF} -> ${anchor_short}"
echo ">> vendor manifest: ci/build/vendor-manifest.json"
