#!/usr/bin/env bash
# Build a chosen llama.cpp ref (fork branch, release tag, or local tree) into a
# shared-lib set that llama-cpp-python can load via LLAMA_CPP_LIB_PATH (Mode B)
# or that a source rebuild of llama-cpp-python can bundle (Mode A).
#
# Ported from an earlier internal build script, with the deltas this
# pipeline needs:
#
#   * default REPO is the fork (https://github.com/Bullish-Design/llama-infernal),
#     overridable with --repo. The script never pushes (upstream stays fetch-only).
#   * --ref accepts BRANCH NAMES (feat/p2-mixed-batch-lora, integration, ...):
#     the checkout detaches at the branch tip and ref_built records the tip short
#     sha, so out-<profile>-<ref> dir naming is unchanged from the reference.
#   * the p2fork profile's hard-coded anchor/patch defaults become repo-local
#     defaults: REF defaults to feat/p2-mixed-batch-lora and NO patch is applied
#     (the P2 change is real commits on the fork, never a patch). The manifest
#     still records patch_sha256, derived from the commit provenance: each P2
#     commit message carries "sha256 <hex>" (verified 2026-08-01: commits
#     108d4d756 / 52da9663c / fa8cab171 on feat/p2-mixed-batch-lora all carry
#     sha256 8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752).
#
# Run inside `devenv shell` (gcc, cmake, ninja, ccache, and the CUDA toolkit are
# devenv packages; NIXPKGS_ALLOW_UNFREE=1 for the unfree CUDA toolkit). See
# ci/build/build-manifest.md for the manifest contract, and
# The ABI-anchor rule: the built commit
# MUST stay ABI-compatible with llama-cpp-python 0.3.34's bindings (anchor
# 0ab9d6fed). Run the ABI smoke gate + cffi bindgen after every build.
#
# Usage:
#   build-llamacpp.sh --ref <git-ref|local:PATH> \
#       --profile <cpu-light|cuda-3060|cuda-3060-server|p2fork|p2fork-server
#                  |vulkan-mi25|vulkan-mi25-server
#                  |vulkan-mi25-p2fork-server> [--out DIR]
#
# The vulkan-mi25-p2fork-server profile is the P2 fork's source tree built with
# the Vulkan backend instead of CUDA: the mixed-batch per-sequence
# LoRA routing feature is backend-agnostic C++ (it calls only core ggml ops), so
# it needs no code change to run on the MI25s — only a profile that asks for both
# axes at once, which no profile did before.
#
# The p2fork-server profile is p2fork plus the server binary
# (005-server-seq-routing spike S4). It shares p2fork's CUDA flags, its src
# checkout, and its provenance checks; it adds -DLLAMA_BUILD_SERVER=ON, builds
# the explicit `llama llama-server` target pair, and collects bin/llama-server
# into <out>/bin/. Every other profile is byte-identical to before.
set -euo pipefail

REF=""
PROFILE="cuda-3060"
OUT=""
REPO="https://github.com/Bullish-Design/llama-infernal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)     REF="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    --repo)    REPO="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${here}/.llamacpp-builds"
mkdir -p "$work"

# --- profile family flags -----------------------------------------------------
# p2fork-server is a member of the p2fork family: same src checkout, same ref
# default, same fail-closed provenance checks, same seq_adapter_routing=true.
# It differs only in what cmake builds. Deriving two booleans here keeps every
# downstream `p2fork` test in one place instead of five string compares.
# vulkan-mi25-p2fork-server is the same family again: the P2 source tree, built
# for Vulkan instead of CUDA. The source-tree axis (is_p2fork) and
# the server axis (build_server) are already derived independently here; only the
# flag `case` below was flat, so the profile is added there too rather than
# refactored — the two axes that matter are already separate.
case "$PROFILE" in
  p2fork | p2fork-server | vulkan-mi25-p2fork-server) is_p2fork=1 ;;
  *)                                                  is_p2fork=0 ;;
esac
build_server=$([[ "$PROFILE" == *-server ]] && echo 1 || echo 0)

# --- p2fork profile: repo-local defaults --------------------------------------
# The P2 mixed-batch per-sequence LoRA routing change is REAL COMMITS on the
# fork's feat/p2-mixed-batch-lora branch (no patch file exists in this repo).
# The profile therefore defaults the ref to that branch and, after checkout,
# derives patch_sha256 from the tip commit's provenance (see header comment).
# seq_adapter_routing is derived per-profile below (true iff p2fork), so a
# manifest can never claim routing for a stock tree.
if (( is_p2fork )); then
  : "${REF:=feat/p2-mixed-batch-lora}"
fi
[[ -n "$REF" ]] || { echo "--ref required (git ref, or local:/abs/path)" >&2; exit 2; }

# --- resolve source tree ------------------------------------------------------
if [[ "$REF" == local:* ]]; then
  src="${REF#local:}"
  built_ref="$(git -C "$src" rev-parse --short HEAD 2>/dev/null || echo local-worktree)"
  # A local: tree can hold uncommitted work, and HEAD then names a commit whose
  # source is NOT what is being compiled. Naming the out dir after it silently
  # overwrites that commit's artefacts with different code, and any later
  # measurement against the dir believes the label. Measured cost: one stage-6A
  # A/B where both arms turned out to be the same binary.
  #
  # A dirty tree gets its own name instead. Nothing is refused - building
  # uncommitted work is the normal edit loop - but it can never claim to be a
  # commit it is not.
  if [[ "$built_ref" != "local-worktree" ]] && ! git -C "$src" diff --quiet HEAD 2>/dev/null; then
    built_ref="${built_ref}-dirty"
    echo ">> ${src} has uncommitted changes: naming this build ${built_ref}"
  fi
else
  # p2fork uses a dedicated src checkout so its provenance is never confused
  # with a stock cuda-3060 build of the same tree (reference behavior kept).
  # A non-default --repo (the canary builds stock ggml-org/llama.cpp) also gets
  # a dedicated checkout: the default src/ clone's origin is the fork, and a
  # `git fetch origin` against it would silently rebuild the fork instead of
  # stock upstream — never let a --repo override land on the wrong tree.
  if (( is_p2fork )); then
    src="${work}/src-p2fork"
  elif [[ "$REPO" != "https://github.com/Bullish-Design/llama-infernal" ]]; then
    src="${work}/src-${REPO##*/}"
  else
    src="${work}/src"
  fi
  if [[ ! -d "$src/.git" ]]; then
    git clone "$REPO" "$src"
  fi
  # fetch all branches + tags from the fork (origin = the fork; upstream on the
  # fork clone stays fetch-only and is never touched here). Branch names such as
  # feat/p2-mixed-batch-lora live on refs/remotes/origin/* after this.
  git -C "$src" fetch --tags --force origin
  git -C "$src" reset --hard >/dev/null 2>&1 || true
  # Resolve the ref to a commit FIRST, then detach: `git checkout --detach <name>`
  # trips git's DWIM branch-creation for a name that only exists as a remote
  # branch (fatal: '--detach' cannot be used with '-b/-B/--orphan'), which is
  # exactly the branch-name case this port adds.
  #
  # Prefer the freshly-fetched origin ref for branch names: a clone's LOCAL
  # branch (e.g. master, created by `git clone`) goes stale as upstream moves,
  # and `git fetch` updates only refs/remotes/origin/* — a local branch would
  # silently shadow the fetched tip and rebuild the OLD commit (field finding
  # 2026-08-02: --ref master resolved 11924d4c1 instead of the fetched
  # 0ab9d6fed). Tags and shas are not under refs/remotes/origin/, so they fall
  # through to the plain-ref resolution below.
  resolved="$(git -C "$src" rev-parse --verify --quiet "refs/remotes/origin/${REF}^{commit}")" \
    || resolved="$(git -C "$src" rev-parse --verify --quiet "${REF}^{commit}")" \
    || { echo "!! cannot resolve --ref '$REF' to a commit in $src" >&2; exit 2; }
  git -C "$src" checkout --detach "$resolved"
  git -C "$src" reset --hard "$resolved" >/dev/null 2>&1 || true
  git -C "$src" clean -fd >/dev/null 2>&1 || true
  built_ref="$(git -C "$src" rev-parse --short HEAD)"
fi

residency_patch_sha=""
if [[ "$REF" != local:* && "$(git -C "$src" rev-parse HEAD)" == "ecc0cb0d34e17595ff0546b89c44fa100d076c0d" ]]; then
  residency_patch="${here}/patches/vulkan-residency.patch"
  residency_patch_sha="$(sha256sum "$residency_patch" | cut -d ' ' -f 1)"
  git -C "$src" apply --check "$residency_patch"
  git -C "$src" apply "$residency_patch"
fi

# --- p2fork: derive patch provenance from HEAD's history (fail-closed) -------
# The manifest records the ORIGINAL source patch's sha256 so the cross-repo
# contract (inferference's diagnostics.py) can tie this build to the P2 change.
# The value is not invented here: the fork's P2 commits carry "sha256 <hex>"
# provenance next to the patch path in their commit messages. A --no-ff merge
# (e.g. integration = anchor + P2 + nanbeige merge commits) moves the P2 commits
# into HEAD's history without changing their messages, so we search ALL of
# HEAD's history (newest-first) for a commit message that both names the P2
# patch and carries its sha256. This subsumes the old branch-tip behavior (a
# tip that IS a P2 commit still matches) and fixes the merged-tree case, where
# the tip is a merge commit with no P2 sha256 line of its own.
#
# Two independent checks, both fail-closed:
#   1. provenance: a P2 marker (patch path + sha256 line) reachable from HEAD;
#   2. surface:    the checked-out tree actually declares llama_set_seq_adapters
#                  in include/llama.h (the tree is the build input, so this is
#                  the direct proof the routing surface exists in THIS lib).
# A p2fork manifest must never claim seq_adapter_routing for a lib that lacks
# the routing symbols, so either failure exits 3.
patch_sha=""
if (( is_p2fork )); then
  patch_sha="$(git -C "$src" log --format=%B HEAD \
    | grep -A1 -m1 'p2-mixed-batch-lora\.patch' \
    | sed -n 's/^sha256 \([0-9a-f]\{64\}\)$/\1/p' \
    | head -1 || true)"
  if [[ -z "$patch_sha" ]]; then
    echo "!! p2fork: no P2 provenance found in HEAD's history (no commit message" >&2
    echo "!! naming p2-mixed-batch-lora.patch with a 'sha256 <hex>' line)." >&2
    echo "!! A p2fork build must include the P2 change (feature branch tip or a" >&2
    echo "!! merged tree). Refusing to claim seq_adapter_routing=true." >&2
    exit 3
  fi
  if ! grep -q "llama_set_seq_adapters" "${src}/include/llama.h"; then
    echo "!! p2fork: tree at ${built_ref} does not declare llama_set_seq_adapters" >&2
    echo "!! in include/llama.h — the routing surface is absent from THIS lib." >&2
    echo "!! Refusing to emit a manifest that claims seq_adapter_routing=true." >&2
    exit 3
  fi
  echo ">> p2fork: patch provenance sha256=${patch_sha} (P2 marker in HEAD history of ${built_ref})"
fi

# --- features: feat/* branches merged into this tree (additive manifest key) --
# The merged tree records which feature branches
# are folded into it. Derived from ancestry, never hardcoded: any fork feature
# branch (refs/remotes/origin/feat/*) whose tip is an ancestor of the built ref
# is listed. Auto-discovers new feat/* branches; [] for stock builds; skipped
# for local: trees (no origin refs to ask). Additive only — the cross-repo
# contract keys (ci/build/build-manifest.md) are unchanged.
# A local: tree is asked too. It was skipped before because a local path was
# assumed to have no origin refs, but the local: trees this repo builds are
# WORKTREES of src-p2fork and share its refs. Skipping them wrote features=[],
# which ci/runner/run-suite.sh reads as "stock build" and turns every
# fork-capability suite into a SKIP — a silent pass on a fork build. A tree with
# no origin refs still yields [], so nothing else changes.
features=()
while IFS= read -r br; do
  if git -C "$src" merge-base --is-ancestor "$br" HEAD 2>/dev/null; then
    features+=("\"${br#refs/remotes/origin/feat/}\"")
  fi
done < <(git -C "$src" for-each-ref --format='%(refname)' 'refs/remotes/origin/feat/*' 2>/dev/null | sort)
features_json="[$(IFS=,; echo "${features[*]}")]"

# --- flag profiles --------------------------
common=(
  -DCMAKE_BUILD_TYPE=Release
  -G Ninja
  -DBUILD_SHARED_LIBS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_SERVER=OFF
  -DGGML_NATIVE=ON
  -DGGML_CCACHE=ON
)
case "$PROFILE" in
  cpu-light)
    flags=( "${common[@]}" -DGGML_CUDA=OFF )
    ;;
  cuda-3060 | p2fork)
    # 3060 = compute capability 8.6 / sm_86. p2fork uses the identical CUDA flags
    # (ABI-compatible with the stock cuda-3060 lib); only the source tree differs
    # (the fork branch instead of a patch).
    flags=( "${common[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 )
    ;;
  cuda-3060-server)
    # Stock CUDA server: same CUDA flags as cuda-3060, plus the server. This is
    # the validation vehicle for stock tools/server/ fixes (upstream candidates),
    # which must be built against a stock tree, not a p2fork one. Mirrors
    # p2fork-server without the p2fork provenance check.
    flags=( "${common[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
            -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF )
    ;;
  p2fork-server)
    # Same CUDA flags as p2fork, plus the server. LLAMA_BUILD_SERVER=OFF stays in
    # common=() so no other profile changes; the later -D on the same cmake
    # command line wins. That last-wins assumption is VERIFIED after configure
    # against CMakeCache.txt below — never assumed.
    #
    # LLAMA_BUILD_SERVER=ON also pulls in tools/ui and tools/cli
    # (tools/CMakeLists.txt:24-28), and llama-server-impl links llama-ui
    # (tools/server/CMakeLists.txt:52). The UI provisioner runs npm when
    # LLAMA_BUILD_UI=ON and otherwise downloads prebuilt assets from a Hugging
    # Face bucket when LLAMA_USE_PREBUILT_UI=ON (scripts/ui-assets.cmake:305-350).
    # Both are network or toolchain dependencies this rig must not acquire at
    # build time, so both are OFF. llama-ui still builds, from an empty asset set
    # (scripts/ui-assets.cmake:353-360). The binary serves the HTTP API and has
    # no embedded web UI. That is what a headless consumer needs.
    flags=( "${common[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
            -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF )
    ;;
  vulkan-mi25 | vulkan-mi25-server | vulkan-mi25-p2fork-server)
    # AMD Instinct MI25 (Vega 10 / gfx900), two cards, capped at 110 W.
    #
    # WHY VULKAN AND NOT ROCM/HIP: ROCm dropped gfx900 after 5.7, so a HIP build
    # of this tree needs an out-of-support rocBLAS built for a target nixpkgs
    # does not ship. ggml's Vulkan backend runs on mesa's RADV driver, which
    # already enumerates both cards on this rig with no driver install. That is
    # what makes this a baseline spike; HIP is the optimization session's lever.
    #
    # No -DCMAKE_CUDA_ARCHITECTURES analogue exists: the Vulkan backend compiles
    # one SPIR-V shader set (glslc, from the devenv `shaderc` package) and picks
    # the device at RUNTIME, so one lib set serves any Vulkan GPU. The card is
    # selected with GGML_VK_VISIBLE_DEVICES / --device, not at build time.
    #
    # GGML_CUDA is pinned OFF rather than left unset: this rig still carries the
    # CUDA toolkit in devenv, and an auto-detected CUDA backend in a lib set
    # labelled "vulkan-mi25" would be a manifest that lies about what it runs on.
    flags=( "${common[@]}" -DGGML_CUDA=OFF -DGGML_VULKAN=ON )
    if (( build_server )); then
      # Same UI reasoning as the CUDA server profiles: no npm, no Hugging Face
      # asset download at build time. Headless HTTP API only.
      flags+=( -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF )
    fi
    ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

# Native build: make -march=native reach the compiler.
# The devenv shell sets NIX_ENFORCE_NO_NATIVE=1. The Nix gcc wrapper then drops
# -march=native, so GGML_NATIVE=ON gives no native code. Set it to 0 here.
# ccache may hash only the command line, not this variable (nvcc, for example),
# and so reuse old non-native objects. A ccache namespace adds a string to the
# hash. Native and non-native objects then never share an entry, and native
# builds stay cached.
if [[ " ${flags[*]} " == *" -DGGML_NATIVE=ON "* ]]; then
  export NIX_ENFORCE_NO_NATIVE=0
  export CCACHE_NAMESPACE="native"
fi

artifact_ref="${built_ref}${residency_patch_sha:+-residency-${residency_patch_sha:0:12}}"
: "${OUT:=${work}/out-${PROFILE}-${artifact_ref}}"
build="${work}/build-${PROFILE}-${artifact_ref}"
# Clean build dir each run: incremental builds risk stale state across refs
# Persistent CCACHE_DIR is what makes the wipe free —
# GGML_CCACHE=ON auto-wires ccache and cached nvcc objects survive the rm -rf.
rm -rf "$build"; mkdir -p "$build" "${OUT}/lib"

# --- stable compile path (this is what makes ccache work) ---------------------
# ccache keys every object on the absolute source path, so a build from a new
# worktree missed the whole CUDA cache and paid for it. Measured 2026-08-25:
# CUDA is 96.2% of a cold build (5639 s CPU over 141 targets, 737 s wall) and
# 0.7% of a warm one (3.7 s, 70 s wall). One ggml-cuda TU costs ~50 s, and the
# fork branches only ever touch src/llama-*.cpp — so every build should be warm.
#
# Compile through one stable symlink and every ref, worktree and profile shares
# a single object cache. cmake records the path as given and does not resolve
# the link (verified: 882 symlink paths and 0 real paths in
# compile_commands.json). CCACHE_BASEDIR does NOT substitute for this: it
# rewrites absolute paths to CWD-relative ones, and two worktrees have genuinely
# different relative paths, so it still misses (measured).
#
# The link is shared state, so hold a lock across the whole build. Two builds
# running at once would otherwise repoint it under each other and silently
# compile the wrong tree.
src_link="${work}/src-active"
exec 9>"${work}/.src-active.lock"
if ! flock -w 3600 9; then
  echo "!! another build holds ${work}/.src-active.lock and did not release it" >&2
  echo "!! within an hour. Refusing to repoint the shared compile path." >&2
  exit 3
fi
ln -sfn "$src" "$src_link"
# Only the compile goes through the link. Git provenance, the header copy and
# the server grep keep using the real path, so nothing recorded depends on where
# the link happens to point.
#
# Record the real tree: a build dir left on disk holds symlink paths, so a later
# manual `ninja -C <build dir>` would compile whatever src-active points at THEN,
# not what produced that dir. The script itself is safe — it wipes and
# reconfigures the build dir on every run, above.
echo "$src" > "${build}/SOURCE-TREE"

echo ">> building llama.cpp ref=${built_ref} (requested ${REF}) profile=${PROFILE}"
echo ">> flags: ${flags[*]}"
echo ">> compile path: ${src_link} -> ${src}"

cmake -S "$src_link" -B "$build" "${flags[@]}"

# --- verify the server flag actually took (fail-closed) -----------------------
# p2fork-server passes -DLLAMA_BUILD_SERVER twice on one command line (OFF from
# common=(), then ON from the profile arm) and relies on last-wins. Read the
# cache back rather than trust it: a silently-OFF server would produce a green
# build with no binary, which the collect step below would then have to guess at.
if (( build_server )); then
  if ! grep -qx 'LLAMA_BUILD_SERVER:BOOL=ON' "${build}/CMakeCache.txt"; then
    echo "!! ${PROFILE}: LLAMA_BUILD_SERVER is not ON in ${build}/CMakeCache.txt." >&2
    echo "!! The duplicate -D last-wins assumption does not hold on this cmake." >&2
    echo "!! Refusing to build a profile that would silently omit the server." >&2
    exit 3
  fi
fi

# Build the shared llama library target only. The default `all` target also
# builds the `app/llama` executable; with LLAMA_BUILD_SERVER=OFF that executable
# can still reference the disabled llama-server-impl/llama-cli-impl targets and
# fail after all required libraries have already built.
# The same reasoning names the targets explicitly for p2fork-server: `all` is
# never built, so the server is added to the list rather than switched on.
targets=( llama )
(( build_server )) && targets+=( llama-server )
cmake --build "$build" --target "${targets[@]}" --config Release -j"$(nproc)"

# --- collect the shared-lib set ----------------------------------------------
# llama-cpp-python loads libllama + libggml{,-base,-cpu} (+ libmtmd if present).
find "$build" -name 'libllama.so*' -o -name 'libggml*.so*' -o -name 'libmtmd.so*' \
  | while read -r f; do cp -av "$f" "${OUT}/lib/"; done

# --- collect the server binary (p2fork-server only) ---------------------------
# The binary goes in <out>/bin/ beside <out>/lib/, so the manifest at
# <out>/build-manifest.json stays the lib dir's parent — the exact layout
# diagnostics.py expects (ci/build/build-manifest.md:15-18).
#
# The lib set above is the one llama-cpp-python loads. llama-server needs two
# more that no Python consumer does: libllama-common (the `common/` layer) and
# libllama-server-impl (the server logic; llama-server itself is a 16 KiB
# main.cpp shim, tools/server/CMakeLists.txt:57-63). Collect them too, or the
# copied binary is not an artifact — it is a pointer into the build tree.
#
# cmake writes a build-tree RUNPATH: an ABSOLUTE path into
# .llamacpp-builds/build-<profile>-<ref>/bin. That directory is prunable
# (ci/build/prune-builds.sh keeps only the newest build-* per profile), so the
# collected binary would break on the next rebuild of any other ref. Prepend
# $ORIGIN-relative entries so <out>/ is self-contained, and KEEP the existing
# entries so the nix-store glibc/gcc/CUDA paths still resolve.
server_built=false
if (( build_server )); then
  srv="$(find "$build" -type f -name 'llama-server' -perm -u+x | head -1)"
  if [[ -z "$srv" ]]; then
    echo "!! ${PROFILE}: build succeeded but no llama-server binary was produced" >&2
    echo "!! under ${build}. Refusing to emit a manifest claiming server_built." >&2
    exit 3
  fi
  find "$build" -name 'libllama-common.so*' -o -name 'libllama-server-impl.so*' \
    | while read -r f; do cp -av "$f" "${OUT}/lib/"; done
  mkdir -p "${OUT}/bin"
  cp -av "$srv" "${OUT}/bin/"

  if ! command -v patchelf >/dev/null 2>&1; then
    echo "!! ${PROFILE}: patchelf not found. The collected binary would keep an" >&2
    echo "!! absolute RUNPATH into the prunable build garage. Refusing to emit a" >&2
    echo "!! manifest claiming server_built for a non-relocatable artifact." >&2
    exit 3
  fi
  old="$(patchelf --print-rpath "${OUT}/bin/llama-server")"
  patchelf --set-rpath "\$ORIGIN/../lib:${old}" "${OUT}/bin/llama-server"
  for so in "${OUT}"/lib/*.so*; do
    [[ -f "$so" && ! -L "$so" ]] || continue
    old_so="$(patchelf --print-rpath "$so" 2>/dev/null || true)"
    patchelf --set-rpath "\$ORIGIN:${old_so}" "$so" 2>/dev/null || true
  done

  # Prove the collect, do not assert it: resolve the binary's libraries with the
  # build tree hidden from the loader. A single "not found" means the artifact
  # is incomplete, and an incomplete artifact must never ship a green manifest.
  missing="$(env -u LD_LIBRARY_PATH ldd "${OUT}/bin/llama-server" 2>/dev/null \
             | grep 'not found' || true)"
  if [[ -n "$missing" ]]; then
    echo "!! ${PROFILE}: the collected llama-server has unresolved libraries:" >&2
    echo "$missing" >&2
    exit 3
  fi
  if env -u LD_LIBRARY_PATH ldd "${OUT}/bin/llama-server" 2>/dev/null | grep -q "${build}/"; then
    echo "!! ${PROFILE}: the collected llama-server still resolves libraries from" >&2
    echo "!! the prunable build tree ${build}. The artifact is not self-contained." >&2
    exit 3
  fi
  server_built=true
fi

# Keep the matching public headers beside the library set so cffi API-mode
# bindgen can compile against exactly the source that produced these .so files.
mkdir -p "${OUT}/include"
cp -a "${src}/include/." "${OUT}/include/"
cp -a "${src}/ggml/include/." "${OUT}/include/"

# --- build manifest (cross-repo contract; see ci/build/build-manifest.md) -----
# Record the versioned soname, never the unversioned link. A glob sorts
# libggml.so.0 ahead of libggml.so.0.18.0, so the old `ls ... | head -1` named
# the link in every manifest written before 2026-08-25. `sort -V | tail -1`
# takes the most specific name. Sort rather than resolve the link, because
# mode-a-build.sh makes the same selection over a wheel, whose bundled libs are
# real files rather than symlinks.
# The `|| true` is load-bearing: `set -euo pipefail` makes the no-match `ls`
# abort the assignment, which would leave the check below unreachable — the same
# dead-fallback defect this change removes.
ggml_lib_path="$(ls "${OUT}"/lib/libggml.so.* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "$ggml_lib_path" ]]; then
  echo "!! ${PROFILE}: no libggml.so.* under ${OUT}/lib. Refusing to emit a" >&2
  echo "!! manifest for a lib set with no ggml library." >&2
  exit 3
fi
ggml_ver="$(basename "$ggml_lib_path")"
if (( is_p2fork )); then seq_routing=true; else seq_routing=false; fi
# server_seq_routing: does the server binary in THIS build call the routing API?
# Derived from the compiled source, never from the profile name — the same
# fail-closed shape as the include/llama.h check above. A stock tools/server/
# tree gives false even on the p2fork-server profile, which is exactly the S4
# stage-1 answer; it flips to true only when the stage-4 edits land.
server_seq_routing=false
if [[ "$server_built" == true ]] \
   && grep -rq 'llama_set_seq_adapter' "${src}/tools/server/"; then
  server_seq_routing=true
fi
# p2fork: the "patch" key records provenance (the P2 change is commits, not a
# patch file); the value is informational — the reader (diagnostics.py) never
# reads "patch", only patch_sha256 / seq_adapter_routing.
p2_provenance="commits on feat/p2-mixed-batch-lora (P2 patch provenance sha256=${patch_sha}; no patch applied)"
if (( is_p2fork )); then patch_note="$(printf '"%s"' "${p2_provenance}")"; else patch_note='""'; fi
cat > "${OUT}/build-manifest.json" <<JSON
{
  "repo": "${REPO}",
  "ref_requested": "${REF}",
  "ref_built": "${built_ref}",
  "profile": "${PROFILE}",
  "cmake_flags": "${flags[*]}",
  "ggml_lib": "${ggml_ver}",
  "ggml_version": "${ggml_ver}",
  "llama_cpp_commit": "${built_ref}",
  "build_id": "${PROFILE}-${artifact_ref}",
  "vulkan_residency_patch_sha256": "${residency_patch_sha}",
  "features": ${features_json},
  "patch": ${patch_note},
  "patch_sha256": "${patch_sha}",
  "seq_adapter_routing": ${seq_routing},
  "server_built": ${server_built},
  "server_seq_routing": ${server_seq_routing}
}
JSON

echo ">> done. lib set in ${OUT}/lib"
echo ">> Mode B:  export LLAMA_CPP_LIB_PATH=${OUT}/lib"
if [[ "$server_built" == true ]]; then
  echo ">> server:  ${OUT}/bin/llama-server (server_seq_routing=${server_seq_routing})"
  echo ">> Mode B does not apply to an executable — run this binary directly."
fi
echo ">> Next:    run the ABI smoke gate, then the cffi bindgen gate."
