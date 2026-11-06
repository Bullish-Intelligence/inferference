# build-manifest.json — schema (cross-repo contract)

`ci/build/build-llamacpp.sh` emits `build-manifest.json` next to every lib set
(`<out>/build-manifest.json`, where `<out>` is `out-<profile>-<ref_built>`).
The keys are a **contract**: the library's diagnostics reader
(`src/inferference/diagnostics.py`) reads a fixed set of them. Keys are **stable and additive only** — never rename
or remove a key.

## How the contract is read

The reader, `src/inferference/diagnostics.py`:

- `_build_manifest(library_path)` (diagnostics.py:39-52) — given
  `LLAMA_CPP_LIB_PATH`, looks for `build-manifest.json` in the lib dir's
  **parent** first, then the lib dir itself. So the manifest lives at
  `<out>/build-manifest.json` and the lib at `<out>/lib/`.
- `_text(manifest, *names)` (diagnostics.py:54-60) — first non-`None` key wins.
- `_flag(manifest, name)` (diagnostics.py:62-65) — `bool(value)`.
- Resolved values (diagnostics.py:78-81):
  - `llama_cpp_commit` ← `_text(manifest, "llama_cpp_commit", "commit", "ref")`
  - `llama_cpp_build_id` ← `_text(manifest, "build_id", "profile")`
  - `seq_adapter_routing` ← `_flag(manifest, "seq_adapter_routing")`
  - `ggml_version` ← `_text(manifest, "ggml_version")`

The emitter is `ci/build/build-llamacpp.sh`.

## Keys

| key | type | meaning | read by diagnostics? |
| --- | --- | --- | --- |
| `repo` | string | git remote the source tree came from (defaults to the fork `https://github.com/Bullish-Design/llama-infernal`; `--repo` overrides) | no |
| `ref_requested` | string | the `--ref` value as given (branch name, tag, sha, or `local:path`) | no |
| `ref_built` | string | short sha of the checked-out tip actually built (a branch ref resolves to its tip) | no |
| `profile` | string | build profile: `cpu-light` / `cuda-3060` / `cuda-3060-server` / `p2fork` / `p2fork-server` | no (but is the `build_id` fallback, below) |
| `cmake_flags` | string | the full cmake flag set used | no |
| `ggml_lib` | string | basename of the built `libggml.so.<ver>` (e.g. `libggml.so.0.16.0`) | no |
| `ggml_version` | string | same as `ggml_lib` — **read** | **yes** |
| `llama_cpp_commit` | string | short sha of the built commit — **read** (fallback keys `commit` / `ref` are NOT emitted) | **yes** |
| `build_id` | string | `"<profile>-<ref_built>"` — **read** (falls back to `profile`) | **yes** |
| `patch` | string | provenance note. `""` for stock profiles; for `p2fork`, records that the P2 change is commits on `feat/p2-mixed-batch-lora` (no patch file) with the provenance sha256 | no |
| `patch_sha256` | string | sha256 of the original source patch. For `p2fork` it is **derived from the P2 provenance marker anywhere in the built ref's history** (not just the tip): each P2 commit message on `feat/p2-mixed-batch-lora` carries `sha256 8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752` next to the patch path, and a `--no-ff` merge (integration) keeps those commits (and their messages) reachable from HEAD. The build script searches HEAD's history for a message naming `p2-mixed-batch-lora.patch` with a `sha256 <hex>` line (branch-tip refs still match — the tip is in its own history), **and** verifies the checked-out tree declares `llama_set_seq_adapters` in `include/llama.h` as a second, direct check. It **fails closed (exit 3)** if either fails — a p2fork manifest must never claim routing against a lib without the symbols. `""` for stock profiles | no |
| `seq_adapter_routing` | bool | `true` iff the lib is a P2 fork build. **`p2fork` profile ⇒ `true`; `cpu-light` / `cuda-3060` ⇒ `false`** — explicit per profile, never inferred from `patch_sha256` presence (in the reference it was `true` iff the P2 patch was applied; in this repo the P2 change is commits, so the flag is profile-derived) | **yes** |
| `features` | list[string] | the fork feature branches folded into this tree — one entry per `feat/*` branch, e.g. `["p2-mixed-batch-lora"]`. **Derived from ancestry**, never hardcoded: every `refs/remotes/origin/feat/*` branch whose tip is an ancestor of the built ref is listed (auto-discovers new branches; `[]` for stock builds; skipped for `local:` trees). Additive key — no reader depends on it yet; a future diagnostics reader may probe it to select the library's capability surface | no |
| `server_built` | bool | `true` iff this build produced `<out>/bin/llama-server`. Only the `*-server` profiles (`p2fork-server`, `cuda-3060-server`) do; every other profile emits `false`. The build fails closed (exit 3) if it claims to build the server and then finds no binary, so `true` always means the file exists beside the lib set | no |
| `server_seq_routing` | bool | `true` iff the server binary in **this build** calls the fork's per-sequence routing API. **Derived from the compiled source**, not from the profile name: the script greps `tools/server/` for `llama_set_seq_adapter` in the tree it just built, the same fail-closed shape as the `include/llama.h` check that guards `seq_adapter_routing`. A `p2fork-server` build of a stock `tools/server/` therefore emits `false` — which is the correct answer for 005-server-seq-routing stage 1. It flips to `true` only when the routed-server edits land. `server_built=false` forces `false` | no |

### `ggml_lib` / `ggml_version` named the unversioned link until 2026-08-25

The emitter took the first glob match — `ls "${OUT}"/lib/libggml.so.* | head -1`
— and a glob sorts `libggml.so.0` ahead of `libggml.so.0.18.0`. The first match
is the symlink, so **every manifest built before 2026-08-25 records
`libggml.so.0`**, not the versioned basename this table documents. All nine
manifests in the build garage carry the wrong value.

The emitter now resolves the `libggml.so` link instead of sorting the glob, and
fails closed when the link is absent. The old form could not fail: `ls` writes
nothing to stdout on no match and `head` still exits 0, so the `|| echo unknown`
fallback was unreachable and a missing library produced an empty string.

Readers are unaffected. `diagnostics.py` passes `ggml_version` through as text
and never parses the version out of it.

## Byte-compatibility rules

1. The four reader keys (`llama_cpp_commit`, `build_id`, `seq_adapter_routing`,
   `ggml_version`) must always be present and typed as above. The fallbacks
   `commit` / `ref` are reader conveniences only — this repo emits
   `llama_cpp_commit` directly.
2. The manifest is emitted at `<out>/build-manifest.json` (parent of `<out>/lib`),
   exactly where `_build_manifest` looks first.
3. `features` is additive-only (added 2026-08-02 for the merged tree); the four
   reader keys above are byte-identical before/after.
3. `seq_adapter_routing` semantics: the P2 mixed-batch per-sequence LoRA routing
   capability is a property of the *build*, not the tree. `p2fork` ⇒ `true`
   (the fork's `feat/p2-mixed-batch-lora` branch exports
   `llama_set_seq_adapters` / `llama_set_seq_adapter`, include/llama.h:699-712).
   Any other profile ⇒ `false`, even if `--ref` points at a branch that happens
   to contain the P2 commits — the manifest records what the profile built.
4. `ref_built` vs `ref_requested`: both are emitted so a reader can tell what
   was asked for vs what the tree resolved to (a branch name resolves to its tip
   short sha). Additive-only means we may add new keys later; we may never
   rename or drop existing ones.
5. `server_built` / `server_seq_routing` are additive-only (added 2026-08-24 for
   005-server-seq-routing spike S4). Every existing key is byte-identical before
   and after, and every profile other than `p2fork-server` emits `false` for
   both. `seq_adapter_routing` describes the **library**; `server_seq_routing`
   describes the **server binary**. The two are independent: a `p2fork-server`
   build of a stock `tools/server/` emits `seq_adapter_routing: true` and
   `server_seq_routing: false`, which reads as "the lib can route, the server
   does not ask it to".

## The `p2fork-server` profile

`cuda-3060-server` is the stock-tree sibling of `p2fork-server`: the same CUDA
flags and server target list, but the `cuda-3060` family (`is_p2fork=0`), so no
P2 provenance check and `seq_adapter_routing: false`. It exists to build and
validate stock `tools/server/` / `common/` fixes (upstream candidates) against a
stock tree.

`p2fork-server` is a member of the `p2fork` family. It shares the `src-p2fork`
checkout, the `feat/p2-mixed-batch-lora` ref default, both fail-closed
provenance checks, and `seq_adapter_routing: true`. It differs in three ways:

- `-DLLAMA_BUILD_SERVER=ON` is appended after the `common=()` entry that sets it
  `OFF`, so no other profile changes. The script reads `CMakeCache.txt` back and
  exits 3 if the flag did not take — the last-wins behaviour is verified, never
  assumed.
- `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF`. `LLAMA_BUILD_SERVER=ON`
  pulls in `tools/ui`, whose provisioner runs `npm` or downloads prebuilt assets
  from a Hugging Face bucket. Neither belongs in a rig build. The binary serves
  the HTTP API with no embedded web UI.
- The build target list is `llama llama-server`, named explicitly. `all` is never
  built, for the reason already recorded in `build-llamacpp.sh`.

The binary lands at `<out>/bin/llama-server`, beside `<out>/lib/`, so
`<out>/build-manifest.json` stays the lib dir's parent — the layout
`_build_manifest` looks in first (see "How the contract is read" below).

## Example (p2fork build of integration with both features merged, 2026-08-02)

```json
{
  "repo": "https://github.com/Bullish-Design/llama-infernal",
  "ref_requested": "integration",
  "ref_built": "2dad55136",
  "profile": "p2fork",
  "cmake_flags": "-DCMAKE_BUILD_TYPE=Release -G Ninja -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON -DGGML_CCACHE=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86",
  "ggml_lib": "libggml.so.0.16.0",
  "ggml_version": "libggml.so.0.16.0",
  "llama_cpp_commit": "2dad55136",
  "build_id": "p2fork-2dad55136",
  "features": ["p2-mixed-batch-lora"],
  "patch": "commits on feat/p2-mixed-batch-lora (P2 patch provenance sha256=8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752; no patch applied)",
  "patch_sha256": "8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752",
  "seq_adapter_routing": true
}
```

`patch_sha256` above is derived from the P2 provenance marker found in HEAD's
history (the integration merge brings the P2 commits' messages along), and the
`features` list is derived from branch ancestry — both fail-closed, documented
above. The two `libggml.so.0.16.0` values show the contract, not that build's
output: the 2026-08-02 build recorded `libggml.so.0`, per the note above. A
`diagnostics.py`-compatible reader resolves, from this manifest:

| field | value |
| --- | --- |
| `llama_cpp_commit` | `fa8cab171` |
| `llama_cpp_build_id` | `p2fork-fa8cab171` |
| `seq_adapter_routing` | `True` |
| `ggml_version` | `libggml.so.0.16.0` |

## The shared compile path — `src-active`

Every build compiles through `ci/build/.llamacpp-builds/src-active`, a symlink
`build-llamacpp.sh` repoints at the requested tree, under a lock held for the
whole build. ccache keys each object on the absolute source path, so building a
new worktree directly used to miss the entire CUDA cache.

Measured 2026-08-25, same profile, a **different worktree** each time:

| | per-worktree path | through `src-active` |
| --- | --- | --- |
| wall | 737 s | **69.0 s** |
| CUDA CPU time | 5639 s over 141 targets | **5.5 s over 144 targets** |
| CUDA share of build | 96.2% | 1.0% |

Changing the key costs one cold build (783.6 s, measured). After that, every
ref, worktree and profile shares one object cache.

Two consequences to know:

- Each build dir records its real source tree in `<build dir>/SOURCE-TREE`. The
  build dir's ninja files hold `src-active` paths, so a **manual** `ninja -C
  <build dir>` compiles whatever the link points at then, not what produced that
  dir. The script itself is safe — it wipes and reconfigures every run.
- `prune-builds.sh` skips the symlink, so it never counts as a stale clone.

## Housekeeping

`ci/build/.llamacpp-builds/` is the gitignored build garage and grows without
bound (each new ref/profile adds an `out-<profile>-<sha>` + `build-<profile>-<sha>`
pair; each new `--repo` adds a `src-<repo>` clone; the clones grow as upstream
history grows — "curate or the disk dies" warning).

Run `ci/build/prune-builds.sh` (or `--dry-run` first) to prune. Policy:

- `out-*` — keep the **newest per profile** (the Mode B lib-swap targets and
  manifest evidence); delete the rest.
- `build-*` — keep the **newest per profile** (cmake cache); delete the rest
  (rebuilds are cheap via the persistent `CCACHE_DIR`).
- `src/` + `src-p2fork/` — always kept (the build script reuses them via
  fetch+reset).
- other `src-*` clones (the shared stock upstream clone, e.g. `src-llama.cpp`)
  — keep the single newest as reference; delete stale ones.
