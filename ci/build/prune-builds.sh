#!/usr/bin/env bash
# Housekeeping for ci/build/.llamacpp-builds — the gitignored build "garage".
#
# Policy (curate, or the disk fills quietly mid-benchmark):
#
#   * out-<profile>-<sha>: keep the NEWEST out dir per profile
#     (p2fork / cpu-light / cuda-3060), delete the rest. The out dirs are the
#     Mode B lib-swap targets + build-manifest.json evidence; the newest is
#     what the next gate run will use. (If you are mid-experiment pointing
#     LLAMA_CPP_LIB_PATH at an older out dir, don't run this.)
#   * build-<profile>-<sha>: keep the NEWEST build dir per profile (the cmake
#     cache "just in case"), delete the rest. Pure cache — a rebuild is cheap
#     anyway via the persistent CCACHE_DIR (~/.cache/llamacpp-ccache).
#   * src/ and src-p2fork/ (the fork build sources): ALWAYS kept — the build
#     script reuses them via fetch+reset; deleting forces a slow re-clone.
#   * src-active: the shared compile path (a symlink, see build-llamacpp.sh).
#     Never deleted; it is not a clone.
#   * git worktrees (src-p2fork-*): ALWAYS kept, and only listed. A worktree
#     holds a checked-out branch plus any uncommitted work on it, and this
#     script cannot tell a throwaway from a live one. Use `git worktree remove`.
#   * any other src-* clone (the stock upstream clone, e.g. src-llama.cpp):
#     keep the single NEWEST one as reference, delete the rest.
#
# Usage:
#   ci/build/prune-builds.sh            # prune
#   ci/build/prune-builds.sh --dry-run  # show what would be deleted
set -euo pipefail

dry=0
[[ "${1:-}" == "--dry-run" ]] && dry=1

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="${here}/.llamacpp-builds"
[[ -d "$work" ]] || { echo "no ${work}; nothing to prune"; exit 0; }

cd "$work"

deleted=0
freed=0

announce() { # $1 = path being deleted
  if [[ $dry -eq 1 ]]; then
    echo "  would delete $1"
  else
    local sz
    sz="$(du -sb "$1" 2>/dev/null | cut -f1 || echo 0)"
    rm -rf "$1"
    freed=$((freed + sz))
    deleted=$((deleted + 1))
    echo "  deleted $1"
  fi
}

# keep the newest dir per profile (the token between the prefix and the last '-')
keep_newest_per_profile() { # $1 = prefix (out- / build-)
  local prefix="$1" dir profile
  local -A kept=()
  # ls -dt sorts newest-first; the first dir seen per profile is the keeper
  while IFS= read -r dir; do
    dir="${dir%/}"
    profile="${dir#${prefix}}"
    profile="${profile%-*}"
    [[ -n "$profile" && "$dir" == "${prefix}"* ]] || continue
    if [[ -z "${kept[$profile]:-}" ]]; then
      kept[$profile]=1
    else
      announce "$dir"
    fi
  done < <(ls -dt ${prefix}*-*/ 2>/dev/null || true)
}

keep_newest_stock_clone() {
  # keep src/ and src-p2fork/ (primary build sources) ALWAYS, plus the single
  # NEWEST other src-* clone (the shared stock upstream clone, e.g. src-llama.cpp)
  local -a others=() worktrees=() d
  for d in src-*/; do
    [[ -d "$d" ]] || continue
    d="${d%/}"
    # src-active is the shared compile path: a symlink build-llamacpp.sh
    # repoints per build, for the ccache key. Not a clone; never delete it.
    if [[ -L "$d" ]]; then
      continue
    fi
    # A linked git worktree carries .git as a FILE holding "gitdir: <path>";
    # a clone carries it as a directory. Worktrees hold a checked-out branch
    # and any uncommitted work on it, and this script cannot tell a throwaway
    # from a live one. Report and keep.
    if [[ -f "$d/.git" ]]; then
      worktrees+=("$d")
      continue
    fi
    case "$d" in src|src-p2fork) ;; *) others+=("$d") ;; esac
  done
  if [[ ${#others[@]} -gt 1 ]]; then
    local newest
    newest="$(printf '%s\n' "${others[@]}" | xargs -r ls -dt | head -1)"
    for d in "${others[@]}"; do
      [[ "$d" != "$newest" ]] && announce "$d"
    done
  else
    echo "  (no stale stock clones)"
  fi
  if [[ ${#worktrees[@]} -gt 0 ]]; then
    echo "  keeping ${#worktrees[@]} git worktree(s) — never pruned automatically:"
    printf '    %s\n' "${worktrees[@]}"
    echo "    to remove one: git -C src-p2fork worktree remove <path>"
  fi
}

echo "== prune .llamacpp-builds ($(du -sh . | cut -f1) before)"
echo "-- out-* (keep newest per profile)"
keep_newest_per_profile "out-"
echo "-- build-* (keep newest per profile)"
keep_newest_per_profile "build-"
echo "-- src-* (keep src, src-p2fork, and the newest stock clone)"
keep_newest_stock_clone

echo "== done"
if [[ $dry -eq 1 ]]; then
  echo "(dry run — nothing deleted)"
else
  echo "deleted $deleted dir(s); freed $(numfmt --to=iec "$freed" 2>/dev/null || echo "$freed bytes"); $(du -sh . | cut -f1) after"
fi
