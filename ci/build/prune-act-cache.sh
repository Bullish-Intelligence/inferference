#!/usr/bin/env bash
# Housekeeping for the Act workdir cache (~/.cache/act) — the step-6 companion
# to prune-builds.sh.
#
# Why: every `act -j <job>` run materializes a fresh checkout under
# ~/.cache/act/<hash>/ and keeps it for reuse. Each workdir carries the repo
# (including gitignored build garages + venvs when the job builds them), so the
# cache grows per job and the disk hits ~98% without anyone noticing (observed
# at step-5 end; the session trimmed to the 2 newest by hand).
#
# Policy:
#   * keep the N newest workdirs (default 2 — the step-5 convention), delete
#     the rest.
#   * NEVER run this mid-job: act workdirs are in use while a job runs. Run it
#     outside act (between sessions), like prune-builds.sh.
#   * The bench baselines store is OUTSIDE the workdirs
#     ($BASELINES_ROOT in rig-env.sh) — trimming workdirs never touches
#     baselines, artifacts are re-emitted by the next run.
#
# Usage:
#   ci/build/prune-act-cache.sh [N]        # keep the N newest (default 2)
#   ci/build/prune-act-cache.sh --dry-run  # show what would be deleted
set -euo pipefail

keep="${1:-2}"
dry=0
[[ "${1:-}" == "--dry-run" ]] && { dry=1; keep=2; }
case "$keep" in
  ''|*[!0-9]*) echo "!! keep count must be a number" >&2; exit 2 ;;
esac

cache="${ACT_CACHE_DIR:-${HOME}/.cache/act}"
if [[ ! -d "${cache}" ]]; then
  echo "no ${cache}; nothing to prune"
  exit 0
fi

mapfile -t dirs < <(find "${cache}" -mindepth 1 -maxdepth 1 -type d -name '[0-9a-f]*' | xargs -r ls -dt)
total=${#dirs[@]}
echo "== prune act cache ${cache} (${total} workdirs; keeping ${keep})"

deleted=0
freed=0
for ((i = keep; i < total; i++)); do
  d="${dirs[$i]}"
  if [[ "${dry}" == "1" ]]; then
    echo "  would delete ${d}"
  else
    sz="$(du -sb "${d}" 2>/dev/null | cut -f1 || echo 0)"
    rm -rf "${d}"
    freed=$((freed + sz))
    deleted=$((deleted + 1))
    echo "  deleted ${d}"
  fi
done

if [[ "${dry}" == "1" ]]; then
  echo "(dry run — nothing deleted)"
else
  echo "deleted ${deleted} workdir(s); freed $(numfmt --to=iec "${freed}" 2>/dev/null || echo "${freed} bytes"); $(du -sh "${cache}" | cut -f1) after"
fi
