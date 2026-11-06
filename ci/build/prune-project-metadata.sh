#!/usr/bin/env bash
# Remove source-management metadata from a generated vendor tree.
set -euo pipefail

if [[ $# -ne 1 || ! -d "$1" ]]; then
  echo "usage: $0 <source-tree>" >&2
  exit 2
fi

source_tree="$(realpath -e "$1")"
[[ "$source_tree" != "/" ]] || { echo "refusing to prune /" >&2; exit 2; }

find "$source_tree" -type d -name .agents -prune -exec rm -rf -- {} +
find "$source_tree" -type f \
  \( -name '.copier-answers*.yml' -o -name '.copier-answers*.yaml' \) \
  -delete
