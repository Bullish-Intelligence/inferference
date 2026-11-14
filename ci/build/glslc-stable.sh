#!/usr/bin/env bash
# glslc wrapper that gives glslc argument paths of a fixed length.
#
# Why: glslc leaves unused bits unset in a few SPIR-V constants. The bits depend
# on the length of the input and output paths in its arguments. The shader
# `tri_f16` showed it: the same source built in two lanes gave two different
# `OpConstant` words (the upper half of an f16 zero), and so two different
# libggml-vulkan.so files. The lane name sets the path length. The wrapper
# links the shader dir and the output dir under a short temp dir with a name of
# fixed length. glslc then sees the same argument lengths in every lane.
#
# build-llamacpp.sh passes this script as -DVulkan_GLSLC_EXECUTABLE and exports
# GLSLC_REAL (the real glslc). Any call that is not "glslc ... IN -o OUT" (the
# cmake feature probes write to stdout with `-o -`) runs the real glslc as is.
set -euo pipefail

real="${GLSLC_REAL:?GLSLC_REAL names the real glslc}"

# The wrapper changes directory before it runs glslc. Resolve every path first, while the
# caller's directory still applies. A path that does not exist stays as given.
_abs() {
  local p="$1"
  if [[ "$p" == /* ]]; then
    printf '%s' "$p"
  elif [[ -d "$p" ]]; then
    (cd "$p" && pwd -P)
  elif [[ -e "$p" ]]; then
    printf '%s/%s' "$(cd "$(dirname "$p")" && pwd -P)" "$(basename "$p")"
  else
    printf '%s' "$p"
  fi
}
# A real glslc named with a slash may be relative. A bare name is a PATH lookup.
[[ "$real" == */* ]] && real="$(_abs "$real")"

in="" out="" dep=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    -o)  out="${args[i + 1]:-}"; i=$((i + 1)) ;;
    -MF) dep="${args[i + 1]:-}"; i=$((i + 1)) ;;
    -*)  ;;
    *)   [[ -f "${args[i]}" ]] && in="${args[i]}" ;;
  esac
done
if [[ -z "$in" || -z "$out" || "$out" == "-" ]]; then
  exec "$real" "$@"
fi

tmp="$(mktemp -d /tmp/glslc-stable.XXXXXXXXXX)"
trap 'rm -rf "$tmp"' EXIT
in_dir="$(dirname "$in")"
out_dir="$(dirname "$out")"
ln -s "$(cd "$in_dir" && pwd -P)" "$tmp/s"
ln -s "$(cd "$out_dir" && pwd -P)" "$tmp/o"

new=()
for ((i = 0; i < ${#args[@]}; i++)); do
  a="${args[i]}"
  if [[ "$a" == "$in" ]]; then
    new+=("$tmp/s/$(basename "$in")")
  elif [[ "$a" == "-o" ]]; then
    new+=("$a" "$tmp/o/$(basename "$out")"); i=$((i + 1))
  elif [[ "$a" == "-MF" ]]; then
    new+=("$a" "$tmp/o/$(basename "$dep")"); i=$((i + 1))
  elif [[ "$a" == -I ]]; then
    new+=("$a" "$(_abs "${args[i + 1]:-}")"); i=$((i + 1))
  elif [[ "$a" == -I?* ]]; then
    new+=("-I$(_abs "${a#-I}")")
  elif [[ "$a" == -* ]]; then
    new+=("$a")
  else
    new+=("$(_abs "$a")")
  fi
done

(cd "$tmp" && "$real" "${new[@]}")

# glslc wrote through the link, so the files are already in the real dirs. The
# dependency file names the temp links; point them at the real dirs again.
if [[ -n "$dep" && -f "$dep" ]]; then
  real_in="$(cd "$in_dir" && pwd -P)"
  real_out="$(cd "$out_dir" && pwd -P)"
  sed -i "s|$tmp/s|$real_in|g; s|$tmp/o|$real_out|g" "$dep"
fi
