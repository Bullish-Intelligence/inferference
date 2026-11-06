#!/usr/bin/env bash
# Read the driver's GTT counter and reject an increase.

gtt_read_bytes() {
  local value
  value="$(cat "$1")" || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

gtt_did_not_increase() {
  [[ "$1" =~ ^[0-9]+$ && "$2" =~ ^[0-9]+$ ]] || return 1
  (( $2 <= $1 ))
}
