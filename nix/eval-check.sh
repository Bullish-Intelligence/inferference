#!/usr/bin/env bash
# Evaluate the Nix outputs of a flake repository. Build nothing.
#
# Usage: nix-eval-check.sh [DIR]
#
# DIR defaults to $DEVENV_ROOT, then to the current directory. The script
# forces the drvPath of every packages, checks and devShells output of the
# current system. It lists the attribute names of homeManagerModules and
# nixosModules. A missing output kind counts as zero. A broken output makes
# nix exit non-zero; the last "eval KIND.NAME" line before the error names it.
# The script never writes flake.lock.
#
# Source: devman bin/nix-eval-check.sh. Repositories carry a copy at
# nix/eval-check.sh. Change the plane copy first, then copy it.
set -euo pipefail

cd "${1:-${DEVENV_ROOT:-$PWD}}"

if [ ! -f flake.nix ]; then
  echo "nix-eval-check: no flake.nix, nothing to evaluate"
  exit 0
fi

# A Testee task shell drops HOME and the host PATH. Restore both.
if [ -z "${HOME:-}" ]; then
  HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
  export HOME
fi
if ! command -v nix >/dev/null 2>&1; then
  PATH="$PATH:/run/current-system/sw/bin:/etc/profiles/per-user/$(id -un)/bin:/nix/var/nix/profiles/default/bin"
  export PATH
fi

# The Nix text holds ${...} on purpose; the shell must not expand it.
# shellcheck disable=SC2016
# getFlake reads the repository as a flake and refuses to update flake.lock.
FLAKE_DIR=$PWD nix eval --impure --raw --no-write-lock-file --expr '
  let
    flake = builtins.getFlake (builtins.getEnv "FLAKE_DIR");
    system = builtins.currentSystem;
    kind = k: if flake ? ${k} && flake.${k} ? ${system} then flake.${k}.${system} else { };
    drvs = k: builtins.mapAttrs (n: d: builtins.trace "eval ${k}.${n}" d.drvPath) (kind k);
    names = k: if flake ? ${k} then builtins.attrNames flake.${k} else [ ];
    count = x: toString (builtins.length (builtins.attrNames x));
    p = drvs "packages";
    c = drvs "checks";
    s = drvs "devShells";
    h = names "homeManagerModules";
    n = names "nixosModules";
  in
  builtins.deepSeq [ p c s h n ] (
    "nix-eval-check: ok (${system}) packages ${count p}, checks ${count c}, devShells ${count s}, "
    + "homeManagerModules ${toString (builtins.length h)}, nixosModules ${toString (builtins.length n)}\n"
  )'
