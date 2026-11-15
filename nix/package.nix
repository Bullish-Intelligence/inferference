# The serving package: the locked Python env (core dependencies only) and the
# runner scripts, wrapped so that they need no checkout. Built from `uv.lock`
# with uv2nix. The interpreter comes from nixpkgs (python313).
{ pkgs, pyproject-nix, uv2nix, pyproject-build-systems }:

let
  inherit (pkgs) lib;

  workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ../.; };

  # Only the package inputs reach the build, so unrelated edits do not rebuild it.
  source = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../src
      ../bench/workloads # hatch force-includes it into the wheel
      ../pyproject.toml
      ../README.md
      ../LICENSE
    ];
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages { python = pkgs.python313; }).overrideScope (
      lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        (workspace.mkPyprojectOverlay { sourcePreference = "wheel"; })
        (final: prev: {
          inferference = prev.inferference.overrideAttrs (old: {
            src = source;
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ final.resolveBuildSystem {
              hatchling = [ ];
            };
          });
        })
      ]
    );

  # Core dependencies only: no extra, no dependency group. The env holds no
  # llama-cpp-python, xgrammar, or torch. The server is a separate package.
  env = pythonSet.mkVirtualEnv "inferference-env" { inferference = [ ]; };

  runner = lib.fileset.toSource {
    root = ../.;
    fileset = lib.fileset.unions [
      ../ci/runner/inferference-router-launch.sh
      ../ci/runner/inferference-router-reload.sh
      ../ci/runner/inferference-balancer-launch.sh
      ../ci/runner/inferference-reconcile.py
      ../ci/runner/gpu-thermal-guard.sh
      ../ci/runner/gpu-thermal-signal-relay.py
      ../ci/build/patches/vulkan-residency.patch
    ];
  };

  tools = lib.makeBinPath (with pkgs; [ bash coreutils findutils gnugrep gnused util-linux ]);
in
{
  inherit env;

  inferference =
    pkgs.runCommand "inferference-${(builtins.fromTOML (builtins.readFile ../pyproject.toml)).project.version}"
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
        meta.mainProgram = "inferference-router-launch";
      }
      ''
        # The scripts keep the layout `ci/runner` and find the patch beside them.
        share="$out/share/inferference"
        mkdir -p "$share/ci/runner" "$out/bin"
        cp ${runner}/ci/runner/* "$share/ci/runner/"
        cp ${runner}/ci/build/patches/vulkan-residency.patch "$share/ci/runner/"
        chmod -R u+w "$share"
        patchShebangs "$share/ci/runner"

        wrap() {  # name script
          makeWrapper "$share/ci/runner/$2" "$out/bin/$1" \
            --set INFERFERENCE_PYTHON ${env}/bin/python \
            --set INFERFERENCE_RESIDENCY_PATCH "$share/ci/runner/vulkan-residency.patch" \
            --set GPU_THERMAL_TRACE_PYTHON ${env}/bin/python \
            --set PYTHONDONTWRITEBYTECODE 1 \
            --prefix PATH : ${tools}
        }
        wrap inferference-router-launch   inferference-router-launch.sh
        wrap inferference-router-reload   inferference-router-reload.sh
        wrap inferference-balancer-launch inferference-balancer-launch.sh
        wrap gpu-thermal-guard            gpu-thermal-guard.sh

        # The console scripts of the locked env.
        for name in inferference-reconcile inferference-balancer; do
          makeWrapper ${env}/bin/$name "$out/bin/$name" --set PYTHONDONTWRITEBYTECODE 1
        done
      '';
}
