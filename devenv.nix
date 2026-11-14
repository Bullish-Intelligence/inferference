# Inferference — repoman-enabled Python devenv.
#
# `.repoman/project.toml` owns the lifecycle manager roster. Testee checks
# each commit in its own workspace and keeps one verdict per check. The machine
# supplies the `testee` command. The `tasks` table below declares the checks.
#
# Two environments. The required gate (check:*, test:full) runs in the locked core
# environment: `uv sync --locked` plus numpy at its locked version. It runs the tests that
# do not carry the `runtime` marker. `opt:runtime` builds the native library venv
# (CPython 3.13, `ci/runner/library-venv.sh`: llama-cpp-python, xgrammar, torch) under
# `.devenv/state/library-venv` and runs every test there.
#
# CUDA build toolchain: the llama.cpp build pipeline (ci/build/build-llamacpp.sh)
# builds CUDA libs (profiles cuda-3060 / p2fork) with gcc + cmake + ninja + the
# CUDA toolkit: cmake, ninja, ccache, gcc, cuda_nvcc, cuda_cudart, cuda_cccl,
# libcublas, cuda_nvrtc. AGENTS.md makes devenv the only dev environment, so the
# toolchain lives here, not in a side shell. ccache stays persistent via
# CCACHE_DIR.
{ pkgs, ... }:

{

  # Unfree: the CUDA toolkit (cuda_nvcc/cudart/etc.) is unfree. devenv has no
  # nixpkgs.config option; allow it with NIXPKGS_ALLOW_UNFREE=1 on entry:
  #   NIXPKGS_ALLOW_UNFREE=1 devenv shell
  # (devenv-nixpkgs docs, using/configuration.chapter.md:105).

  # Python toolchain. The venv hosts the app. Run `uv sync` once to fill it.
  # The host profile puts the managers (copyroom, gitman, docman) and `testee`
  # on PATH.
  languages.python = {
    enable = true;
    venv.enable = true;
    uv.enable = true;
  };

  packages = [
    # systemctl on the clean-shell PATH: tests/test_gpu_thermal_guard.py
    # asserts shutil.which("systemctl") is not None. The v0.4 interactive
    # environment carried the host PATH; the v2 clean shell does not.
    pkgs.systemd
    pkgs.git
    pkgs.cmake
    pkgs.ninja
    pkgs.ccache
    pkgs.gcc
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cuda_cudart
    pkgs.cudaPackages.cuda_cccl
    pkgs.cudaPackages.libcublas
    pkgs.cudaPackages.cuda_nvrtc
    # Vulkan build toolchain — the AMD MI25 (Vega 10 / gfx900) profiles.
    # ROCm dropped gfx900 after 5.7, so the MI25 baseline goes through ggml's
    # Vulkan backend on mesa's RADV driver, which enumerates both cards today
    # (deviceName "AMD Radeon Graphics (RADV VEGA10)"). GGML_VULKAN=ON needs the
    # headers, the loader, and glslc (shaderc) to compile the shader set at
    # build time; vulkan-tools gives `vulkaninfo` for the device probe.
    pkgs.vulkan-headers
    pkgs.vulkan-loader
    pkgs.shaderc
    pkgs.glslang
    pkgs.vulkan-tools
    # patchelf — the server-profile collect step rewrites the binary's RUNPATH
    # and REFUSES to emit a manifest without it (build-llamacpp.sh).
    pkgs.patchelf
    # Act — the CI runner for the rig (ci/workflows/*.yml are GitHub-Actions-
    # format workflows executed locally by Act).
    # Run `act` inside this shell; `.actrc` maps every runner label to the host
    # so no containers are needed and the devenv toolchain + GPU are visible.
    pkgs.act
  ];

  # Operator entrypoints. evict-and-run-eval is the measurement run sequence
  # with the waiting built in: free the measured card from the router via the
  # supported `evict` route (placement config untouched, `apply` restores
  # serving), poll until the card drains below the runner's residency floor,
  # then exec the experiment run CLI. It needs an explicit experiment spec.
  # Full contract and test overrides: ci/runner/evict-and-run-eval.sh.
  # Nothing here writes sysfs.
  # The exec lines resolve the project root at runtime (PROJECT_ROOT or the
  # caller's cwd), because devenv may evaluate this file from a store copy and
  # a baked-in path would point at a stale tree.
  scripts = {
    evict-and-run-eval = {
      description = "Evict the router's model from a card, wait for it to drain, then run an experiment spec.";
      exec = ''bash "''${PROJECT_ROOT:-$(pwd)}/ci/runner/evict-and-run-eval.sh" "$@"'';
    };
    reconcile = {
      description = "Reconcile the router to the placement YAML (ci/runner/inferference-reconcile.py).";
      exec = ''python "''${PROJECT_ROOT:-$(pwd)}/ci/runner/inferference-reconcile.py" "$@"'';
    };
  };


  enterShell = ''
    # Persistent ccache for llama.cpp builds. The build
    # dir is rm -rf'd per run; the persistent CCACHE_DIR is what makes rebuilds
    # fast. GGML_CCACHE=ON auto-wires ccache as the compiler launcher.
    export CCACHE_DIR="''${CCACHE_DIR:-$HOME/.cache/llamacpp-ccache}"
    mkdir -p "$CCACHE_DIR"
    # Library-load path for native Python extensions in the library venv
    # (numpy / xgrammar / torch load nix gcc's libstdc++ + libgomp and zlib's
    # libz at runtime; without them the extensions fail to dlopen in a bare
    # shell). GPU jobs additionally source ci/runner/rig-env.sh, which prepends
    # /run/opengl-driver/lib (libcuda) and the build lib dir — this is the
    # CPU-side base every local python needs.
    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.gcc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
    git --version
  '';

  # https://devenv.sh/tasks/
  #
  # Testee runs `setup:*` once, then the standard tasks (check:*, test:*).
  # Every task declares `input.testee`, so Testee can reuse a verdict when a commit
  # changes none of its inputs. Run `testee doctor` after any edit to this table.
  # devman's `base` group calls `base:check` and `base:test`.
  tasks =
    let
      # Files that no lint, format, type, or test task reads.
      unread = [
        "AGENTS.md"
        "CLAUDE.md"
        ".gitignore"
        ".tmuxp.yaml"
        "flake.nix"
        "flake.lock"
        "docs/**"
      ];
    in
    {
      "setup:sync" = {
        exec = ''
          uv sync --locked
          # numpy at the version that uv.lock pins for the runtime extra. The rest of the extra
          # (llama-cpp-python, xgrammar, torch) stays out of the required gate.
          numpy_pin="$(uv export --locked --no-hashes --no-emit-project --extra runtime | grep -E '^numpy==')"
          uv pip install --python "''${UV_PROJECT_ENVIRONMENT:-.devenv/state/venv}" --no-deps "$numpy_pin"
        '';
        # hatchling reads the readme when uv builds the editable install.
        input.testee.reads = [ "uv.lock" "pyproject.toml" "README.md" ];
      };

      "check:lint" = {
        exec = ''
          out="''${TESTEE_RUN_DIR:-.devenv/state}"
          mkdir -p "$out"
          set -o pipefail
          uv run --no-sync ruff check . --output-format json | tee "$out/check:lint.ruff.json"
        '';
        input.testee.ignores = unread;
      };

      "check:format" = {
        exec = "uv run --no-sync ruff format --check .";
        input.testee.ignores = unread;
      };

      "check:types" = {
        # pyproject.toml [tool.ty.analysis] lets the native runtime modules stay unresolved here.
        exec = "uv run --no-sync ty check .";
        input.testee.ignores = unread;
      };

      "test:quick" = {
        exec = ''
          echo "n/a: no quick test subset"
          exit 0
        '';
        input.testee.ignores = unread;
      };

      "test:full" = {
        # About 4 min warm: five thermal guard tests take about 30 s each. The default limit of
        # 300 s was too tight. The `runtime` marker names the 3 tests that import the native stack.
        exec = ''
          out="''${TESTEE_RUN_DIR:-.devenv/state}"
          mkdir -p "$out"
          uv run --no-sync python -m pytest -m "not runtime" --junitxml="$out/test:full.junit.xml"
        '';
        input.testee = { timeout_s = 600; ignores = unread; };
      };

      # The whole suite and the type check in the native library venv. It builds llama-cpp-python
      # and downloads CPU torch on the first run; the venv stays under .devenv/ for later runs.
      "opt:runtime" = {
        exec = ''
          set -eu
          cd "$DEVENV_ROOT"
          out="''${TESTEE_RUN_DIR:-.devenv/state}"
          mkdir -p "$out"
          export LD_LIBRARY_PATH="${pkgs.zlib}/lib:${pkgs.gcc.cc.lib}/lib:''${LD_LIBRARY_PATH:-}"
          export LIBRARY_VENV="$DEVENV_ROOT/.devenv/state/library-venv"
          bash ci/runner/library-venv.sh
          "$LIBRARY_VENV/bin/python" -m pytest --junitxml="$out/opt:runtime.junit.xml"
          "$LIBRARY_VENV/bin/ty" check . --python "$LIBRARY_VENV/bin/python"
        '';
        input.testee = { timeout_s = 3600; ignores = unread; };
      };

      "base:check" = {
        after = [ "check:lint" "check:format" "check:types" ];
        input.testee.ignores = unread;
      };
      "base:test" = {
        after = [ "test:full" ];
        input.testee.ignores = unread;
      };
    };
}
