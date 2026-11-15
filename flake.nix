{
  description = "inferference: the llama.cpp serving layer (router launcher, balancer, reconcile CLI).";

  inputs = {
    # The pin that basix uses, so the interpreter and the tools match the host.
    nixpkgs.url = "github:NixOS/nixpkgs/e7439b6b14ad3cc35d05608ebca9bce01a25f5f8";

    # The Python build chain (decision 2026-10-15). These inputs live here and
    # nowhere else. Do not repeat them in `devenv.yaml`: the devenv shell uses
    # `uv` and does not need them.
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs = { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }:
    let
      systems = [ "x86_64-linux" ];
      forSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
    in
    {
      packages = forSystems (pkgs:
        let
          built = import ./nix/package.nix { inherit pkgs pyproject-nix uv2nix pyproject-build-systems; };
        in
        {
          # The serving package: bin/inferference-router-launch, -router-reload,
          # -balancer-launch, -reconcile, -balancer, gpu-thermal-guard.
          default = built.inferference;
          inferference = built.inferference;
          # The locked Python env: core dependencies only, never the `runtime` extra.
          env = built.env;

          # The Mode A llama-server (nix/llama-server.nix): bin/llama-server,
          # lib/, the manifests. Point INFERFERENCE_SERVER_DIR at this output.
          server = pkgs.callPackage ./nix/llama-server.nix { };
        });

      checks = forSystems (pkgs: {
        # Build the serving package. It runs the install check of the wrappers.
        package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;

        # Evaluate the llama-server derivation and force its store path. The
        # check does not build it: a build compiles llama.cpp and takes long.
        # Build it with `nix build .#server`.
        server-eval =
          let
            drv = self.packages.${pkgs.stdenv.hostPlatform.system}.server;
          in
          pkgs.runCommand "inferference-server-eval" { } ''
            echo ${builtins.unsafeDiscardStringContext drv.drvPath} > "$out"
          '';

        # A stale uv.lock fails this check (verify runs `uv lock --check` too).
        lock = pkgs.runCommand "inferference-lock-check"
          { nativeBuildInputs = [ pkgs.uv pkgs.python313 ]; }
          ''
            cp ${./pyproject.toml} pyproject.toml
            cp ${./uv.lock} uv.lock
            cp ${./README.md} README.md
            mkdir -p src/inferference && touch src/inferference/__init__.py
            export UV_NO_CACHE=1 HOME=$TMPDIR UV_PYTHON=${pkgs.python313}/bin/python3
            uv lock --check --offline --no-managed-python --no-python-downloads
            touch "$out"
          '';
      });
    };
}
