# The compile step of the Mode A llama-server.
#
# This derivation holds the part that costs time: it compiles the llama-infernal
# fork at one commit, with the residency patch, Vulkan and no CUDA. It carries
# no build id, no manifest and no timestamp. Its hash depends only on the
# source, the patches, the cmake flags and the stdenv. A change to the profile
# name or to the manifest text does not rebuild it. `nix/llama-server.nix` adds
# those on top.
#
# It reuses the nixpkgs `llama-cpp` recipe and overrides only what differs:
#
#   * the source, the patch list and the version strings;
#   * Vulkan on, and no dynamic CPU backends. The launcher and the smoke gate
#     need `lib/libggml-vulkan.so*`, so every backend links in directly;
#   * no web UI. The live build also has no UI, and the UI needs npm;
#   * explicit x86 flags in place of GGML_NATIVE (-march=native). The flags
#     match the Xeon W-2125 (Skylake-W: AVX2, FMA, F16C, BMI2, AVX-512);
#   * only the llama-server target. `llama-server-only.patch` removes the other
#     tools (cli, bench, quantize, imatrix, tts and the rest) from the cmake
#     tree, and `LLAMA_BUILD_APP` is off. The server keeps ggml with every
#     backend, libllama, libllama-common and libmtmd.
#
# The output holds `bin/llama-server` and `lib/`. The nixpkgs install step
# gives the files their RUNPATH.
#
# `stdenv` is an argument. A consumer can swap the compiler setup:
#
#   (pkgs.callPackage ./llama-server-bin.nix { }).override { stdenv = pkgs.ccacheStdenv; }
#
# Use:  pkgs.callPackage ./llama-server-bin.nix { }
{
  lib,
  stdenv,
  llama-cpp,
  nodejs_latest,
  npmHooks,
  # The fork source. Override to test another tree. The manifest then says the
  # pinned commit, so only override with the same commit.
  forkSrc ? null,
  # The pinned fork commit. `ref_built` in the manifest is its first nine digits.
  forkRev ? "ecc0cb0d34e17595ff0546b89c44fa100d076c0d",
  forkHash ? "sha256-zXphppTK4c6ya5/xL19M+OASAamvHIkqqW6gNzLaK5s=",
  fetchFromGitHub,
  residencyPatch ? ../ci/build/patches/vulkan-residency.patch,
  # Build-only cmake change: compile the server and nothing else.
  serverOnlyPatch ? ./llama-server-only.patch,
}:

let
  forkRepo = "https://github.com/Bullish-Design/llama-infernal";
  rev = forkRev;
  shortRev = builtins.substring 0 9 rev;
  patchSha = builtins.hashFile "sha256" residencyPatch;
  resolvedSrc =
    if forkSrc != null then
      forkSrc
    else
      fetchFromGitHub {
        owner = "Bullish-Design";
        repo = "llama-infernal";
        inherit rev;
        hash = forkHash;
      };

  # The flags that this file adds or sets again.
  extraCmakeFlags = [
    (lib.cmakeBool "LLAMA_BUILD_UI" false)
    (lib.cmakeBool "LLAMA_USE_PREBUILT_UI" false)
    # llama-app links the tools that the patch removes.
    (lib.cmakeBool "LLAMA_BUILD_APP" false)
    # The live build has no TLS. The router loads local files only.
    (lib.cmakeBool "LLAMA_OPENSSL" false)
    (lib.cmakeBool "GGML_CCACHE" false)
    # The build number gives the sonames of the live build (libllama.so.0.0.10267).
    (lib.cmakeFeature "LLAMA_BUILD_NUMBER" "10267")
    (lib.cmakeFeature "LLAMA_BUILD_COMMIT" shortRev)
    # Xeon W-2125 (Skylake-W). The CPU has AVX-512 F/DQ/CD/BW/VL and no VNNI,
    # no VBMI and no BF16.
    (lib.cmakeBool "GGML_AVX" true)
    (lib.cmakeBool "GGML_AVX2" true)
    (lib.cmakeBool "GGML_FMA" true)
    (lib.cmakeBool "GGML_F16C" true)
    (lib.cmakeBool "GGML_BMI2" true)
    (lib.cmakeBool "GGML_AVX512" true)
    (lib.cmakeBool "GGML_AVX512_VBMI" false)
    (lib.cmakeBool "GGML_AVX512_VNNI" false)
    (lib.cmakeBool "GGML_AVX512_BF16" false)
  ];
in
(llama-cpp.override {
  inherit stdenv;
  vulkanSupport = true;
  cpuArchDynamicDispatch = false;
}).overrideAttrs
  (old:
    let
      # Drop the nixpkgs values that this file sets again. The manifest then
      # lists each flag once.
      cmakeFlags =
        (lib.filter (
          f:
          !(lib.any (k: lib.hasInfix k f) [
            "LLAMA_OPENSSL"
            "LLAMA_BUILD_NUMBER"
            "LLAMA_BUILD_COMMIT"
          ])
        ) old.cmakeFlags)
        ++ extraCmakeFlags;
    in
    {
      pname = "llama-server-bin";
      version = "0.0.10267-${shortRev}";
      src = resolvedSrc;
      patches = [
        residencyPatch
        serverOnlyPatch
      ];

      # One output. The install step of the trimmed tree has no headers to split.
      outputs = [ "out" ];

      # No web UI: drop npm from the build.
      nativeBuildInputs = lib.filter (
        p: p != nodejs_latest && p != npmHooks.npmConfigHook
      ) old.nativeBuildInputs;
      npmDeps = null;
      npmRoot = null;
      preConfigure = "";

      inherit cmakeFlags;

      nativeCheckInputs = [ ];
      doCheck = false;

      postPatch = (old.postPatch or "") + ''
        # The patch file must be the one that the manifest names.
        echo "${patchSha}  ${residencyPatch}" | sha256sum -c -
        # Fail closed: the tree must hold the P2 routing surface.
        grep -q llama_set_seq_adapters include/llama.h
      '';

      # The nixpkgs bash completion step runs the installed server. Skip it.
      postInstall = "";

      # What the wrapper (nix/llama-server.nix) needs to write its manifests.
      # These values read only the inputs of this derivation, so a use of them
      # does not rebuild it.
      passthru = (old.passthru or { }) // {
        inherit
          forkRepo
          forkRev
          shortRev
          patchSha
          residencyPatch
          ;
        forkSource = resolvedSrc;
        # The text of `cmake_flags` in the manifest.
        manifestCmakeFlags = lib.concatStringsSep " " (
          [ "-DCMAKE_BUILD_TYPE=Release" ]
          ++ lib.filter (f: lib.any (k: lib.hasPrefix "-D${k}" f) [ "GGML_" "LLAMA_" "BUILD_SHARED_LIBS" ]) cmakeFlags
        );
      };

      meta = old.meta // {
        description = "Mode A llama-server, compile step: the llama-infernal fork with the Vulkan residency patch";
        homepage = forkRepo;
        mainProgram = "llama-server";
      };
    }
  )
