# The Mode A llama-server as a Nix derivation.
#
# This builds what `ci/build/build-llamacpp.sh --profile
# vulkan-mi25-p2fork-server` builds, without a hand-built directory in the
# checkout. The profile is the llama-infernal fork at one commit, plus
# `ci/build/patches/vulkan-residency.patch`, built with Vulkan and no CUDA.
#
# It reuses the nixpkgs `llama-cpp` recipe and overrides only what differs:
#
#   * the source, the patch list and the version strings;
#   * Vulkan on, and no dynamic CPU backends. The launcher and the smoke gate
#     need `lib/libggml-vulkan.so*`, so every backend links in directly;
#   * no web UI. The live build also has no UI, and the UI needs npm;
#   * explicit x86 flags in place of GGML_NATIVE (-march=native). The flags
#     match the Xeon W-2125 (Skylake-W: AVX2, FMA, F16C, BMI2, AVX-512).
#
# The output has the layout of a published artefact (`ci/modea/server/<id>`):
#
#   bin/llama-server  lib/  LICENSE.llama.cpp  build-manifest.json
#   modea-provenance.json
#
# The Vulkan driver (the ICD) comes from the host. The vulkan-loader in the
# store finds it in /run/opengl-driver/share/vulkan/icd.d, which NixOS fills
# when `hardware.graphics.enable` is set.
#
# Use:  pkgs.callPackage ./llama-server.nix { }
{
  lib,
  llama-cpp,
  jq,
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
  # The record of the Mode A wheel build. The provenance file compares it.
  vendorManifest ? ../ci/build/vendor-manifest.json,
}:

let
  forkRepo = "https://github.com/Bullish-Design/llama-infernal";
  shortRev = builtins.substring 0 9 rev;
  patchSha = builtins.hashFile "sha256" residencyPatch;
  profile = "vulkan-mi25-p2fork-server";
  buildId = "${profile}-${shortRev}-residency-${builtins.substring 0 12 patchSha}";
  # The P2 patch provenance that the fork commit messages carry.
  p2Sha = "8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752";
  p2Features = [
    "p2-mixed-batch-lora"
    "p2-seq-lora-narrow"
    "p2-server-seq-routing"
  ];
  vendor = (builtins.fromJSON (builtins.readFile vendorManifest)).llama_cpp;
  rev = forkRev;
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
in
(llama-cpp.override {
  vulkanSupport = true;
  cpuArchDynamicDispatch = false;
}).overrideAttrs
  (old: {
    pname = "llama-server-modea";
    version = "0.0.10267-${shortRev}";
    src = resolvedSrc;
    patches = [ residencyPatch ];

    # No web UI: drop npm from the build.
    nativeBuildInputs = lib.filter (p: p != nodejs_latest && p != npmHooks.npmConfigHook) old.nativeBuildInputs;
    npmDeps = null;
    npmRoot = null;
    preConfigure = "";


    # Drop the nixpkgs values that this file sets again. The manifest then
    # lists each flag once.
    cmakeFlags =
      (lib.filter (
        f: !(lib.any (k: lib.hasInfix k f) [ "LLAMA_OPENSSL" "LLAMA_BUILD_NUMBER" "LLAMA_BUILD_COMMIT" ])
      ) old.cmakeFlags)
      ++ [
        (lib.cmakeBool "LLAMA_BUILD_UI" false)
        (lib.cmakeBool "LLAMA_USE_PREBUILT_UI" false)
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

    nativeCheckInputs = [ ];
    doCheck = false;

    nativeInstallCheckInputs = [ jq ];

    postPatch = (old.postPatch or "") + ''
      # The patch file must be the one the manifest names.
      echo "${patchSha}  ${residencyPatch}" | sha256sum -c -
      # Fail closed: the tree must hold the P2 routing surface.
      grep -q llama_set_seq_adapters include/llama.h
    '';

    # Keep the nixpkgs bash completion step. Add the licence and the manifest.
    postInstall = (old.postInstall or "") + ''
      # postInstall runs in the cmake build directory. The source is one level up.
      srcdir="$NIX_BUILD_TOP/$sourceRoot"
      install -m 0444 "$srcdir/LICENSE" "$out/LICENSE.llama.cpp"

      ggml_lib="$(basename "$(readlink -f "$out/lib/libggml.so")")"
      server_seq_routing=false
      if grep -rq llama_set_seq_adapter "$srcdir/tools/server/"; then server_seq_routing=true; fi

      ${jq}/bin/jq -n \
        --arg repo ${lib.escapeShellArg forkRepo} \
        --arg rev ${rev} \
        --arg short ${shortRev} \
        --arg profile ${profile} \
        --arg flags "$(printf '%s\n' "''${cmakeFlags[@]}" | grep -E '^-D(GGML_|LLAMA_|BUILD_SHARED_LIBS|CMAKE_BUILD_TYPE)' | tr '\n' ' ')" \
        --arg ggml "$ggml_lib" \
        --arg build_id ${buildId} \
        --arg residency ${patchSha} \
        --argjson features ${lib.escapeShellArg (builtins.toJSON p2Features)} \
        --arg p2 ${p2Sha} \
        --argjson ssr "$server_seq_routing" \
        '{
          repo: $repo,
          ref_requested: $rev,
          ref_built: $short,
          profile: $profile,
          cmake_flags: ($flags | rtrimstr(" ")),
          ggml_lib: $ggml,
          ggml_version: $ggml,
          llama_cpp_commit: $short,
          build_id: $build_id,
          vulkan_residency_patch_sha256: $residency,
          features: $features,
          patch: ("commits on feat/p2-mixed-batch-lora (P2 patch provenance sha256=" + $p2 + "; no patch applied)"),
          patch_sha256: $p2,
          seq_adapter_routing: true,
          server_built: true,
          server_seq_routing: $ssr,
          path_prefix_maps: "",
          build_system: "nix"
        }' > "$out/build-manifest.json"
    '';

    # The provenance file holds file hashes. Hash after fixup, because fixup
    # edits the RUNPATH and strips the files.
    postFixup = (old.postFixup or "") + ''
      hashes="{}"
      for f in "$out"/bin/* "$out"/lib/*; do
        [ -f "$f" ] && [ ! -L "$f" ] || continue
        rel="''${f#"$out"/}"
        hashes="$(${jq}/bin/jq -c --arg k "$rel" --arg v "$(sha256sum "$f" | cut -d' ' -f1)" '. + {($k): $v}' <<<"$hashes")"
      done

      ${jq}/bin/jq -n \
        --arg out "$out" \
        --arg short ${shortRev} \
        --arg residency ${patchSha} \
        --arg wheel_commit ${vendor.commit} \
        --arg wheel_patch ${vendor.vulkan_residency_patch_sha256} \
        --argjson hashes "$hashes" \
        '{
          artefact: "llama-server",
          mode: "A",
          mode_b_applies: false,
          mode_b_reason: "LLAMA_CPP_LIB_PATH swaps a shared library, never an executable. Every server artefact is Mode A provenance by construction.",
          source_out_dir: $out,
          source_out_dir_note: "The Nix store path of this build. Prove the artefact with artefact_sha256.",
          vulkan_residency_patch_sha256: $residency,
          wheel_vulkan_residency_patch_sha256: $wheel_patch,
          artefact_sha256: $hashes,
          server_llama_cpp_commit: $short,
          wheel_llama_cpp_commit: $wheel_commit,
          wheel_and_server_same_commit: (($wheel_commit | startswith($short)) and $wheel_patch == $residency),
          build_system: "nix"
        }' > "$out/modea-provenance.json"
    '';

    meta = old.meta // {
      description = "Mode A llama-server: the llama-infernal fork with the Vulkan residency patch";
      homepage = forkRepo;
      mainProgram = "llama-server";
    };
  })
