# The Mode A llama-server as a Nix derivation: the cheap wrapper.
#
# This builds what `ci/build/build-llamacpp.sh --profile
# vulkan-p2fork-server` builds, without a hand-built directory in the
# checkout. The compile step lives in `nix/llama-server-bin.nix`. This file
# adds the launcher layout, the licence and the two manifests on top of it. It
# runs in seconds, because it compiles nothing. A change to the profile id, the
# build id or the manifest text rebuilds this wrapper and leaves the binary
# alone.
#
# The output has the layout of a published artefact (`ci/modea/server/<id>`):
#
#   bin/llama-server  lib/  LICENSE.llama.cpp  build-manifest.json
#   modea-provenance.json
#
# `bin/llama-server` and `lib` are symbolic links into the compile output. The
# RUNPATH of the binary names that output, so the links work from any path.
# `ci/build/mode-a-server.sh` copies with `cp -a`. Resolve the links first when
# you publish a copy (`cp -aL`).
#
# The Vulkan driver (the ICD) comes from the host. The vulkan-loader in the
# store finds it in /run/opengl-driver/share/vulkan/icd.d, which NixOS fills
# when `hardware.graphics.enable` is set.
#
# Use:
#   pkgs.callPackage ./llama-server.nix {
#     llama-server-bin = pkgs.callPackage ./llama-server-bin.nix { };
#   }
{
  lib,
  runCommand,
  jq,
  # The compiled server (nix/llama-server-bin.nix). Override it to change the
  # compiler setup, for example `.override { stdenv = pkgs.ccacheStdenv; }`.
  llama-server-bin,
  # The record of the Mode A wheel build. The provenance file compares it.
  vendorManifest ? ../ci/build/vendor-manifest.json,
}:

let
  # METADATA. These plain values change often. They reach the wrapper only,
  # never the compile step.
  profile = "vulkan-p2fork-server";
  buildId = "${profile}-${shortRev}-residency-${builtins.substring 0 12 patchSha}";
  # The P2 patch provenance that the fork commit messages carry.
  p2Sha = "8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752";
  p2Features = [
    "p2-mixed-batch-lora"
    "p2-seq-lora-narrow"
    "p2-server-seq-routing"
  ];

  # Facts of the compile step.
  inherit (llama-server-bin)
    forkRepo
    forkRev
    shortRev
    patchSha
    forkSource
    manifestCmakeFlags
    ;
  vendor = (builtins.fromJSON (builtins.readFile vendorManifest)).llama_cpp;
in
runCommand "llama-server-modea-${llama-server-bin.version}"
  {
    nativeBuildInputs = [ jq ];
    # The work is a few small files. A build farm or a cache costs more.
    preferLocalBuild = true;
    allowSubstitutes = false;
    passthru = {
      bin = llama-server-bin;
    };
    meta = llama-server-bin.meta // {
      description = "Mode A llama-server: the llama-infernal fork with the Vulkan residency patch";
      mainProgram = "llama-server";
    };
  }
  ''
    bin=${llama-server-bin}
    mkdir -p "$out/bin"
    ln -s "$bin/bin/llama-server" "$out/bin/llama-server"
    ln -s "$bin/lib" "$out/lib"
    install -m 0444 ${forkSource}/LICENSE "$out/LICENSE.llama.cpp"

    ggml_lib="$(basename "$(readlink -f "$bin/lib/libggml.so")")"
    server_seq_routing=false
    if grep -rq llama_set_seq_adapter ${forkSource}/tools/server/; then server_seq_routing=true; fi

    jq -n \
      --arg repo ${lib.escapeShellArg forkRepo} \
      --arg rev ${forkRev} \
      --arg short ${shortRev} \
      --arg profile ${profile} \
      --arg flags ${lib.escapeShellArg manifestCmakeFlags} \
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
        cmake_flags: $flags,
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

    # The provenance file holds file hashes of the compiled files. Hash the
    # real files in the compile output, not the links.
    hashes="{}"
    for f in "$bin"/bin/* "$bin"/lib/*; do
      [ -f "$f" ] && [ ! -L "$f" ] || continue
      rel="''${f#"$bin"/}"
      hashes="$(jq -c --arg k "$rel" --arg v "$(sha256sum "$f" | cut -d' ' -f1)" '. + {($k): $v}' <<<"$hashes")"
    done

    jq -n \
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
        source_out_dir_note: "The Nix store path of this artefact. The files in bin/ and lib/ are links into the compile output. Prove the binaries with artefact_sha256.",
        vulkan_residency_patch_sha256: $residency,
        wheel_vulkan_residency_patch_sha256: $wheel_patch,
        artefact_sha256: $hashes,
        server_llama_cpp_commit: $short,
        wheel_llama_cpp_commit: $wheel_commit,
        wheel_and_server_same_commit: (($wheel_commit | startswith($short)) and $wheel_patch == $residency),
        build_system: "nix"
      }' > "$out/modea-provenance.json"
  ''
