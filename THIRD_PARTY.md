# Third-party notices

Inferference builds and ships code from the projects below. Each project keeps its own copyright and licence. The full licence text of each is in that project's source tree.

## llama.cpp and ggml

- Project: llama.cpp, with its ggml tensor library.
- Copyright: The ggml authors (2023-2026).
- Licence: MIT.
- Source: https://github.com/ggml-org/llama.cpp
- Use here: the built libraries (`libllama`, `libggml*`) and the `llama-server` binary compile llama.cpp source. The build takes this source from the llama-infernal fork (below).

The build scripts copy the llama.cpp `LICENSE` file into each future build as `LICENSE.llama.cpp`. The copy sits beside `build-manifest.json` in a build, in the published server directory, and in the Mode A wheel package directory (`llama_cpp/`).

## llama-cpp-python

- Project: llama-cpp-python.
- Copyright: Andrei Betlen (2023).
- Licence: MIT.
- Source: https://github.com/abetlen/llama-cpp-python
- Use here: Mode A builds the wheel from this source (version 0.3.34). The wheel carries its own licence file in its metadata. Mode B and the Python bindings load libraries through its ctypes interface.

## llama-infernal

- Project: llama-infernal, a fork of llama.cpp.
- Source: https://github.com/Bullish-Design/llama-infernal
- Licence: MIT, inherited from llama.cpp. The fork adds commits for per-sequence adapter routing and other changes.
- Use here: the build scripts take the llama.cpp source from this fork at a pinned commit. They apply `ci/build/patches/vulkan-residency.patch` to it. The patch is part of this repository and is under this repository's licence.

## This repository

The code in this repository is under the MIT licence in `LICENSE`.
