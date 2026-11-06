# Vulkan residency patch

`vulkan-residency.patch` applies to llama-infernal commit
`ecc0cb0d34e17595ff0546b89c44fa100d076c0d`. Build scripts record the patch
SHA-256 with the fork commit. The scripts refuse a Mode A build from a
different commit.

The patch allocates Vulkan staging buffers in device-local memory. It uses CPU
memory for host buffers. It adds two functions that report a model's assigned
layer and output devices. It does not change the anchored structures.

When `INFERFERENCE_RESIDENCY_REQUIRED` is set, a model server requires an
explicit GPU list. The server samples each card after backend initialization.
It checks all layer assignments, each PCI address, VRAM growth, and GTT before
it reports readiness. The router includes this evidence in `/models`.
The router launcher enables this check and requires the recorded patch hash.
Model sleep is unavailable with this check because wakeup would need a new
load-time proof. The check requires AMD DRM memory counters.

The check leaves llama.cpp's pipeline scheduling unchanged. One split model
load increased GTT while pipeline scheduling was enabled. The allocation
source is unknown. The residency check refuses any increase and reports the
card and counters. The server does not change scheduling to make the check pass.
