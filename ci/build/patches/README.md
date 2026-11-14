# Vulkan residency patch

`vulkan-residency.patch` applies to llama-infernal commit
`ecc0cb0d34e17595ff0546b89c44fa100d076c0d`. Build scripts record the patch
SHA-256 with the fork commit. The scripts refuse a Mode A build from a
different commit.

The patch allocates Vulkan write staging buffers in device-local memory. It uses
CPU memory for host buffers.

Readback staging stays in cached host memory. A copy out of a device-local
host-visible buffer reads an uncached PCIe mapping at about 7 MiB/s. That made
each recurrent-state checkpoint take about 20 s. The patch adds a separate
`sync_staging_read` buffer. `ggml_backend_vk_init` allocates 16 MiB of it before
the server samples the residency baseline, so the block is part of the baseline
and GTT stays flat after it. The block never grows. Checkpoint copies and logits
readback use it. A read larger than 16 MiB goes through the block in chunks.
Each chunk copies into the block, submits, waits, and copies out before the
next chunk reuses the block. A request above the block size is an assertion
failure, not a reallocation. The patch also adds two functions that report a
model's assigned layer and output devices. It does not change the anchored
structures.

The read path takes `device->mutex` for the whole chunk loop. The read staging
block belongs to the device, so two contexts on one device in two threads must
not interleave their chunks. The read path creates the block when a device never
went through `ggml_backend_vk_init`. The caller holds the mutex at that point.

## Hardware requirement

The patch needs a Vulkan device with a memory type that is DeviceLocal,
HostVisible and HostCoherent at the same time. In practice that means a
discrete GPU with a large BAR (resizable BAR, or a card that exposes its whole
VRAM to the host). The patch has no fallback. On a device without such a memory
type, the first write staging allocation throws `vk::OutOfDeviceMemoryError`
("No suitable memory type found"). The model load fails. The patch never
selects another memory type.

The patch also makes `get_host_buffer_type` return `nullptr` for every Vulkan
device. A patched build offers no pinned host buffer type on any Vulkan device.

The build profiles `vulkan-p2fork`, `vulkan-mi25-p2fork-server` and the other
Vulkan profiles name no hardware, but a patched build only runs on a device that
meets this requirement. Check a new device before you serve from it: list its memory types (for
example with `vulkaninfo`) and look for one with all three flags and a heap as
large as the write staging buffers need.

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
