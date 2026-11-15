#!/usr/bin/env bash
# Deprecated name. Use ci/gates/vulkan_smoke.sh. This shim stays for one release.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "vulkan_mi25_smoke.sh is deprecated: use vulkan_smoke.sh" >&2
exec "${here}/vulkan_smoke.sh" "$@"
