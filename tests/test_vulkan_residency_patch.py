"""Static checks on `vulkan-residency.patch`.

A native build of the patched Vulkan backend needs a GPU, so these tests read
the patch text. They guard the one stall that a real build showed: a readback
through device-local staging memory ran at about 7 MiB/s. They also guard the
read staging block: it is allocated once at init, never grows, and serves a
larger read in chunks. One test compiles the chunk planner and checks it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_PATCH = Path(__file__).parent.parent / "ci" / "build" / "patches" / "vulkan-residency.patch"


def _added() -> str:
    lines = _PATCH.read_text().splitlines()
    return "\n".join(line[1:] for line in lines if line.startswith("+") and not line.startswith("+++"))


def _patched() -> str:
    """Context and added lines: the text after the patch applies."""
    lines = _PATCH.read_text().splitlines()
    return "\n".join(line[1:] for line in lines if line.startswith((" ", "+")) and not line.startswith("+++"))


def test_read_staging_is_cached_host_memory() -> None:
    lines = _patched().splitlines()
    call = next(i for i, line in enumerate(lines) if "sync_staging_read = ggml_vk_create_buffer_check(" in line)
    flags = "\n".join(lines[call : call + 3])
    assert "eHostCached" in flags
    assert "eDeviceLocal" not in flags


def test_read_staging_exists_before_the_residency_baseline() -> None:
    # ggml_backend_vk_init runs before the server samples GTT, so the fixed block counts in the baseline.
    patched = _patched()
    call = "ggml_vk_init_sync_staging_read_buffer(ctx->device);"
    assert f"ggml_vk_init(ctx, dev_num);\n    {call}" in patched
    assert "static constexpr size_t GGML_VK_SYNC_STAGING_READ_SIZE = 16u * 1024 * 1024;" in patched


def test_read_staging_never_grows() -> None:
    patched = _patched()
    # No grow function and no size comparison that would trigger a reallocation.
    assert "ggml_vk_ensure_sync_staging_read_buffer" not in patched
    assert "sync_staging_read->size < size" not in patched
    # The block is created in one place only, and only when it is missing.
    assert patched.count("sync_staging_read = ggml_vk_create_buffer_check(") == 1
    init = _function(patched, "static void ggml_vk_init_sync_staging_read_buffer(")
    assert "if (device->sync_staging_read == nullptr) {" in init
    assert "ggml_vk_destroy_buffer" not in init
    # A request above the block is an assertion, not a reallocation.
    assert "GGML_ASSERT(staging_size <= staging_buffer->size);" in patched


def test_both_read_paths_split_into_chunks() -> None:
    patched = _patched()
    # The patch text holds hunks, not whole functions, so each path is found by its own lines.
    start = patched.index("auto read_chunk = [&](")
    loop = patched.index("for (const vk_read_chunk & c : ggml_vk_split_read(width, height))", start)
    # The chunk reader submits, waits and copies out before it returns, so the loop reuses the block safely.
    reader = patched[start:loop]
    assert reader.index("ggml_vk_submit(subctx, src->device->fence);") < reader.index("waitForFences")
    assert reader.index("waitForFences") < reader.index("memcpy(cpy.dst, cpy.src, cpy.n);")
    assert "read_chunk(c.row, c.rows, c.col, c.cols);" in patched[loop : loop + 200]

    loop = patched.index("for (const vk_read_chunk & c : ggml_vk_split_read(size, n_copies))")
    body = patched[loop : loop + 700]
    # Each chunk ends with a synchronize, which submits, waits and copies out.
    assert body.index("ggml_vk_buffer_read_2d_async(") < body.index("ggml_vk_synchronize(ctx);")
    assert "compute_ctx = ggml_vk_get_compute_ctx(ctx);" in body


def test_async_read_loop_holds_the_device_mutex() -> None:
    # The read staging block belongs to the device. Two contexts on one device in two threads
    # must not interleave their chunks, so the lock covers the whole loop.
    patched = _patched()
    loop = patched.index("for (const vk_read_chunk & c : ggml_vk_split_read(size, n_copies))")
    lock = "std::lock_guard<std::recursive_mutex> guard(ctx->device->mutex);"
    start = patched.rfind(lock, 0, loop)
    assert start != -1
    # The lock sits in the same branch as the loop: no closing brace at column 4 between them.
    assert "\n    }\n" not in patched[start:loop]
    # The sync read path already holds the same mutex before its chunk loop.
    assert "std::lock_guard<std::recursive_mutex> guard(src->device->mutex);" in patched


def test_read_staging_is_created_when_the_device_has_none() -> None:
    # A device that never went through ggml_backend_vk_init has no block. The read path creates it
    # (under the caller's device mutex) instead of aborting on a null buffer.
    patched = _patched()
    take = "vk_buffer& staging_buffer = src->device->sync_staging_read;"
    create = "ggml_vk_init_sync_staging_read_buffer(src->device);"
    assert patched.index(create) < patched.index(take)
    assert "ggml_vk_init_sync_staging_read_buffer(ctx->device);" in patched


def test_chunk_planner_covers_every_byte_once() -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler on PATH")
    patched = _patched()
    constant = next(
        line
        for line in patched.splitlines()
        if line.startswith("static constexpr size_t GGML_VK_SYNC_STAGING_READ_SIZE")
    )
    chunk_struct = _block(patched, "struct vk_read_chunk {", "\n};\n")
    planner = _function(patched, "static std::vector<vk_read_chunk> ggml_vk_split_read(")
    source = _HARNESS.replace("@@PLAN@@", "\n".join([constant, chunk_struct, planner]))
    return_code = _compile_and_run(compiler, source)
    assert return_code == 0


def test_readbacks_use_the_read_staging_buffer() -> None:
    added = _added()
    assert "vk_buffer& staging_buffer = src->device->sync_staging_read;" in added
    # The old per-context copy in get_tensor_2d_async is gone.
    assert "ctx->device->sync_staging_read" not in added


def test_write_staging_stays_device_local() -> None:
    added = _added()
    assert added.count("eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible") == 2


def _block(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i) + len(end)
    return text[i:j]


def _function(text: str, signature: str) -> str:
    """One top-level function: from its signature to the first closing brace at column 0."""
    return _block(text, signature, "\n}\n")


_HARNESS = r"""
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define GGML_ASSERT(x) do { if (!(x)) { std::fprintf(stderr, "assert: %s\n", #x); std::abort(); } } while (0)

@@PLAN@@

static void check(size_t width, size_t height) {
    const auto chunks = ggml_vk_split_read(width, height);
    size_t row = 0, col = 0;
    for (const auto & c : chunks) {
        GGML_ASSERT(c.rows > 0 && c.cols > 0);
        GGML_ASSERT(c.rows * c.cols <= GGML_VK_SYNC_STAGING_READ_SIZE);
        GGML_ASSERT(c.row == row && c.col == col);
        if (c.rows > 1) GGML_ASSERT(c.col == 0 && c.cols == width);
        if (c.col + c.cols == width) { row += c.rows; col = 0; } else { GGML_ASSERT(c.rows == 1); col += c.cols; }
    }
    GGML_ASSERT(row == height && col == 0);
}

int main() {
    const size_t cap = GGML_VK_SYNC_STAGING_READ_SIZE;
    const size_t widths[] = {1, 3, 4096, cap - 1, cap, cap + 1, 40 * 1024 * 1024, 2 * cap, 3 * cap + 5};
    const size_t heights[] = {1, 2, 7, 1000, 10000000};
    for (size_t w : widths) {
        for (size_t h : heights) {
            if (w > cap && h > 7) continue;
            check(w, h);
        }
    }
    return 0;
}
"""


def _compile_and_run(compiler: str, source: str) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "plan.cpp"
        exe = Path(tmp) / "plan"
        src.write_text(source)
        build = subprocess.run(
            [compiler, "-std=c++17", "-O1", "-o", str(exe), str(src)], capture_output=True, text=True
        )
        assert build.returncode == 0, build.stderr
        return subprocess.run([str(exe)], capture_output=True, text=True).returncode
