"""`ci/build/mode-a-server.sh` publication checks.

The script runs against a fake repository root in a tmp dir. The fake server is
a shell script that answers `--version`. No GPU and no native stack are needed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "ci" / "build" / "mode-a-server.sh"
_COMMIT = "ecc0cb0d34e17595ff0546b89c44fa100d076c0d"
_PATCH = "a" * 64
_BUILD_ID = "test-build-ecc0cb0d3-residency-aaaaaaaaaaaa"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_repo(tmp_path: Path, *, wheel_patch: str = _PATCH) -> tuple[Path, Path, Path]:
    """Return (script, out dir, destination root)."""
    repo = tmp_path / "repo"
    build = repo / "ci" / "build"
    build.mkdir(parents=True)
    script = build / "mode-a-server.sh"
    shutil.copy(_SCRIPT, script)
    (build / "vendor-manifest.json").write_text(
        json.dumps({"llama_cpp": {"commit": _COMMIT, "vulkan_residency_patch_sha256": wheel_patch}})
    )
    out = tmp_path / "out"
    (out / "bin").mkdir(parents=True)
    (out / "lib").mkdir()
    (out / "include").mkdir()
    server = out / "bin" / "llama-server"
    server.write_text("#!/bin/sh\necho 'version: 1 (test)'\n")
    server.chmod(0o755)
    (out / "lib" / "libggml-vulkan.so").write_bytes(b"vulkan backend bytes")
    (out / "lib" / "libllama.so.0").write_bytes(b"llama bytes")
    (out / "lib" / "libllama.so").symlink_to("libllama.so.0")
    (out / "build-manifest.json").write_text(
        json.dumps(
            {
                "build_id": _BUILD_ID,
                "ref_built": _COMMIT,
                "profile": "test",
                "server_built": True,
                "seq_adapter_routing": True,
                "server_seq_routing": True,
                "vulkan_residency_patch_sha256": _PATCH,
            }
        )
    )
    dest_root = tmp_path / "dest"
    dest_root.mkdir()
    return script, out, dest_root


def _run(script: Path, out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), "--out", str(out), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_publishes_into_the_destination_root_with_hashes(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    result = _run(script, out, "--no-current", "--release", "--dest-root", str(dest_root))
    assert result.returncode == 0, result.stderr
    published = dest_root / _BUILD_ID
    assert (published / "bin" / "llama-server").is_file()
    assert not (tmp_path / "repo" / "ci" / "modea").exists(), "the script wrote into its own repo root"
    assert not (dest_root / "current").exists()

    prov = json.loads((published / "modea-provenance.json").read_text())
    assert prov["vulkan_residency_patch_sha256"] == _PATCH
    assert prov["wheel_vulkan_residency_patch_sha256"] == _PATCH
    assert prov["wheel_and_server_same_commit"] is True
    # Regular files only: the libllama.so symlink has no entry of its own.
    assert prov["artefact_sha256"] == {
        "bin/llama-server": _sha256((published / "bin" / "llama-server").read_bytes()),
        "lib/libggml-vulkan.so": _sha256(b"vulkan backend bytes"),
        "lib/libllama.so.0": _sha256(b"llama bytes"),
    }


def test_publishes_the_llamacpp_licence_with_the_server(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    (out / "LICENSE.llama.cpp").write_text("MIT License\n")
    result = _run(script, out, "--no-current", "--release", "--dest-root", str(dest_root))
    assert result.returncode == 0, result.stderr
    assert (dest_root / _BUILD_ID / "LICENSE.llama.cpp").read_text() == "MIT License\n"


def test_an_older_build_with_no_licence_file_publishes_with_a_warning(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    result = _run(script, out, "--no-current", "--release", "--dest-root", str(dest_root))
    assert result.returncode == 0, result.stderr
    assert "no LICENSE.llama.cpp" in result.stderr
    assert not (dest_root / _BUILD_ID / "LICENSE.llama.cpp").exists()


def test_release_refuses_a_patch_hash_that_the_wheel_does_not_share(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path, wheel_patch="b" * 64)
    result = _run(script, out, "--no-current", "--release", "--dest-root", str(dest_root))
    assert result.returncode != 0
    assert "same Vulkan residency patch" in result.stderr
    assert not (dest_root / _BUILD_ID).exists()


def test_refuses_to_replace_the_build_that_current_names(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    kept = dest_root / _BUILD_ID
    kept.mkdir()
    (kept / "marker").write_text("keep")
    (dest_root / "current").symlink_to(_BUILD_ID)
    result = _run(script, out, "--no-current", "--dest-root", str(dest_root))
    assert result.returncode != 0
    assert "'current' names" in result.stderr
    assert (kept / "marker").read_text() == "keep"


def test_republishing_a_build_that_is_not_current_replaces_it(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    old = dest_root / _BUILD_ID
    old.mkdir()
    (old / "marker").write_text("old")
    other = dest_root / "other-build"
    other.mkdir()
    (dest_root / "current").symlink_to("other-build")
    result = _run(script, out, "--no-current", "--dest-root", str(dest_root))
    assert result.returncode == 0, result.stderr
    assert not (old / "marker").exists()
    assert (dest_root / "current").readlink() == Path("other-build")


def test_refuses_an_artefact_with_no_server(tmp_path: Path) -> None:
    script, out, dest_root = _make_repo(tmp_path)
    (out / "bin" / "llama-server").unlink()
    result = _run(script, out, "--no-current", "--dest-root", str(dest_root))
    assert result.returncode != 0
    assert not (dest_root / _BUILD_ID).exists()
