"""The backend label comes from the library, and a changed label never reads an old baseline.

No GPU and no native library: each test builds a fake library directory.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from inferference import gpu_env
from inferference.benchkit import baseline_gate
from inferference.benchkit.baselines import BaselineStore
from inferference.fingerprint import ArtifactIdentity, LlamaEngineFingerprint

_MANIFEST: dict[str, object] = {"llama_cpp_commit": "abc123", "features": ["p2-mixed-batch-lora"]}
_SUMMARY = {"end_to_end": {"generation_tokens_per_second": 30.0}}


def _lib(tmp_path: Path, *names: str, flags: str | None = None) -> Path:
    lib = tmp_path / "out" / "lib"
    lib.mkdir(parents=True)
    for name in names:
        (lib / name).write_text("")
    if flags is not None:
        (lib.parent / "build-manifest.json").write_text(json.dumps({"cmake_flags": flags}))
    return lib


def test_label_reads_the_backend_library(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0.18.0", "libggml-cpu.so.0", "libggml-vulkan.so.0.18.0")
    assert gpu_env.lib_backend_label(lib) == "vulkan"


def test_label_names_a_cuda_library_as_cuda(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", "libggml-cuda.so.0")
    assert gpu_env.lib_backend_label(lib) == "cuda"


def test_label_joins_several_backends_in_sorted_order(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml-vulkan.so", "libggml-cuda.so")
    assert gpu_env.lib_backend_label(lib) == "cuda+vulkan"


def test_label_falls_back_to_manifest_flags(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", flags="-DGGML_CUDA=OFF -DGGML_VULKAN=ON -G Ninja")
    assert gpu_env.lib_backend_label(lib) == "vulkan"


def test_label_ignores_a_backend_that_is_off(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", flags="-DGGML_CUDA=OFF")
    assert gpu_env.lib_backend_label(lib) == "cpu"


def test_label_says_cpu_for_a_lib_with_no_gpu_evidence(tmp_path: Path) -> None:
    assert gpu_env.lib_backend_label(_lib(tmp_path, "libggml.so.0", "libggml-cpu.so.0")) == "cpu"


def test_label_without_evidence_is_unknown_not_gpu(tmp_path: Path) -> None:
    assert gpu_env.lib_backend_label(None, env={}) == gpu_env.UNKNOWN_BACKEND_LABEL
    assert gpu_env.lib_backend_label(tmp_path / "missing") == gpu_env.UNKNOWN_BACKEND_LABEL
    assert gpu_env.lib_backend_label(_lib(tmp_path)) == gpu_env.UNKNOWN_BACKEND_LABEL
    assert gpu_env.UNKNOWN_BACKEND_LABEL != gpu_env.GPU_BACKEND_LABEL


def test_label_of_an_unreadable_directory_is_unknown(tmp_path: Path) -> None:
    afile = tmp_path / "not-a-dir"
    afile.write_text("")
    assert gpu_env.lib_backend_label(afile) == gpu_env.UNKNOWN_BACKEND_LABEL


def test_label_names_a_hip_library_as_hip(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", "libggml-cpu.so.0", "libggml-hip.so.0")
    assert gpu_env.lib_backend_label(lib) == "hip"


def test_label_reads_a_hip_flag_from_the_manifest(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", flags="-DGGML_HIP=ON -DGGML_CUDA=OFF")
    assert gpu_env.lib_backend_label(lib) == "hip"


def test_label_without_a_manifest_does_not_claim_cpu(tmp_path: Path) -> None:
    # The backend can sit inside libggml.so. With no manifest nothing says which.
    lib = _lib(tmp_path, "libggml.so.0", "libllama.so.0")
    assert gpu_env.lib_backend_label(lib) == gpu_env.UNKNOWN_BACKEND_LABEL


def test_label_with_a_manifest_that_has_no_flags_is_unknown(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0", flags="")
    assert gpu_env.lib_backend_label(lib) == gpu_env.UNKNOWN_BACKEND_LABEL


def test_label_reads_the_manifest_that_the_caller_passes(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml.so.0")
    assert gpu_env.lib_backend_label(lib, {"cmake_flags": "-DGGML_VULKAN=ON"}) == "vulkan"


def test_strict_label_raises_instead_of_returning_unknown(tmp_path: Path) -> None:
    with pytest.raises(gpu_env.BackendUnknownError, match="no evidence"):
        gpu_env.lib_backend_label(tmp_path / "missing", strict=True)
    assert gpu_env.lib_backend_label(_lib(tmp_path, "libggml-vulkan.so"), strict=True) == "vulkan"


def test_label_falls_back_to_the_loaded_library_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded = _lib(tmp_path, "libggml-hip.so", "libllama.so")
    fake = types.SimpleNamespace(_lib=types.SimpleNamespace(_name=str(loaded / "libllama.so")))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake)
    # The variable names another directory. The loaded library is the one that runs, so it wins.
    other = tmp_path / "other"
    other.mkdir()
    (other / "libggml-cuda.so").write_text("")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(other))
    assert gpu_env.lib_backend_label() == "hip"
    # An explicit directory still wins over the loaded library.
    assert gpu_env.lib_backend_label(other) == "cuda"
    # An explicit environment mapping is used as given.
    assert gpu_env.lib_backend_label(env={"LLAMA_CPP_LIB_PATH": str(other)}) == "cuda"


def test_label_uses_the_variable_when_no_library_is_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "llama_cpp.llama_cpp", raising=False)
    lib = _lib(tmp_path, "libggml-vulkan.so")
    monkeypatch.setenv("LLAMA_CPP_LIB_PATH", str(lib))
    assert gpu_env.lib_backend_label() == "vulkan"


def test_an_unknown_backend_builds_no_cache_key() -> None:
    ident = ArtifactIdentity(path="/m.gguf", sha256="0" * 64, size_bytes=1, mtime_ns=1)
    fp = LlamaEngineFingerprint(
        model=ident, tokenizer=ident, llama_cpp_python_version="0", backend=gpu_env.UNKNOWN_BACKEND_LABEL, n_ctx=8
    )
    with pytest.raises(ValueError, match="backend is unknown"):
        fp.cache_key()


def test_label_defaults_to_the_lib_path_variable(tmp_path: Path) -> None:
    lib = _lib(tmp_path, "libggml-vulkan.so")
    assert gpu_env.lib_backend_label(env={"LLAMA_CPP_LIB_PATH": str(lib)}) == "vulkan"


def test_a_changed_backend_changes_the_cache_key() -> None:
    ident = ArtifactIdentity(path="/m.gguf", sha256="0" * 64, size_bytes=1, mtime_ns=1)

    def key(backend: str) -> str:
        return LlamaEngineFingerprint(
            model=ident, tokenizer=ident, llama_cpp_python_version="0", backend=backend, n_ctx=8
        ).cache_key()

    assert key("vulkan") != key("cuda")
    assert key("vulkan") == key("vulkan")


def test_no_runner_hard_codes_a_backend_label() -> None:
    root = Path(gpu_env.__file__).parent / "benchkit" / "runners"
    for path in sorted(root.glob("*.py")):
        assert "backend=" + '"cu' + 'da"' not in path.read_text(), path.name


def _run_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lib: Path, *extra: str) -> int:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(_SUMMARY))
    manifest = lib.parent / "build-manifest.json"
    manifest.write_text(json.dumps(_MANIFEST))
    monkeypatch.setattr(baseline_gate, "visible_gpu_uuid", lambda: "gpu-uuid")
    argv = ["baseline_gate", "--run-dir", str(run_dir), "--manifest", str(manifest)]
    (tmp_path / "store").mkdir(exist_ok=True)  # an absent root falls back to the repo ci/.ci-artifacts
    argv += ["--artifacts", str(tmp_path / "store"), *extra]
    monkeypatch.setattr("sys.argv", argv)
    return baseline_gate.main()


def test_legacy_baseline_is_not_a_match_and_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lib = _lib(tmp_path, "libggml-vulkan.so")
    store = BaselineStore(tmp_path / "store")
    legacy = baseline_gate.legacy_baseline_key(_MANIFEST, "json_workload")
    store.save(legacy, {"end_to_end": {"generation_tokens_per_second": 1000.0}})

    # Compare-only: the old 1000 tok/s baseline must not read as a match, and the miss must fail loudly.
    assert _run_gate(monkeypatch, tmp_path, lib) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "no baseline" in out and "legacy baseline" in out
    assert store.load(legacy) is not None  # never deleted
    assert not list(store.dir.glob("*@vulkan.json"))


def test_save_run_reports_that_nothing_was_compared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lib = _lib(tmp_path, "libggml-vulkan.so")
    assert _run_gate(monkeypatch, tmp_path, lib, "--save") == 0
    assert "NO COMPARISON MADE" in capsys.readouterr().out
    key = baseline_gate.baseline_key(_MANIFEST, "json_workload", "vulkan")
    saved = BaselineStore(tmp_path / "store").load(key)
    assert saved is not None and saved["_baseline_backend"] == "vulkan"
    # The second run compares against it.
    assert _run_gate(monkeypatch, tmp_path, lib) == 0
    assert "PASS" in capsys.readouterr().out


def test_a_cuda_baseline_is_not_read_by_a_vulkan_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = BaselineStore(tmp_path / "store")
    store.save(baseline_gate.baseline_key(_MANIFEST, "json_workload", "cuda"), _SUMMARY)
    lib = _lib(tmp_path, "libggml-vulkan.so")
    assert _run_gate(monkeypatch, tmp_path, lib) == 1


def test_baseline_gate_refuses_an_unknown_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lib = _lib(tmp_path)  # an empty library directory
    assert _run_gate(monkeypatch, tmp_path, lib, "--save") == 1
    assert "no evidence of its backend" in capsys.readouterr().out
    assert not list((tmp_path / "store").rglob("*.json"))
