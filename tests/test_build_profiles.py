"""Build profile checks for `ci/build/build-llamacpp.sh`. No GPU and no native build.

The script runs from a copy in a tmp dir, against a tiny local git repository,
with a fake `cmake` first on PATH. The fake records its arguments and writes the
files that a real build would. So the tests read the real flag table, the real
provenance checks and the real manifest writer.

The residency patch needs the fork commit itself. One test applies the patch to
a clone of the fork when the machine holds one, and skips when it does not.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).parent.parent
_SCRIPT = _ROOT / "ci" / "build" / "build-llamacpp.sh"
_PATCH = _ROOT / "ci" / "build" / "patches" / "vulkan-residency.patch"
_ANCHOR = "ecc0cb0d34e17595ff0546b89c44fa100d076c0d"
_P2_SHA = "8b618ac212fc667188cef8abad8054f40ea5e7c67b2023f0fcb6466e534ff752"

_FAKE_CMAKE = r"""#!/usr/bin/env bash
set -euo pipefail
args="$*"
echo "$args" >> "$FAKE_CMAKE_LOG"
if [[ "$1" == "-S" ]]; then
  build=""
  while [[ $# -gt 0 ]]; do
    [[ "$1" == "-B" ]] && build="$2"
    shift
  done
  server=OFF
  [[ "$args" == *"-DLLAMA_BUILD_SERVER=ON"* ]] && server=ON
  echo "LLAMA_BUILD_SERVER:BOOL=$server" > "$build/CMakeCache.txt"
else
  build="$2"
  mkdir -p "$build/bin"
  for name in libllama.so.0 libggml.so.0.1.0 libggml-base.so.0 libggml-cpu.so.0 libggml-vulkan.so; do
    echo "$name" > "$build/bin/$name"
  done
  if [[ "$args" == *llama-server* ]]; then
    for name in libllama-common.so.0 libllama-server-impl.so llama-server; do
      echo "$name" > "$build/bin/$name"
    done
    chmod +x "$build/bin/llama-server"
  fi
fi
"""


# A fake patchelf. `--print-rpath` answers with a RUNPATH that holds a build-tree entry, an empty
# entry and a nix-store entry. `--set-rpath` logs its arguments. The script must remove the first
# two and keep the third.
_FAKE_PATCHELF = r"""#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  --print-rpath)
    state="$FAKE_PATCHELF_STATE/$(basename "$2")"
    if [[ -f "$state" ]]; then cat "$state"; else
      echo "${FAKE_BUILD_RPATH}:/nix/store/zzz-devenv-shell-env/lib:/nix/store/aaa-glibc/lib::${FAKE_GCC_RPATH}"
    fi
    ;;
  --set-rpath)
    echo "$2 $3" >> "$FAKE_PATCHELF_LOG"
    echo "$2" > "$FAKE_PATCHELF_STATE/$(basename "$3")"
    ;;
  *) exit 2 ;;
esac
"""


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
    done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env, check=True)
    return done.stdout


def _make_fork(tmp_path: Path, *, with_p2: bool = True, with_license: bool = True) -> Path:
    fork = tmp_path / "fork"
    (fork / "include").mkdir(parents=True)
    (fork / "ggml" / "include").mkdir(parents=True)
    (fork / "include" / "llama.h").write_text("void llama_set_seq_adapters(void);\n" if with_p2 else "\n")
    (fork / "ggml" / "include" / "ggml.h").write_text("\n")
    if with_license:
        (fork / "LICENSE").write_text("MIT License\n\nCopyright (c) The ggml authors\n")
    _git(fork, "init", "-q")
    _git(fork, "add", ".")
    message = f"P2 routing\n\np2-mixed-batch-lora.patch\nsha256 {_P2_SHA}\n" if with_p2 else "stock\n"
    _git(fork, "commit", "-q", "-m", message)
    return fork


def _run_build(
    tmp_path: Path, profile: str, *, with_p2: bool = True, with_license: bool = True
) -> tuple[dict, str, subprocess.CompletedProcess]:
    """Run the script. Return the manifest, the fake cmake log, and the process."""
    fork = _make_fork(tmp_path, with_p2=with_p2, with_license=with_license)
    script_dir = tmp_path / "repo" / "ci" / "build"
    script_dir.mkdir(parents=True)
    script = script_dir / "build-llamacpp.sh"
    shutil.copy(_SCRIPT, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cmake = bin_dir / "cmake"
    cmake.write_text(_FAKE_CMAKE)
    cmake.chmod(0o755)
    patchelf = bin_dir / "patchelf"
    patchelf.write_text(_FAKE_PATCHELF)
    patchelf.chmod(0o755)
    glslc = bin_dir / "glslc"
    glslc.write_text("#!/usr/bin/env bash\nexit 0\n")
    glslc.chmod(0o755)
    (tmp_path / "patchelf-state").mkdir()
    log = tmp_path / "cmake.log"
    out = tmp_path / "out"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_CMAKE_LOG": str(log),
        "FAKE_PATCHELF_LOG": str(tmp_path / "patchelf.log"),
        "FAKE_PATCHELF_STATE": str(tmp_path / "patchelf-state"),
        "FAKE_GCC_RPATH": f"/nix/store/bbb-gcc-lib/lib:{script_dir / '.llamacpp-builds' / 'build-fake' / 'bin'}",
        "FAKE_BUILD_RPATH": str(script_dir / ".llamacpp-builds" / "build-fake" / "bin"),
    }
    done = subprocess.run(
        ["bash", str(script), "--ref", f"local:{fork}", "--profile", profile, "--out", str(out)],
        capture_output=True,
        text=True,
        env=env,
    )
    manifest = {}
    if (out / "build-manifest.json").is_file():
        manifest = json.loads((out / "build-manifest.json").read_text())
    return manifest, log.read_text() if log.is_file() else "", done


def test_vulkan_p2fork_flags_and_manifest(tmp_path: Path) -> None:
    manifest, log, done = _run_build(tmp_path, "vulkan-p2fork")
    assert done.returncode == 0, done.stderr
    configure = log.splitlines()[0]
    assert "-DGGML_VULKAN=ON" in configure
    assert "-DGGML_CUDA=OFF" in configure
    assert "-DGGML_CUDA=ON" not in configure
    assert "-DLLAMA_BUILD_SERVER=OFF" in configure
    assert "-DLLAMA_BUILD_SERVER=ON" not in configure
    # The library target only: no server target.
    assert re.search(r"--build \S+ --target llama --config Release", log)
    assert "llama-server" not in log
    assert manifest["profile"] == "vulkan-p2fork"
    assert manifest["cmake_flags"].count("-DGGML_VULKAN=ON") == 1
    assert "-DGGML_CUDA=OFF" in manifest["cmake_flags"]
    assert manifest["build_id"].startswith("vulkan-p2fork-")
    assert manifest["seq_adapter_routing"] is True
    assert manifest["patch_sha256"] == _P2_SHA
    assert manifest["server_built"] is False
    assert manifest["server_seq_routing"] is False
    # A local tree is never patched: the patch field exists and is empty.
    assert manifest["vulkan_residency_patch_sha256"] == ""


def test_the_build_copies_the_llamacpp_licence_beside_the_manifest(tmp_path: Path) -> None:
    _, _, done = _run_build(tmp_path, "vulkan-p2fork")
    assert done.returncode == 0, done.stderr
    copied = tmp_path / "out" / "LICENSE.llama.cpp"
    assert copied.read_text() == (tmp_path / "fork" / "LICENSE").read_text()
    # The notice sits outside bin/ and lib/, so it never changes a file hash.
    assert not (tmp_path / "out" / "lib" / "LICENSE.llama.cpp").exists()


def test_the_build_refuses_a_source_tree_with_no_licence(tmp_path: Path) -> None:
    _, _, done = _run_build(tmp_path, "vulkan-p2fork", with_license=False)
    assert done.returncode == 3
    assert "no LICENSE" in done.stderr


def test_vulkan_p2fork_collects_vulkan_libs_and_headers(tmp_path: Path) -> None:
    _, _, done = _run_build(tmp_path, "vulkan-p2fork")
    assert done.returncode == 0, done.stderr
    names = {p.name for p in (tmp_path / "out" / "lib").iterdir()}
    assert "libggml-vulkan.so" in names
    assert not any(name.startswith("libggml-cuda") for name in names)
    assert (tmp_path / "out" / "include" / "llama.h").is_file()
    assert not (tmp_path / "out" / "bin").exists()


def test_vulkan_p2fork_refuses_a_tree_without_p2(tmp_path: Path) -> None:
    manifest, _, done = _run_build(tmp_path, "vulkan-p2fork", with_p2=False)
    assert done.returncode == 3
    assert "no P2 provenance" in done.stderr
    assert manifest == {}


@pytest.mark.parametrize(
    ("profile", "flag", "absent"),
    [
        ("p2fork", "-DGGML_CUDA=ON", "-DGGML_VULKAN=ON"),
        ("cpu-light", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=ON"),
    ],
)
def test_existing_profiles_keep_their_flags(tmp_path: Path, profile: str, flag: str, absent: str) -> None:
    manifest, log, done = _run_build(tmp_path, profile)
    assert done.returncode == 0, done.stderr
    assert flag in log.splitlines()[0]
    assert absent not in log.splitlines()[0]
    assert manifest["profile"] == profile


def test_profile_table_names_vulkan_p2fork_in_each_arm() -> None:
    text = _SCRIPT.read_text()
    # The P2 family arm and the Vulkan flag arm both list the profile.
    family = re.search(r'case "\$PROFILE" in\n\s+(p2fork \|[^)]*)\)\s+is_p2fork=1', text)
    assert family is not None
    assert "vulkan-p2fork" in family.group(1)
    vulkan = re.search(r"\n  (vulkan-mi25 \|[^)]*)\)\n", text)
    assert vulkan is not None
    assert "vulkan-p2fork" in vulkan.group(1)


def test_abi_gate_knows_the_profile() -> None:
    # The gate checks its profile list against the build script, so a missing entry fails at argparse.
    gate = (_ROOT / "ci" / "gates" / "cffi_bindgen.py").read_text()
    assert '"vulkan-p2fork": [],' in gate


def test_workflows_name_the_new_profile_for_gpu_library_cells() -> None:
    for name in ("build.yml", "suites.yml"):
        text = (_ROOT / "ci" / "workflows" / name).read_text()
        assert "profile: vulkan-p2fork" in text
        # No library cell builds the CUDA profile on the Vulkan runner any more.
        assert not re.search(r"profile: p2fork\n\s+runner: \[self-hosted, gpu, vulkan\]", text)
        assert "cuda" not in re.findall(r"runner: \[([^\]]*)\]", text)


_WORKFLOWS = _ROOT / "ci" / "workflows"
_CUDA_PROFILES = {"cuda-3060", "cuda-3060-server", "p2fork", "p2fork-server"}


def _vulkan_profiles() -> set[str]:
    """The profiles in the Vulkan flag arm of the build script."""
    arm = re.search(r"\n  (vulkan-mi25 \|[^)]*)\)\n", _SCRIPT.read_text())
    assert arm is not None
    return {part.strip() for part in arm.group(1).split("|")}


def _workflow(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text())


def _gpu_cells() -> list[tuple[str, str, str]]:
    """Return (workflow file, job, profile) for each GPU cell, matrix or literal."""
    cells: list[tuple[str, str, str]] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        for job_id, job in _workflow(path.name)["jobs"].items():
            matrix = job.get("strategy", {}).get("matrix", {}).get("include")
            if matrix:
                for cell in matrix:
                    if "gpu" in (cell.get("runner") or []) and cell.get("profile"):
                        cells.append((path.name, job_id, cell["profile"]))
            elif "gpu" in (job.get("runs-on") or []):
                text = "\n".join(str(step.get("run", "")) for step in job["steps"])
                for profile in re.findall(r"--profile\s+(\S+)", text):
                    cells.append((path.name, job_id, profile))
    return cells


def test_every_workflow_file_loads() -> None:
    names = sorted(p.name for p in _WORKFLOWS.glob("*.yml"))
    assert names
    for name in names:
        assert "jobs" in _workflow(name)


def test_the_vulkan_profile_arm_holds_no_cuda_profile() -> None:
    assert not (_vulkan_profiles() & _CUDA_PROFILES)


def test_gpu_cells_name_vulkan_profiles_only() -> None:
    cells = _gpu_cells()
    # build, bindgen (2), suites (4), bench: at least these cells exist.
    assert len(cells) >= 8
    vulkan = _vulkan_profiles()
    bad = [cell for cell in cells if cell[2] not in vulkan or cell[2] in _CUDA_PROFILES]
    assert not bad, f"GPU cells that name a profile this host cannot build: {bad}"


def test_server_cells_name_a_server_profile() -> None:
    server_cells = [c for c in _gpu_cells() if c[1] != "bench" and c[2].endswith("-server")]
    assert {c[0] for c in server_cells} == {"build.yml", "bindgen.yml", "suites.yml"}
    for _, _, profile in server_cells:
        assert profile == "vulkan-mi25-p2fork-server"


def test_bindgen_smoke_runs_on_the_vulkan_library_cell() -> None:
    steps = _workflow("bindgen.yml")["jobs"]["bindgen"]["steps"]
    smoke = [s for s in steps if str(s.get("name", "")).startswith("SMOKE")]
    assert len(smoke) == 1
    assert smoke[0]["if"] == "matrix.profile == 'vulkan-p2fork'"
    profiles = {c["profile"] for c in _workflow("bindgen.yml")["jobs"]["bindgen"]["strategy"]["matrix"]["include"]}
    assert "vulkan-p2fork" in profiles


def test_bindgen_gate_knows_every_profile_a_cell_names() -> None:
    gate = (_ROOT / "ci" / "gates" / "cffi_bindgen.py").read_text()
    for _, _, profile in _gpu_cells():
        assert f'"{profile}": [],' in gate


def test_server_suite_fails_loudly_without_nanbeige_inventory(tmp_path: Path) -> None:
    """The p2-server-seq-routing cell cannot pass or skip on a machine with no nanbeige model."""
    out = tmp_path / "out"
    (out / "lib").mkdir(parents=True)
    manifest = {"features": ["p2-mixed-batch-lora"], "profile": "vulkan-mi25-p2fork-server"}
    (out / "build-manifest.json").write_text(json.dumps(manifest))
    env = {k: v for k, v in os.environ.items() if not k.startswith("MODEL_NANBEIGE")}
    done = subprocess.run(
        [
            "bash",
            str(_ROOT / "ci" / "runner" / "run-suite.sh"),
            "--suite",
            "p2-server-seq-routing",
            "--out",
            str(out),
            "--library-python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 1
    assert "[suite-verdict] p2-server-seq-routing: FAIL" in done.stdout
    assert "inventory gap: MODEL_NANBEIGE is not set" in done.stderr


def _fork_clone() -> Path | None:
    candidates = [
        os.environ.get("INFERFERENCE_FORK_SRC", ""),
        str(_ROOT / "ci" / "build" / ".llamacpp-builds" / "src-p2fork"),
    ]
    for cand in candidates:
        if cand and (Path(cand) / ".git").exists():
            have = subprocess.run(["git", "-C", cand, "cat-file", "-e", f"{_ANCHOR}^{{commit}}"], capture_output=True)
            if have.returncode == 0:
                return Path(cand)
    return None


def test_residency_patch_applies_to_the_anchor_commit(tmp_path: Path) -> None:
    clone = _fork_clone()
    if clone is None:
        pytest.skip("no fork clone with the anchor commit (set INFERFERENCE_FORK_SRC)")
    tree = tmp_path / "tree"
    subprocess.run(["git", "-C", str(clone), "worktree", "add", "--detach", "-q", str(tree), _ANCHOR], check=True)
    try:
        subprocess.run(["git", "-C", str(tree), "apply", "--check", str(_PATCH)], check=True)
    finally:
        subprocess.run(["git", "-C", str(clone), "worktree", "remove", "--force", str(tree)], check=False)


_ALL_PROFILES = [
    "cpu-light",
    "cuda-3060",
    "cuda-3060-server",
    "p2fork",
    "p2fork-server",
    "vulkan-mi25",
    "vulkan-mi25-server",
    "vulkan-mi25-p2fork-server",
    "vulkan-p2fork",
]


def _flag_value(configure: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}=(.*?)(?= -D| -G|$)", configure)
    assert match is not None, f"{name} missing from: {configure}"
    return match.group(1)


def test_every_profile_in_the_table_is_covered() -> None:
    # A new profile must join the lists above, or the reproducibility tests skip it.
    text = _SCRIPT.read_text()
    arms = re.findall(r"\n  ([a-z0-9][a-z0-9 |-]*)\)\n", text)
    named = {part.strip() for arm in arms for part in arm.split("|")}
    assert set(_ALL_PROFILES) <= named


@pytest.mark.parametrize("profile", _ALL_PROFILES)
def test_every_profile_builds_without_a_lane_path(tmp_path: Path, profile: str) -> None:
    """One fake build per profile. Check the maps, the RUNPATH, the manifest and the glslc wrapper."""
    manifest, log, done = _run_build(tmp_path, profile)
    assert done.returncode == 0, done.stderr
    configure = log.splitlines()[0]

    # Compile: every build path maps to a fixed token, for the C and the C++ compiler.
    for name in ("-DCMAKE_C_FLAGS", "-DCMAKE_CXX_FLAGS"):
        value = _flag_value(configure, name)
        maps = re.findall(r"-ffile-prefix-map=(\S+?)=(/llama\.cpp/\S+)", value)
        sources = {old for old, new in maps if new == "/llama.cpp/src"}
        builds = {old for old, new in maps if new == "/llama.cpp/build"}
        # The compile link, the real source tree and the build dir are each mapped.
        assert any(old.endswith("/.llamacpp-builds/src-active") for old in sources), value
        assert any("/.llamacpp-builds/build-" in old for old in builds), value
        assert any(old.endswith("/fork") for old in sources), value
        assert {new for _, new in maps} == {"/llama.cpp/src", "/llama.cpp/build"}
    if "cuda" in profile or profile.startswith("p2fork"):
        cuda = _flag_value(configure, "-DCMAKE_CUDA_FLAGS")
        assert "-Xcompiler -ffile-prefix-map=" in cuda
        assert "=/llama.cpp/src" in cuda
        assert "=/llama.cpp/build" in cuda
    # No other flag holds a lane path.
    stripped = re.sub(r"-ffile-prefix-map=\S+|-DVulkan_GLSLC_EXECUTABLE=\S+", "", configure)
    assert str(tmp_path) not in stripped.replace(f"-B {tmp_path}", "").replace(f"-S {tmp_path}", "")

    # Manifest: no lane path, and it names the tokens.
    assert str(tmp_path) not in manifest["cmake_flags"]
    assert "-ffile-prefix-map" not in manifest["cmake_flags"]
    assert manifest["path_prefix_maps"] == "/llama.cpp/src /llama.cpp/build"

    # RUNPATH: cmake writes none, and every collected file gets $ORIGIN first.
    assert "-DCMAKE_SKIP_BUILD_RPATH=ON" in configure
    lines = (tmp_path / "patchelf.log").read_text().splitlines()
    lib_names = {p.name for p in (tmp_path / "out" / "lib").iterdir() if p.is_file() and not p.is_symlink()}
    seen = set()
    for line in lines:
        runpath, path = line.rsplit(" ", 1)
        name = Path(path).name
        seen.add(name)
        # $ORIGIN first, the nix-store entries kept once, no empty entry, no build path.
        origin = "$ORIGIN/../lib" if name == "llama-server" else "$ORIGIN"
        assert runpath == f"{origin}:/nix/store/aaa-glibc/lib:/nix/store/bbb-gcc-lib/lib", runpath
        assert str(tmp_path) not in runpath
    assert lib_names <= seen
    assert ("llama-server" in seen) == profile.endswith("-server")

    # glslc: only the Vulkan profiles route it through the fixed-length wrapper.
    if profile.startswith("vulkan"):
        assert "-DVulkan_GLSLC_EXECUTABLE=" in configure
        assert "ci/build/glslc-stable.sh" in configure
    else:
        assert "Vulkan_GLSLC_EXECUTABLE" not in configure


def test_runpath_check_refuses_a_leftover_build_path(tmp_path: Path) -> None:
    # A patchelf that keeps the build path must make the build fail, not ship.
    fork = _make_fork(tmp_path)
    script_dir = tmp_path / "repo" / "ci" / "build"
    script_dir.mkdir(parents=True)
    shutil.copy(_SCRIPT, script_dir / "build-llamacpp.sh")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("cmake", _FAKE_CMAKE),
        ("patchelf", '#!/usr/bin/env bash\n[[ "$1" == --print-rpath ]] && echo "$FAKE_LEAK" || true\n'),
    ):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_CMAKE_LOG": str(tmp_path / "cmake.log"),
        "FAKE_LEAK": f"$ORIGIN:{script_dir}/.llamacpp-builds/build-x/bin",
    }
    done = subprocess.run(
        [
            "bash",
            str(script_dir / "build-llamacpp.sh"),
            "--ref",
            f"local:{fork}",
            "--profile",
            "vulkan-p2fork",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 3
    assert "RUNPATH that depends on the build location" in done.stderr
    assert not (tmp_path / "out" / "build-manifest.json").exists()


def test_the_build_refuses_a_path_with_whitespace(tmp_path: Path) -> None:
    spaced = tmp_path / "a lane"
    spaced.mkdir()
    _, _, done = _run_build(spaced, "vulkan-p2fork")
    assert done.returncode == 3
    assert "whitespace" in done.stderr


def test_ccache_ignores_the_lane_path_in_the_prefix_map() -> None:
    # The map names the lane. ccache must leave it out of the hash, or every lane misses the cache.
    text = _SCRIPT.read_text()
    assert 'export CCACHE_IGNOREOPTIONS="-ffile-prefix-map=*"' in text
    assert 'export CCACHE_BASEDIR="$work"' in text


def test_the_wheel_build_maps_the_vendored_source_path() -> None:
    text = (_ROOT / "ci" / "build" / "mode-a-build.sh").read_text()
    assert "-ffile-prefix-map=${lcpp}=/llama-cpp-python" in text
    assert 'export CFLAGS="' in text
    assert 'export CXXFLAGS="' in text


_VULKAN = [p for p in _ALL_PROFILES if p.startswith("vulkan")]


def test_the_wrapper_gives_glslc_paths_of_one_length_in_every_lane(tmp_path: Path) -> None:
    """glslc output depends on the length of its path arguments. The wrapper fixes the length."""
    wrapper = _ROOT / "ci" / "build" / "glslc-stable.sh"
    assert os.access(wrapper, os.X_OK)
    fake = tmp_path / "glslc-real"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$@" >> "$FAKE_GLSLC_LOG"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  [[ "$1" == -o ]] && echo spv > "$2"\n'
        '  [[ "$1" == -MF ]] && echo "out.spv: $DEP_PREFIX/types.glsl" > "$2"\n'
        "  shift\n"
        "done\n"
    )
    fake.chmod(0o755)
    lengths = set()
    for lane in ("zq1", "zq1-much-longer-lane-name"):
        root = tmp_path / lane / "ci" / "build" / ".llamacpp-builds"
        shaders = root / "src-active" / "ggml" / "vulkan-shaders"
        out = root / "build-x" / "spv"
        shaders.mkdir(parents=True)
        out.mkdir(parents=True)
        (shaders / "tri.comp").write_text("void main() {}\n")
        log = tmp_path / f"{lane}.log"
        env = {
            **os.environ,
            "GLSLC_REAL": str(fake),
            "FAKE_GLSLC_LOG": str(log),
            "DEP_PREFIX": "SEEN",
        }
        done = subprocess.run(
            [
                str(wrapper),
                "-fshader-stage=compute",
                str(shaders / "tri.comp"),
                "-o",
                str(out / "tri_f16.spv"),
                "-MD",
                "-MF",
                str(out / "tri.d"),
                "-DA=1",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert done.returncode == 0, done.stderr
        # The real output and the dependency file land in the real dirs.
        assert (out / "tri_f16.spv").read_text() == "spv\n"
        assert (out / "tri.d").is_file()
        words = log.read_text().split()
        paths = [w for w in words if w.startswith("/")]
        assert len(paths) == 3
        assert all(lane not in w for w in paths)
        lengths.add(tuple(len(w) for w in paths))
    assert len(lengths) == 1


def test_the_wrapper_resolves_relative_paths_before_it_changes_directory(tmp_path: Path) -> None:
    """The wrapper runs glslc from a temp dir. A relative path in the call must still resolve."""
    work = tmp_path / "work"
    (work / "shaders" / "inc").mkdir(parents=True)
    (work / "out").mkdir()
    (work / "shaders" / "tri.comp").write_text("void main() {}\n")
    (work / "bin").mkdir()
    fake = work / "bin" / "glslc-real"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$@" >> "$FAKE_GLSLC_LOG"\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  [[ "$1" == -I ]] && { [[ -d "$2" ]] || exit 7; }\n'
        '  [[ "$1" == -I?* ]] && { [[ -d "${1#-I}" ]] || exit 8; }\n'
        '  [[ "$1" == -o ]] && echo spv > "$2"\n'
        "  shift\n"
        "done\n"
    )
    fake.chmod(0o755)
    log = tmp_path / "rel.log"
    env = {**os.environ, "GLSLC_REAL": "bin/glslc-real", "FAKE_GLSLC_LOG": str(log)}
    wrapper = _ROOT / "ci" / "build" / "glslc-stable.sh"
    done = subprocess.run(
        [
            str(wrapper),
            "-Ishaders/inc",
            "-I",
            "shaders",
            "shaders/tri.comp",
            "-o",
            "out/tri.spv",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    assert (work / "out" / "tri.spv").read_text() == "spv\n"
    words = log.read_text().split()
    assert f"-I{work.resolve()}/shaders/inc" in words
    assert str((work / "shaders").resolve()) in words


def test_the_wrapper_passes_a_probe_through_unchanged(tmp_path: Path) -> None:
    fake = tmp_path / "glslc-real"
    fake.write_text('#!/usr/bin/env bash\necho "$@" > "$FAKE_GLSLC_LOG"\n')
    fake.chmod(0o755)
    log = tmp_path / "probe.log"
    shader = tmp_path / "probe.comp"
    shader.write_text("void main() {}\n")
    env = {**os.environ, "GLSLC_REAL": str(fake), "FAKE_GLSLC_LOG": str(log)}
    wrapper = _ROOT / "ci" / "build" / "glslc-stable.sh"
    done = subprocess.run(
        [str(wrapper), "-o", "-", "-fshader-stage=compute", str(shader)], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert log.read_text().split() == ["-o", "-", "-fshader-stage=compute", str(shader)]
