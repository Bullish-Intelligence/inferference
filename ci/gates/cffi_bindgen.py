#!/usr/bin/env python3
"""cffi API-mode bindgen gate — the ABI-drift tripwire (first-class gate).

Promoted from an earlier build spike to a first-class gate in ci/gates/. It compiles a cffi API-mode CPython
extension whose C source `#include "llama.h"` from THIS build's include/ and
links against THIS build's libllama.so. The C compiler — not a hand-written
ctypes mirror — fills in struct offsets, enum values, and signatures.

    Compile failure == ABI drift == STOP.

There is no continue-on-error: a signature/struct change, a renamed/removed
symbol, or a missing header fails the compile loudly. The extension and the lib
it was compiled against are one paired unit.

The cdef is the spike's real minimal llama.h surface (backend lifecycle,
llama_batch — the ABI-critical struct — decode/logits, vocab, samplers, state
save/restore, LoRA) plus the two P2 fork routing symbols. The fork symbols are
declared unconditionally: the binding is only valid for a lib that exports them
(importing it against a stock lib fails at dlopen with an undefined-symbol
error — capability absent, fail-closed, not a silent pass).

Usage (run inside `devenv shell`, gates venv python, with the build output):
    ci/gates/.venv/bin/python ci/gates/cffi_bindgen.py \
        --build-dir .llamacpp-builds/out-p2fork-<sha> --profile p2fork --check
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from cffi import FFI  # type: ignore

MODULE_NAME = "_llama_cffi"

# --- profile -> extra preprocessor defines -----------------------------------
# Keep in lockstep with ci/build/build-llamacpp.sh's flag profiles. These are a
# starting point; the first real compile against a given header will tell you if
# the header needs a backend define to expose/hide a declaration.
PROFILE_DEFINES: dict[str, list[tuple[str, str | None]]] = {
    # CPU-only lib: no CUDA symbols compiled into ggml; nothing extra needed.
    "cpu-light": [],
    # CUDA libs for the 3060s (sm_86) — cuda-3060 and p2fork are flag-identical
    # . GGML_USE_CUDA gates CUDA-only bits in
    # some ggml headers; forward it so the binding's header view matches the lib.
    "cuda-3060": [("GGML_USE_CUDA", "1")],
    "p2fork": [("GGML_USE_CUDA", "1")],
    # The *-server profiles are LIB-flag-identical to their base: -DLLAMA_BUILD_
    # SERVER=ON adds build targets and changes nothing about libllama's compiled
    # surface, so the binding's header view is the base profile's.
    #
    # They were absent until 2026-08-26, and the bindgen cell added to
    # bindgen.yml could therefore never have run: argparse rejected the value
    # before any header was read. Found by running that cell under Act for the
    # first time. The lockstep
    # this comment claims was not enforced anywhere, and now it is, below.
    "cuda-3060-server": [("GGML_USE_CUDA", "1")],
    "p2fork-server": [("GGML_USE_CUDA", "1")],
    # AMD MI25 (Vega 10 / gfx900) Vulkan libs. No define is forwarded, and that
    # is deliberate rather than an omission: GGML_USE_VULKAN gates nothing in the
    # PUBLIC headers this cdef compiles against. llama.h and ggml.h describe the
    # same ABI whichever backend is compiled in — which is the whole reason a
    # Vulkan build of the anchor stays usable by llama-cpp-python 0.3.34's
    # ctypes. If a future ref makes a public struct backend-conditional, this
    # gate fails to compile, which is exactly the drift tripwire firing.
    "vulkan-mi25": [],
    "vulkan-mi25-server": [],
    # The P2 fork's tree built for Vulkan. Vulkan, so no define,
    # for the same reason as the two above. It is NOT given GGML_USE_CUDA even
    # though the sibling `p2fork` profiles are: those forward it because they are
    # CUDA builds, not because they are fork builds. The fork axis and the
    # backend axis are independent, and this entry is where that is easy to get
    # wrong.
    #
    # This lib DOES carry the fork symbols, so run the gate WITHOUT
    # --no-fork-symbols against it; that flag is for probing a stock tree.
    "vulkan-mi25-p2fork-server": [],
}


def profiles_in(body: str) -> set[str]:
    """Every profile named by a `case "$PROFILE" in` arm in build-llamacpp.sh.

    The script has more than one such block (one derives `is_p2fork`, one holds
    the cmake flags), so take the union. Arms carry trailing code and `esac` sits
    at column 0.
    """
    found: set[str] = set()
    for block in re.findall(r'case "\$PROFILE" in\n(.*?)\nesac', body, re.S):
        for line in block.splitlines():
            m = re.match(r"\s+([a-zA-Z0-9|\-_ ]+?)\)", line)
            if m and "*" not in m.group(1):
                found.update(part.strip() for part in m.group(1).split("|") if part.strip())
    return found


def _assert_profiles_in_lockstep() -> None:
    """Every build-llamacpp.sh profile must have defines here.

    A profile this file does not know is an argparse rejection at gate time,
    which reads as a broken gate rather than as drift. Parse the build script's
    own `case "$PROFILE"` arms so the two lists cannot separate again silently.
    """
    script = Path(__file__).resolve().parents[1] / "build" / "build-llamacpp.sh"
    if not script.is_file():
        return
    known = profiles_in(script.read_text())
    if not known:
        # The script exists but no `case "$PROFILE" in` block parsed. That is a
        # broken check, not a clean bill of health, so say so.
        raise SystemExit(
            f"could not read any profile out of {script}. This check cannot "
            f"confirm the profile lists are in lockstep; fix it rather than skip it."
        )
    missing = known - set(PROFILE_DEFINES)
    if missing:
        raise SystemExit(
            f"build-llamacpp.sh has profile(s) {sorted(missing)} that "
            f"PROFILE_DEFINES does not. Add them, or a gate cell naming one "
            f"fails at argparse and reads as a broken gate rather than as drift."
        )


# ------------------------------------------------------------------ cdef ------
# The spike's real minimal llama.h cdef (validated against the anchor 0ab9d6fed
# / llama-cpp-python 0.3.34, see README-cffi-bindgen.md), plus the two fork
# symbols. `...` (dotdotdot) => "compiler, fill this in": param structs passed
# by value and the llama_batch struct get their exact layout from the compiler;
# the STATE_SEQ flag macros get their values. Opaque handles are bodyless
# `struct` types — we only ever hold pointers to them.
CDEF = r"""
    /* --- scalar typedefs (stable; int32 in every ref to date) --- */
    typedef int32_t  llama_pos;
    typedef int32_t  llama_token;
    typedef int32_t  llama_seq_id;
    typedef uint32_t llama_state_seq_flags;

    /* --- opaque handles: pointer-only, no layout needed --- *
     * The real llama.h declares these as bare `struct X` with NO bare-name
     * typedef, so we must reference them as `struct X` throughout (a plain
     * `typedef struct llama_model llama_model;` here makes cffi emit an
     * undefined bare type -> compile error; that mismatch is caught, not run). */
    struct llama_model;
    struct llama_context;
    struct llama_vocab;
    struct llama_sampler;
    struct llama_adapter_lora;

    /* --- param structs: passed BY VALUE, layout supplied by compiler --- *
     * Also bare `struct X` in the header (no bare-name typedef). */
    struct llama_model_params          { ...; };
    struct llama_context_params        { ...; };
    struct llama_sampler_chain_params  { ...; };

    /* --- llama_batch: real fields, but let compiler own the exact layout --- *
     * The named fields are the ones we touch on the hot path (decode loop).
     * `...;` lets cffi verify/pad against the true struct. Confirmed fields in
     * the anchor (include/llama.h:244-253): n_tokens, token, embd, pos,
     * n_seq_id, seq_id, logits. */
    typedef struct llama_batch {
        int32_t        n_tokens;
        llama_token   *token;
        float         *embd;
        llama_pos     *pos;
        int32_t       *n_seq_id;
        llama_seq_id **seq_id;
        int8_t        *logits;
        ...;
    } llama_batch;

    /* --- STATE_SEQ flag macros: values supplied by compiler --- */
    #define LLAMA_STATE_SEQ_FLAGS_NONE ...
    #define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY ...
    #define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY ...
    #define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE ...

    /* --- backend lifecycle --- */
    void llama_backend_init(void);
    void llama_backend_free(void);
    const char * llama_print_system_info(void);
    bool llama_supports_gpu_offload(void);

    /* --- default params (return structs by value) --- */
    struct llama_model_params         llama_model_default_params(void);
    struct llama_context_params       llama_context_default_params(void);
    struct llama_sampler_chain_params llama_sampler_chain_default_params(void);

    /* --- model / context lifecycle --- */
    struct llama_model *   llama_model_load_from_file(const char *path_model, struct llama_model_params params);
    void                   llama_model_free(struct llama_model *model);
    struct llama_context * llama_init_from_model(struct llama_model *model, struct llama_context_params params);
    void                   llama_free(struct llama_context *ctx);
    uint32_t               llama_n_ctx(const struct llama_context *ctx);

    /* --- vocab --- */
    const struct llama_vocab * llama_model_get_vocab(const struct llama_model *model);
    int32_t                    llama_vocab_n_tokens(const struct llama_vocab *vocab);
    bool                       llama_vocab_is_eog(const struct llama_vocab *vocab, llama_token token);
    llama_token                llama_vocab_bos(const struct llama_vocab *vocab);
    llama_token                llama_vocab_eos(const struct llama_vocab *vocab);

    /* --- decode + zero-copy logits --- */
    llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    llama_batch llama_batch_get_one(llama_token *tokens, int32_t n_tokens);
    void        llama_batch_free(llama_batch batch);
    int32_t     llama_decode(struct llama_context *ctx, llama_batch batch);
    float *     llama_get_logits_ith(struct llama_context *ctx, int32_t i);

    /* --- tokenize / detokenize --- */
    int32_t llama_tokenize(const struct llama_vocab *vocab, const char *text, int32_t text_len,
                           llama_token *tokens, int32_t n_tokens_max,
                           bool add_special, bool parse_special);
    int32_t llama_token_to_piece(const struct llama_vocab *vocab, llama_token token,
                                 char *buf, int32_t length, int32_t lstrip, bool special);

    /* --- sampler chain (owned decode loop) --- */
    struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);
    void                   llama_sampler_chain_add(struct llama_sampler *chain, struct llama_sampler *smpl);
    struct llama_sampler * llama_sampler_init_greedy(void);
    struct llama_sampler * llama_sampler_init_dist(uint32_t seed);
    struct llama_sampler * llama_sampler_init_top_k(int32_t k);
    struct llama_sampler * llama_sampler_init_top_p(float p, size_t min_keep);
    struct llama_sampler * llama_sampler_init_temp(float t);
    llama_token            llama_sampler_sample(struct llama_sampler *smpl, struct llama_context *ctx, int32_t idx);
    void                   llama_sampler_accept(struct llama_sampler *smpl, llama_token token);
    void                   llama_sampler_free(struct llama_sampler *smpl);

    /* --- per-sequence state save/restore (cache pillar) --- */
    size_t llama_state_seq_get_size(struct llama_context *ctx, llama_seq_id seq_id);
    size_t llama_state_seq_get_data(struct llama_context *ctx, uint8_t *dst, size_t size,
                                    llama_seq_id seq_id);
    size_t llama_state_seq_set_data(struct llama_context *ctx, const uint8_t *src, size_t size,
                                    llama_seq_id dest_seq_id);

    /* --- LoRA adapters --- */
    struct llama_adapter_lora * llama_adapter_lora_init(struct llama_model *model, const char *path_lora);
    void                        llama_adapter_lora_free(struct llama_adapter_lora *adapter);
    int32_t                     llama_set_adapters_lora(struct llama_context *ctx, struct llama_adapter_lora **adapters,
                                                        size_t n_adapters, float *scales);

    /* --- P2 fork: mixed-batch per-sequence LoRA routing --- *
     * Fork-only symbols added by the fork's feat/p2-mixed-batch-lora branch
     * (include/llama.h:699-712, commit 108d4d756). Declared unconditionally:
     * the binding is paired with the fork surface; a lib that lacks them is not
     * the lib this binding was built against. */
    int32_t llama_set_seq_adapters(struct llama_context *ctx, struct llama_adapter_lora **adapters, size_t n_adapters);
    int32_t llama_set_seq_adapter(struct llama_context *ctx, llama_seq_id seq_id, int32_t adapter_idx);
"""


# Marker: everything from this comment to the end of the cdef is the P2 fork
# surface. --no-fork-symbols strips it for the canary (probe against STOCK
# upstream, which lacks the symbols; compile then tests the real ABI surface).
FORK_SECTION = "/* --- P2 fork: mixed-batch per-sequence LoRA routing --- *"


def build(include_dir: Path, lib_dir: Path, profile: str, out_dir: Path, no_fork_symbols: bool = False) -> Path:
    if profile not in PROFILE_DEFINES:
        raise SystemExit(f"unknown profile {profile!r}; known: {sorted(PROFILE_DEFINES)}")

    # cffi compiles from a tmpdir (out_dir) with the include dir passed as a
    # relative -I path: a relative --build-dir would point at nothing from
    # that cwd (fatal error: llama.h: No such file or directory — field finding
    # 2026-08-02). Resolve to absolute so relative --build-dir works, and so
    # the embedded rpath below names a real directory.
    include_dir = include_dir.resolve()
    lib_dir = lib_dir.resolve()

    header = include_dir / "llama.h"
    if not header.is_file():
        raise SystemExit(
            f"llama.h not found under {include_dir} .\n"
            "  The header is required for an API-mode compile. Run "
            "ci/build/build-llamacpp.sh to produce a checkout+build (emits "
            "<out>/include and <out>/lib), then point --build-dir at it."
        )

    cdef = CDEF
    if no_fork_symbols:
        # Stock upstream (the canary's target) does not declare the fork symbols;
        # the fork-paired binding must NOT be compiled against it. Stripping the
        # fork section turns this into a pure ABI-surface probe: compile failure
        # then means upstream moved a struct/signature — the red canary.
        cdef = cdef.split(FORK_SECTION)[0]

    ffibuilder = FFI()
    ffibuilder.cdef(cdef)
    ffibuilder.set_source(
        MODULE_NAME,
        '#include "llama.h"',
        libraries=["llama"],
        include_dirs=[str(include_dir)],
        library_dirs=[str(lib_dir)],
        # embed an rpath so the built .so finds libllama + its ggml deps at import
        # without needing LD_LIBRARY_PATH for THEM (nix libstdc++ is separate).
        extra_link_args=[f"-Wl,-rpath,{lib_dir}"],
        define_macros=[(name, val) for name, val in PROFILE_DEFINES[profile]],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    so_path = ffibuilder.compile(tmpdir=str(out_dir), verbose=True)
    return Path(so_path)


def self_check(out_dir: Path, no_fork_symbols: bool = False) -> None:
    """Import the freshly built module and self-test.

    Importing dlopens the extension; if the lib at the embedded rpath lacks any
    symbol the cdef declared (e.g. the fork routing symbols on a stock lib), the
    import fails with an undefined-symbol error. That is fail-closed capability
    absence — we report it clearly and exit non-zero; it is NOT an ABI-drift
    signal (a compile failure is).
    """
    sys.path.insert(0, str(out_dir))
    try:
        mod = __import__(MODULE_NAME)
    except ImportError as exc:
        raise SystemExit(
            f"[check] FAIL — the built binding cannot be imported: {exc}\n"
            "  If the message is an undefined symbol (e.g. llama_set_seq_*), the lib\n"
            "  lacks the fork surface the cdef declares — run the bindgen against a\n"
            "  p2fork build. If the message is a version/ABI error, that IS the\n"
            "  ABI-drift signal: STOP and reconcile the cdef with the header."
        ) from exc
    lib = mod.lib
    ffi = mod.ffi

    lib.llama_backend_init()
    info = ffi.string(lib.llama_print_system_info()).decode(errors="replace")
    gpu = bool(lib.llama_supports_gpu_offload())
    lib.llama_backend_free()
    print(f"[check] llama_backend_init/free OK; supports_gpu_offload={gpu}")
    print(f"[check] system info: {info.strip()[:120]}...")

    # Fork surface probe: both declared symbols must be bound on the loaded lib.
    # Skipped under --no-fork-symbols (canary probe against stock upstream,
    # where the symbols are intentionally not declared).
    if not no_fork_symbols:
        for name in ("llama_set_seq_adapters", "llama_set_seq_adapter"):
            if not hasattr(lib, name):
                print(f"[check] FAIL — declared fork symbol {name} not bound on the loaded lib", flush=True)
                raise SystemExit(1)
        print("[check] P2 fork symbols bound: llama_set_seq_adapters, llama_set_seq_adapter")


def main(argv: list[str] | None = None) -> int:
    # Before argparse builds its choices from PROFILE_DEFINES, so a profile the
    # build script has and this file does not is named as drift rather than
    # rejected as an invalid choice.
    _assert_profiles_in_lockstep()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", type=Path, help="build-llamacpp.sh output dir (expects <dir>/include and <dir>/lib)")
    ap.add_argument("--include-dir", type=Path, help="override include dir (has llama.h)")
    ap.add_argument("--lib-dir", type=Path, help="override lib dir (has libllama.so)")
    ap.add_argument(
        "--profile",
        default="cpu-light",
        choices=sorted(PROFILE_DEFINES),
        help="match the build-llamacpp.sh profile the lib was built with",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="where to emit _llama_cffi*.so (default: this dir)",
    )
    ap.add_argument("--check", action="store_true", help="import the built module and self-test")
    ap.add_argument(
        "--no-fork-symbols",
        action="store_true",
        help="strip the P2 fork symbol decls from the cdef — canary probe against "
        "stock upstream, which lacks them; compile failure then means the stock "
        "ABI surface moved (red canary), not fork-capability absence",
    )
    args = ap.parse_args(argv)

    if args.include_dir and args.lib_dir:
        include_dir, lib_dir = args.include_dir, args.lib_dir
    elif args.build_dir:
        include_dir = args.build_dir / "include"
        lib_dir = args.build_dir / "lib"
    elif os.environ.get("LLAMA_CFFI_INCLUDE") and os.environ.get("LLAMA_CFFI_LIB"):
        include_dir = Path(os.environ["LLAMA_CFFI_INCLUDE"])
        lib_dir = Path(os.environ["LLAMA_CFFI_LIB"])
    else:
        ap.error("supply --build-dir or --include-dir + --lib-dir (or LLAMA_CFFI_INCLUDE / LLAMA_CFFI_LIB env vars)")

    print(f"[build] include={include_dir}")
    print(f"[build] lib    ={lib_dir}")
    print(f"[build] profile={args.profile}  defines={PROFILE_DEFINES[args.profile]}")

    so_path = build(include_dir, lib_dir, args.profile, args.out_dir, no_fork_symbols=args.no_fork_symbols)
    print(f"[build] compiled extension -> {so_path}")

    if args.check:
        self_check(args.out_dir, no_fork_symbols=args.no_fork_symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
