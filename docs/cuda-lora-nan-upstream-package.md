# Upstream report package — CUDA LoRA NaN (ggml-org/llama.cpp)

> **This is a bug-report artifact, not platform documentation.** Inferference is
> model-agnostic; the specific models named below are REPRODUCTION EVIDENCE for
> upstream — "this defect appeared on these architectures, with these tensor
> shapes". Removing them would weaken the report. Nothing here describes how the
> platform works or what it is for.

**Update (2026-08-03, session 002):** the fix is now implemented, stress-tested,
and merged on our fork: `upstream-candidate/cuda-lora-nan-fix` @ `04a8cc3aa`
(merged into `integration` as `24c5d3dbc`). This package is the owner's filing
material — the branch + evidence, ready to push and file under the owner's
ownership (upstream's AGENTS.md forbids AI-authored upstream contribution; an
AI session never pushes upstream).

**Prepared as a FILE for the owner to file upstream.** Nothing in this repo is
pushed to `ggml-org/llama.cpp`. The owner reviews, trims, and files this under
their own authorship.

## Summary

Decoding with a full-coverage LoRA adapter on CUDA can produce **all-NaN
logits**, deterministically, when any input activation exceeds a fixed
magnitude threshold. The cause is an fp16 overflow in the **Q8_1
activation-quantization scale** used by the CUDA GEMV path for quantized
weights. CPU is always correct; the failure is value-dependent, not
graph-count-dependent, and affects both tested anchors (b10103, b10233).

## Minimal repro

1. Build any recent llama.cpp with CUDA (reproduced on b10103 `c588c4f47` and
   b10233 `0ab9d6fed`; kernel line identical in both).
2. Load a quantized model whose arch has an ffn with a large inter-dim
   (reproduced on Nanbeige4.2-3B, 22 layers × ffn_up/gate [3072, 10752], and
   on Ornith-1.0-9B (qwen35), 32 layers × ffn_up/gate [4096, 12288]).
3. Apply a synthetic full-coverage LoRA: every layer, `ffn_up` + `ffn_gate`
   pairs, rank 16, `alpha = 64` (effective scale 4.0 per pair), weights
   `~N(0,1)` (the committed synthesizer
   `ci/canary/synth_nanbeige_lora.py --weights ffn_up,ffn_gate` reproduces the
   GGUF shape contract).
4. Prefill ≥ 1 token, batch-1, `llama_set_adapters_lora` (standard path, no
   fork features), logits on the last token, `n_gpu_layers=-1`.

Observed: first-token logits all-NaN. Token-sequence dependent: e.g. on the
nanbeige model, tokens `[461, 261]` NaN, `[461, 261, 20692, 22744]` clean,
`[461, 261, 20692, 22744, 152363]` NaN; 8× token 42 is clean at every length.

## Localization

With a per-node eval callback (`llama_context_params.cb_eval` +
`ggml_backend_tensor_get`), the first non-finite node is the **ffn_down
mul_mat** at the last physical layer of the first loop pass (`ffn_out-21`,
token index 1 only). Its inputs are finite: the swiglu activation for that
token reaches max-abs 9,426,831.

## Root cause

`ggml/src/ggml-cuda/ggml-cuda.cu` → `ggml_cuda_mul_mat_vec_q` (the GEMV path
for quantized weights × small batch) quantizes the f32 input activation to
Q8_1 on the fly (`quantize_row_q8_1_cuda`, `mmvq.cu:1236`).

- `block_q8_1` stores its scale as fp16: `ggml/src/ggml-common.h:268`
  (`ggml_half d; ggml_half s;`).
- The kernel writes `y[ib].ds = make_half2(d, sum)` with `d = amax/127`:
  `ggml/src/ggml-cuda/quantize.cu:100`.
- When a 32-value block has `amax > 127 * 65504 = 8,319,008`, `d` overflows
  fp16 → inf.
- `vec_dot_q4_K` multiplies by that inf (`ggml/src/ggml-cuda/vecdotq.cuh:133-136`),
  and the block reduction produces `inf - inf` → NaN.

Threshold verified token-by-token on the failing case: the NaN token's swiglu
max-abs (9,426,831) is the only one above 8,319,008; all clean tokens are
below.

## Notes for the fix discussion

- Widening `block_q8_1.d/s` to fp32 (CUDA path) removes the overflow but
  touches the quantize/mmvq/mmq/vecdot kernels and their buffer sizing
  (perf/bit-exactness tradeoff for upstream to decide).
- A cheap clamp (`d = fminf(d, 65504)`) removes the NaN but is numerically
  wrong (not bit-exact vs CPU).
- The CPU backend multiplies f32 activations directly (no Q8_1 input
  quantization) — hence CPU-only correctness.
- Real trained LoRAs have delta scales ≪ 1 and do not approach the threshold;
  the failure needs activation magnitudes > 8.3e6.

## The fix (branch `upstream-candidate/cuda-lora-nan-fix`, commit `04a8cc3aa`)

Store the CUDA *activation* Q8_1 scales `d` and `s` in fp32 in the scratch
layouts only (the on-disk `block_q8_1` GGUF layout is untouched):

- `ggml-common.h` — new `block_q8_1_f32` (`float d; float s; int8 qs[32]`,
  40 B) for the GEMV (mmvq) activation scratch.
- `quantize.cu` — `quantize_q8_1` writes fp32 d/s; `quantize_mmq_q8_1` writes
  `make_float2` into the DS4 slots (D2S6 becomes fp32 too).
- `mmvq.cu` — scratch alloc, the `vec_dot_q_cuda_t` dispatch, and both
  kernels use `block_q8_1_f32`.
- `vecdotq.cuh` — all 23 dispatch consumers read fp32 d/s (`half2 ds8` →
  `float2 ds8`; `__low2float(ds)` → `.d`).
- `mmq.cuh` — `block_q8_1_mmq` union: `half2 ds4[4]` → `float2 ds4[4]`,
  `half d2s6[8]` → `float d2s6[8]`; Y-tile stride `MMQ_TILE_Y_K` 36 → 40 ints;
  fp4 (Blackwell) keeps its 16-byte scale area via `MMQ_TILE_Y_FP4_K`.
- `mmq-vec-dot.cuh` — y-side readers use the float2 row stride
  (`MMQ_TILE_Y_K/2`) and `y_qs` + 8.
- `mmq.cu` — the two per-channel stride `s12` formulas use
  `sizeof(block_q8_1_f32)`.

Why not the alternatives: the raw
block sum `s` overflows fp16 at ~32× lower magnitudes than `d`, so a
power-of-two pre-scale (Option C) either needs a headroom-killing k or stays
broken; bf16 would be a healthy-path precision regression; clamp is wrong.

**Validation (full battery green, merged integration `24c5d3dbc`):**

- Reproducer clean on all trigger adapters (nanbeige probe-a/upgate-L22,
  Ornith) — was all-NaN (166144) — GPU, batch-1 and MMQ batch-64, with and
  without `CUDA_LAUNCH_BLOCKING=1`.
- First-token logits finite, argmax == CPU, `max|gpu−cpu| = 0.226`.
- Applies cleanly to current `upstream/master` (`ee0445c99`); a pure-stock
  build of master+patch runs the reproducer clean (nanbeige + Ornith).
- Bindgen tripwire green (full cdef), ABI smoke PASS, suites PASS
  (p2 17/17, hats 6/6, router 4/4, prefix-cache 2/2, nanbeige), `testee
  verify` green, bench cells green, p2_throughput 69.8 tps vs 64.8 (step-5)
  — same band.

The kernel lines cited above are byte-identical between b10233 (`0ab9d6fed`)
and current upstream master — the bug is open upstream, and this patch is the
proposed fix.
