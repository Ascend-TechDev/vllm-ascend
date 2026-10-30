# MXFP QAT Rollout Fake Quantization

The existing unquantized linear/MoE path supports `w8a16_mxfp8`, `w8a8_mxfp8`,
`w4a8_mxfp4`, and `w4a4_mxfp4`. MXFP4 uses E2M1 weights, E8M0 scales, and groups
of 32; W4A8 uses E4M3 activations and W4A4 uses E2M1 activations. Matmuls remain
BF16/FP16, with unpacked FP32 quantize/dequantize intermediates suitable for A2/A3.

Set `actor_rollout_ref.rollout.qat_fake_quant=True` in verl. Use actor QAT mode
`w4a8_mxfp4` or `w4a4_mxfp4`, `group_size=32`, and `quant_backend=torch`.
The actor's renamed, format-independent QAT options are passed through the
existing `VLLM_ASCEND_QAT_MXFP8_*` environment variables. Their historical names
are retained for compatibility and now cover MXFP4 too. These variables are
registered centrally in `vllm_ascend/envs.py`.

Weight rounding supports `rint`, `round`, `random`, and deterministic `hash`.
Forward activations use nearest-even `rint`, matching verl dense layers and the
new MXFP4 packed-expert modes. Rotation uses the same 32-element Hadamard/sign
matrix before quantization. Fallback/ignore patterns skip complete selected
layers; experts disabled in verl are included in the exported ignore patterns.

`VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS` is a non-sensitive JSON list, default
`["Fprop"]`. Only Fprop controls rollout quantization; Dgrad/Wgrad remain training
targets. Rotation targets remain independent. Weight quantization/rotation is
cached once after loading and refreshed after each actor weight synchronization.
Existing cache invalidation and transpose handling are shared by both formats.

The MXFP4 formula follows
[ops-nn DynamicMxQuant golden.py](https://gitcode.com/cann/ops-nn/blob/8b566a55cd41ee887aa4b82c3fc560c7010ef2c8/quant/dynamic_mx_quant/tests/assets/golden.py).
Native MXFP4 kernels, packed FP4 checkpoints, and real FP4 weight synchronization
are outside this fake-quant path. Existing hardware quantization methods are
unaffected.

Run `python -m pytest -q tests/ut/quantization/test_mxfp8_fake_quant.py` with the
normal project dependencies. Tests cover format dispatch, E2M1 boundaries,
rotation, MoE transpose/cache refresh, fallback, and backward-only targets.
Offline development checked syntax/lint and numerical formulas; torch/NPU,
graph capture, end-to-end rollout, and RL accuracy still require server testing.
