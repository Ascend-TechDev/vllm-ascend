"""Measure rollout A-only full compensation with real NPU GEMM/GMM entry points.

Run in the installed vLLM-Ascend environment on A2, for example:
python benchmarks/benchmark_activation_residual.py --tokens 128 --hidden 1024 --intermediate 1024

This is an isolated operator benchmark: it excludes model loading, routing,
communication, generation, and backward. It reports fake-quant costs and must not
be presented as native W4A8 kernel throughput or end-to-end training overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.ops.fused_moe.moe_mlp import unquant_apply_mlp
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.mxfp8_fake_quant import (
    _maybe_rotate,
    fake_quant_mxfp8_transposed_moe_weight,
    fake_quant_mxfp8_weight,
    initialize_mxfp8_fake_quant_config,
    warmup_mxfp8_rotation_matrix,
)


def configure(enabled: bool, rotation: bool) -> None:
    os.environ.update(
        VLLM_ASCEND_QAT_FAKE_QUANT="1",
        VLLM_ASCEND_QAT_FAKE_QUANT_MODE="w4a8_mxfp4",
        VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND="torch",
        VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE="rint",
        VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE="32",
        VLLM_ASCEND_QAT_MXFP4_SCALE_ALG="standard",
        VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS='["Fprop"]',
        VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE=str(rotation).lower(),
        VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND="block_hadamard_sign",
        VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS='["Fprop"]',
        VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE="32",
        VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED="0",
        VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS="[]",
        VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS="",
        VLLM_ASCEND_QAT_MXFP8_ACTIVATION_RESIDUAL_FPROP=str(enabled).lower(),
    )
    initialize_mxfp8_fake_quant_config()
    warmup_mxfp8_rotation_matrix("npu")


def measure(function, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    memory_before = torch.npu.memory_allocated()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "peak_increment_bytes": torch.npu.max_memory_allocated() - memory_before,
    }


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(actual.float() - expected.float())
        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    ).item()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--rotation", action="store_true")
    args = parser.parse_args()
    if min(args.tokens, args.hidden, args.intermediate, args.experts, args.iterations) <= 0 or args.warmup < 0:
        parser.error("sizes, experts, and iterations must be positive; warmup must be non-negative")
    if args.hidden % 32 or args.intermediate % 32:
        parser.error("hidden and intermediate must be multiples of 32")
    if not torch.npu.is_available():
        raise RuntimeError("This benchmark requires an available Ascend NPU")
    torch.manual_seed(0)
    configure(False, args.rotation)
    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.tokens, args.hidden, dtype=dtype, device="npu") / 4
    w = fake_quant_mxfp8_weight(torch.randn(args.intermediate, args.hidden, dtype=dtype, device="npu") / 8)
    layer = SimpleNamespace(weight=w, _mxfp8_fake_quant_enabled=True)
    dense = lambda: AscendUnquantizedLinearMethod.apply(None, layer, x)
    # Weights are prepared once and are identical in the on/off measurements.
    w1 = fake_quant_mxfp8_transposed_moe_weight(
        torch.randn(args.experts, args.hidden, 2 * args.intermediate, dtype=dtype, device="npu") / 8
    )
    w2 = fake_quant_mxfp8_transposed_moe_weight(
        torch.randn(args.experts, args.intermediate, args.hidden, dtype=dtype, device="npu") / 8
    )
    counts = torch.full((args.experts,), args.tokens // args.experts, dtype=torch.int64, device="npu")
    counts[: args.tokens % args.experts] += 1
    scales = torch.full((args.tokens, 1), 0.5, dtype=dtype, device="npu")
    moe = lambda: unquant_apply_mlp(
        x,
        w1,
        w2,
        counts,
        group_list_type=1,
        topk_scales=scales,
        need_trans=False,
        fake_quant_enabled=True,
    )[0]
    rows = []
    for enabled in (False, True):
        configure(enabled, args.rotation)
        for name, function in (("dense", dense), ("moe", moe)):
            rows.append(
                {
                    "operator": name,
                    "activation_residual_fprop": enabled,
                    **measure(function, args.warmup, args.iterations),
                }
            )
    expected_dense = torch.nn.functional.linear(_maybe_rotate(x), w)
    gate_up = torch_npu.npu_grouped_matmul(
        [_maybe_rotate(x)],
        [w1],
        group_list=counts,
        group_list_type=1,
        split_item=2,
        group_type=0,
    )[0]
    middle = _maybe_rotate(torch_npu.npu_swiglu(gate_up)) * scales
    expected_moe = torch_npu.npu_grouped_matmul(
        [middle],
        [w2],
        group_list=counts,
        group_list_type=1,
        split_item=2,
        group_type=0,
    )[0]
    errors = {"dense": relative_error(dense(), expected_dense), "moe": relative_error(moe(), expected_moe)}
    tolerance = {"bfloat16": 0.02, "float16": 0.005, "float32": 0.0001}[args.dtype]
    for name, error in errors.items():
        if error > tolerance:
            raise AssertionError(f"{name}: relative L2 error {error} exceeds {tolerance}")
    print(
        json.dumps(
            {
                "configuration": vars(args),
                "measurements": rows,
                "compensated_vs_A_times_Wq_relative_l2": errors,
                "scope": "operator fake quant only; no routing, TP, graph capture, or training",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
