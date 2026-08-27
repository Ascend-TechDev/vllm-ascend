#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""MXFP8 fake quantization for high-precision rollout paths.

This mirrors verl's torch MXFP8 QAT math, but returns dequantized high-precision
tensors so the regular BF16/FP16 kernels continue to execute on Ascend A2.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.quantization.mxfp8_rotation import (
    MXFP8RotationConfig,
    _build_block_rotation_matrix,
    validate_mxfp8_rotation_config,
)

_MXFP8_BLOCK_SIZE = 32
_MXFP8_EMAX = 8
_MXFP8_SCALE_EMAX = 127
_MXFP8_MIN_PRIVATE_EXP = -6
_MXFP8_MANTISSA_SCALE = 8.0
_MXFP8_MAX_NORM = 448.0  # torch.float8_e4m3fn max finite value
_MXFP8_ROUNDING_MODES = {"rint", "round", "random", "hash"}
_MXFP8_HASH_MULTIPLIER = 1664525
_MXFP8_HASH_INCREMENT = 1013904223
_MXFP8_HASH_MODULUS = 2**32
_MXFP8_HASH_RANDOM_SHIFT = 2**8
_MXFP8_HASH_RANDOM_LEVELS = 2**24
_LOGGED_KEYS: set[tuple[Any, ...]] = set()
_LOG_LOCK = threading.Lock()

_RUNTIME_FAKE_QUANT_ENABLED = False
_RUNTIME_MODE = "w8a8_mxfp8"
_RUNTIME_QUANT_BACKEND = "torch"
_RUNTIME_ROUNDING_MODE = "rint"
_RUNTIME_GROUP_SIZE = _MXFP8_BLOCK_SIZE
_RUNTIME_ROTATION_ENABLED = False
_RUNTIME_ROTATION = MXFP8RotationConfig()
_RUNTIME_ROTATION_MATRIX: torch.Tensor | None = None
_RUNTIME_IGNORE_PATTERNS: tuple[str, ...] = ()
_WEIGHT_FAKE_QUANT_CACHE_EPOCH = 0


@dataclass(frozen=True)
class MXFP8FakeQuantConfig:
    enable: bool
    mode: str
    quant_backend: str
    rounding_mode: str
    group_size: int
    rotation: MXFP8RotationConfig
    ignore_patterns: tuple[str, ...]

    @property
    def quantize_activation(self) -> bool:
        return self.mode == "w8a8_mxfp8"


def _log_once(key: tuple[Any, ...], message: str, *args):
    with _LOG_LOCK:
        if key in _LOGGED_KEYS:
            return
        _LOGGED_KEYS.add(key)
    logger.warning(message, *args)


def normalize_mxfp8_rounding_mode(rounding_mode: str) -> str:
    rounding_mode = str(rounding_mode).lower()
    if rounding_mode not in _MXFP8_ROUNDING_MODES:
        raise ValueError(
            f"Unsupported MXFP8 rounding mode: {rounding_mode}. Supported modes: {sorted(_MXFP8_ROUNDING_MODES)}"
        )
    return rounding_mode


def normalize_mxfp8_fake_quant_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in {"w8a16_mxfp8", "w8a8_mxfp8"}:
        raise ValueError(f"Unsupported MXFP8 fake quant mode: {mode}")
    return mode


def _read_mxfp8_fake_quant_config() -> MXFP8FakeQuantConfig:
    mode = normalize_mxfp8_fake_quant_mode(envs.VLLM_ASCEND_QAT_FAKE_QUANT_MODE)
    rounding_mode = normalize_mxfp8_rounding_mode(envs.VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE)
    quant_backend = str(envs.VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND).lower()
    if quant_backend not in {"torch", "npu"}:
        raise ValueError(f"Unsupported MXFP8 fake quant backend: {quant_backend}")

    group_size = int(envs.VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE)
    rotation = MXFP8RotationConfig(
        enable=bool(envs.VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE),
        kind=str(envs.VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND),
        block_size=int(envs.VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE),
        seed=int(envs.VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED),
    )
    validate_mxfp8_rotation_config(rotation, group_size=group_size)
    if group_size != _MXFP8_BLOCK_SIZE:
        raise ValueError(f"MXFP8 fake quant requires group_size={_MXFP8_BLOCK_SIZE}, got: {group_size}")
    if quant_backend != "torch":
        _log_once(
            ("fake_quant_backend", quant_backend),
            "MXFP8 fake quant uses the torch formula; received quant_backend=%s from config",
            quant_backend,
        )
    try:
        ignore_patterns = tuple(json.loads(envs.VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS))
    except (TypeError, ValueError) as exc:
        raise ValueError("VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS must be a JSON list") from exc
    if not all(isinstance(pattern, str) for pattern in ignore_patterns):
        raise ValueError("VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS must contain only strings")
    return MXFP8FakeQuantConfig(
        enable=bool(envs.VLLM_ASCEND_QAT_FAKE_QUANT),
        mode=mode,
        quant_backend=quant_backend,
        rounding_mode=rounding_mode,
        group_size=group_size,
        rotation=rotation,
        ignore_patterns=ignore_patterns,
    )


def initialize_mxfp8_fake_quant_config() -> MXFP8FakeQuantConfig:
    """Parse rollout fake-quant settings before torch.compile captures model code."""
    global _RUNTIME_FAKE_QUANT_ENABLED
    global _RUNTIME_MODE
    global _RUNTIME_QUANT_BACKEND
    global _RUNTIME_ROUNDING_MODE
    global _RUNTIME_GROUP_SIZE
    global _RUNTIME_ROTATION_ENABLED
    global _RUNTIME_ROTATION
    global _RUNTIME_ROTATION_MATRIX
    global _RUNTIME_IGNORE_PATTERNS

    config = _read_mxfp8_fake_quant_config()
    _RUNTIME_FAKE_QUANT_ENABLED = config.enable
    _RUNTIME_MODE = config.mode
    _RUNTIME_QUANT_BACKEND = config.quant_backend
    _RUNTIME_ROUNDING_MODE = config.rounding_mode
    _RUNTIME_GROUP_SIZE = config.group_size
    _RUNTIME_ROTATION_ENABLED = config.rotation.enable
    _RUNTIME_ROTATION = config.rotation
    _RUNTIME_IGNORE_PATTERNS = config.ignore_patterns
    _RUNTIME_ROTATION_MATRIX = (
        _build_block_rotation_matrix(config.rotation) if config.rotation.enable else None
    )
    logger.warning(
        "MXFP8 rollout fake quant initialized: enable=%s, mode=%s, quant_backend=%s, rounding_mode=%s, "
        "group_size=%s, rotation_enable=%s, rotation_kind=%s, rotation_block_size=%s, rotation_seed=%s",
        config.enable,
        config.mode,
        config.quant_backend,
        config.rounding_mode,
        config.group_size,
        config.rotation.enable,
        config.rotation.kind,
        config.rotation.block_size,
        config.rotation.seed,
    )
    return config


def get_mxfp8_fake_quant_config() -> MXFP8FakeQuantConfig:
    return MXFP8FakeQuantConfig(
        enable=_RUNTIME_FAKE_QUANT_ENABLED,
        mode=_RUNTIME_MODE,
        quant_backend=_RUNTIME_QUANT_BACKEND,
        rounding_mode=_RUNTIME_ROUNDING_MODE,
        group_size=_RUNTIME_GROUP_SIZE,
        rotation=_RUNTIME_ROTATION,
        ignore_patterns=_RUNTIME_IGNORE_PATTERNS,
    )


def is_mxfp8_fake_quant_enabled() -> bool:
    return _RUNTIME_FAKE_QUANT_ENABLED


def invalidate_mxfp8_weight_fake_quant_cache(reason: str | None = None):
    """Invalidate cached in-place fake-quantized weights after a weight update."""
    global _WEIGHT_FAKE_QUANT_CACHE_EPOCH
    _WEIGHT_FAKE_QUANT_CACHE_EPOCH += 1
    logger.info(
        "MXFP8 rollout fake quant weight cache invalidated: epoch=%s%s",
        _WEIGHT_FAKE_QUANT_CACHE_EPOCH,
        f", reason={reason}" if reason else "",
    )


def should_fake_quantize_layer(layer_name: str | None) -> bool:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return False
    if not layer_name:
        return True
    return not any(
        re.match(pattern[3:], layer_name) is not None if pattern.startswith("re:") else pattern in layer_name
        for pattern in _RUNTIME_IGNORE_PATTERNS
    )


def _check_mxfp8_2d_tensor(tensor: torch.Tensor):
    if tensor.dim() != 2:
        raise ValueError(f"MXFP8 quantization only supports 2D tensors, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] % _MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 quantization requires the last dimension to be divisible by {_MXFP8_BLOCK_SIZE}, "
            f"got shape={tuple(tensor.shape)}"
        )


def _round_mxfp8_scaled_abs(abs_scaled: torch.Tensor, rounding_mode: str) -> torch.Tensor:
    if rounding_mode == "rint":
        return torch.round(abs_scaled)
    if rounding_mode == "round":
        return torch.floor(abs_scaled + 0.5)

    floor_val = torch.floor(abs_scaled)
    frac = abs_scaled - floor_val
    if rounding_mode == "random":
        rand = torch.rand_like(frac)
    elif rounding_mode == "hash":
        int_bits = abs_scaled.contiguous().view(torch.int32).detach().to(torch.int64)
        uint_bits = torch.remainder(int_bits, _MXFP8_HASH_MODULUS)
        hashed = torch.remainder(
            uint_bits * _MXFP8_HASH_MULTIPLIER + _MXFP8_HASH_INCREMENT,
            _MXFP8_HASH_MODULUS,
        )
        rand_bits = torch.div(hashed, _MXFP8_HASH_RANDOM_SHIFT, rounding_mode="floor")
        rand = rand_bits.to(torch.float32) * (1.0 / float(_MXFP8_HASH_RANDOM_LEVELS))
    else:
        raise ValueError(f"Unsupported MXFP8 rounding mode: {rounding_mode}")

    return floor_val + (rand < frac).to(floor_val.dtype)


def _quantize_mxfp8_torch(tensor: torch.Tensor, rounding_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    _check_mxfp8_2d_tensor(tensor)
    tensor_fp32 = tensor.to(torch.float32)
    original_shape = tensor_fp32.shape
    max_norm = _MXFP8_MAX_NORM
    num_blocks = tensor.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = tensor_fp32.reshape(tensor.shape[0], num_blocks, _MXFP8_BLOCK_SIZE)

    amax = blocked.abs().amax(dim=-1)
    amax_safe = torch.where(amax == 0, torch.full_like(amax, torch.finfo(torch.float32).tiny), amax)
    shared_exp = torch.floor(torch.log2(amax_safe)) - _MXFP8_EMAX
    shared_exp = torch.where(shared_exp > _MXFP8_SCALE_EMAX, torch.full_like(shared_exp, float("nan")), shared_exp)

    scale_factor = torch.pow(2.0, shared_exp.unsqueeze(-1))
    normalized = blocked / scale_factor
    abs_norm = normalized.abs()
    private_exp = torch.floor(torch.log2(abs_norm + (abs_norm == 0).float()))
    private_exp = private_exp.clamp(min=_MXFP8_MIN_PRIVATE_EXP)

    private_scale = torch.pow(2.0, private_exp)
    scaled = normalized / private_scale * _MXFP8_MANTISSA_SCALE
    quantized = torch.sign(scaled) * _round_mxfp8_scaled_abs(torch.abs(scaled), rounding_mode)
    quantized = quantized / _MXFP8_MANTISSA_SCALE * private_scale
    quantized = torch.clamp(quantized, min=-max_norm, max=max_norm)
    quantized = torch.where(torch.isinf(normalized), normalized, quantized)
    quantized = torch.where(torch.isnan(normalized), normalized, quantized)

    # This is a fake-quant/dequant path. The quantized payload is consumed
    # immediately by _dequantize_mxfp8, so do not materialize an FP8 tensor on
    # Ascend A2, where the float8_e4m3fn device cast is unsupported.
    quant = quantized.reshape(original_shape)
    shared_exp_fixed = torch.nan_to_num(shared_exp, nan=-127.0)
    # Keep the scale in FP32 as well. The fake path immediately dequantizes it,
    # and A2 does not need the real MXFP8 uint8/FP8 storage representation.
    scale = torch.clamp(shared_exp_fixed + 127.0, 0, 255).round()
    return quant, scale


def _dequantize_mxfp8(tensor_q: torch.Tensor, tensor_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    tensor_fp32 = tensor_q.to(torch.float32)
    num_blocks = tensor_q.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = tensor_fp32.reshape(*tensor_q.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    scale_shape = blocked.shape[:-1]
    descale = torch.exp2(tensor_scale.to(torch.float32).reshape(scale_shape) - 127.0)
    dequant = blocked * descale.unsqueeze(-1)
    return dequant.reshape_as(tensor_q).to(dtype)


def _fake_quantize_mxfp8_tensor(tensor: torch.Tensor, rounding_mode: str) -> torch.Tensor:
    original_shape = tensor.shape
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    tensor_q, tensor_scale = _quantize_mxfp8_torch(tensor_2d, rounding_mode)
    return _dequantize_mxfp8(tensor_q, tensor_scale, tensor.dtype).reshape(original_shape)


def _maybe_rotate(tensor: torch.Tensor) -> torch.Tensor:
    if not _RUNTIME_ROTATION_ENABLED:
        return tensor

    last_dim = tensor.shape[-1]
    work_dtype = torch.float32 if tensor.dtype in (torch.float16, torch.bfloat16) else tensor.dtype
    matrix = _RUNTIME_ROTATION_MATRIX.to(device=tensor.device, dtype=work_dtype)
    tensor_2d = tensor.reshape(-1, last_dim).to(work_dtype)
    tensor_blocked = tensor_2d.reshape(tensor_2d.shape[0], last_dim // _RUNTIME_GROUP_SIZE, _RUNTIME_GROUP_SIZE)
    rotated = torch.matmul(tensor_blocked, matrix)
    return rotated.reshape(tensor_2d.shape).to(tensor.dtype).reshape_as(tensor)


def fake_quant_mxfp8_weight(tensor: torch.Tensor) -> torch.Tensor:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return tensor
    tensor = _maybe_rotate(tensor)
    return _fake_quantize_mxfp8_tensor(tensor, _RUNTIME_ROUNDING_MODE)


def fake_quant_mxfp8_activation(tensor: torch.Tensor) -> torch.Tensor:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return tensor
    tensor = _maybe_rotate(tensor)
    if _RUNTIME_MODE != "w8a8_mxfp8":
        return tensor
    return _fake_quantize_mxfp8_tensor(tensor, _RUNTIME_ROUNDING_MODE)


def fake_quant_mxfp8_transposed_moe_weight(weight: torch.Tensor) -> torch.Tensor:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return weight
    return fake_quant_mxfp8_weight(weight.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


def fake_quant_mxfp8_transposed_moe_weight_list(weights: Any) -> Any:
    if not _RUNTIME_FAKE_QUANT_ENABLED or not isinstance(weights, list):
        return weights
    return [fake_quant_mxfp8_transposed_moe_weight(weight) for weight in weights]


def _tensor_cache_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        int(tensor.data_ptr()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        getattr(tensor, "_version", None),
    )


def _cache_signature(tensors: tuple[torch.Tensor, ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(_tensor_cache_signature(tensor) for tensor in tensors)


def _is_weight_cache_valid(layer: Any, cache_attr: str, tensors: tuple[torch.Tensor, ...]) -> bool:
    cache_state = getattr(layer, cache_attr, None)
    if cache_state is None:
        return False
    cache_epoch, cache_signature = cache_state
    return cache_epoch == _WEIGHT_FAKE_QUANT_CACHE_EPOCH and cache_signature == _cache_signature(tensors)


def _mark_weight_cache_valid(layer: Any, cache_attr: str, tensors: tuple[torch.Tensor, ...]):
    setattr(layer, cache_attr, (_WEIGHT_FAKE_QUANT_CACHE_EPOCH, _cache_signature(tensors)))


def fake_quant_mxfp8_weight_inplace(tensor: torch.Tensor) -> torch.Tensor:
    """Fake-quant/dequant a weight once and overwrite it with the dequantized value."""
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return tensor
    with torch.no_grad():
        tensor.copy_(fake_quant_mxfp8_weight(tensor))
    return tensor


def fake_quant_mxfp8_transposed_moe_weight_inplace(weight: torch.Tensor) -> torch.Tensor:
    """In-place MoE weight fake quant with verl's canonical [E, N, K] layout."""
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return weight
    with torch.no_grad():
        weight.copy_(fake_quant_mxfp8_transposed_moe_weight(weight))
    return weight


def ensure_mxfp8_linear_weight_fake_quantized(layer: Any) -> bool:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return False
    weight = getattr(layer, "weight", None)
    if weight is None:
        return False

    tensors = (weight,)
    cache_attr = "_mxfp8_fake_quant_weight_cache"
    if _is_weight_cache_valid(layer, cache_attr, tensors):
        return False

    fake_quant_mxfp8_weight_inplace(weight)
    _mark_weight_cache_valid(layer, cache_attr, tensors)
    return True


def _iter_tensor_list(value: Any) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(tensor for tensor in value if isinstance(tensor, torch.Tensor))


def ensure_mxfp8_moe_weight_fake_quantized(layer: Any) -> bool:
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return False

    w13_weight = getattr(layer, "w13_weight", None)
    w2_weight = getattr(layer, "w2_weight", None)
    w13_weight_list = _iter_tensor_list(getattr(layer, "w13_weight_list", None))
    w2_weight_list = _iter_tensor_list(getattr(layer, "w2_weight_list", None))
    tensors = tuple(
        tensor
        for tensor in (
            w13_weight,
            w2_weight,
            *w13_weight_list,
            *w2_weight_list,
        )
        if isinstance(tensor, torch.Tensor)
    )
    if not tensors:
        return False

    cache_attr = "_mxfp8_fake_quant_moe_weight_cache"
    if _is_weight_cache_valid(layer, cache_attr, tensors):
        return False

    for tensor in tensors:
        fake_quant_mxfp8_transposed_moe_weight_inplace(tensor)
    _mark_weight_cache_valid(layer, cache_attr, tensors)
    return True


def apply_mxfp8_weight_fake_quant_cache(model: torch.nn.Module) -> int:
    """Refresh all layer-local fake-quantized weight caches in a model."""
    if not _RUNTIME_FAKE_QUANT_ENABLED:
        return 0

    cached_layers = 0
    for module in model.modules():
        if not getattr(module, "_mxfp8_fake_quant_enabled", False):
            continue
        if any(hasattr(module, attr) for attr in ("w13_weight", "w2_weight", "w13_weight_list", "w2_weight_list")):
            cached_layers += int(ensure_mxfp8_moe_weight_fake_quantized(module))
        elif hasattr(module, "weight"):
            cached_layers += int(ensure_mxfp8_linear_weight_fake_quantized(module))

    if cached_layers:
        logger.warning(
            "MXFP8 rollout fake quant weight cache refreshed: layers=%s, epoch=%s",
            cached_layers,
            _WEIGHT_FAKE_QUANT_CACHE_EPOCH,
        )
    return cached_layers


__all__ = [
    "MXFP8FakeQuantConfig",
    "apply_mxfp8_weight_fake_quant_cache",
    "ensure_mxfp8_linear_weight_fake_quantized",
    "ensure_mxfp8_moe_weight_fake_quantized",
    "fake_quant_mxfp8_activation",
    "fake_quant_mxfp8_transposed_moe_weight",
    "fake_quant_mxfp8_transposed_moe_weight_list",
    "fake_quant_mxfp8_weight",
    "fake_quant_mxfp8_weight_inplace",
    "get_mxfp8_fake_quant_config",
    "initialize_mxfp8_fake_quant_config",
    "invalidate_mxfp8_weight_fake_quant_cache",
    "is_mxfp8_fake_quant_enabled",
    "normalize_mxfp8_fake_quant_mode",
    "normalize_mxfp8_rounding_mode",
    "should_fake_quantize_layer",
]
