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
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.quantization.mxfp8_rotation import (
    MXFP8RotationConfig,
    apply_mxfp8_block_rotation,
    validate_mxfp8_rotation_config,
)

_MXFP8_BLOCK_SIZE = 32
_MXFP8_EMAX = 8
_MXFP8_SCALE_EMAX = 127
_MXFP8_MIN_PRIVATE_EXP = -6
_MXFP8_MANTISSA_SCALE = 8.0
_MXFP8_ROUNDING_MODES = {"rint", "round", "random", "hash"}
_MXFP8_HASH_MULTIPLIER = 1664525
_MXFP8_HASH_INCREMENT = 1013904223
_MXFP8_HASH_MODULUS = 2**32
_MXFP8_HASH_RANDOM_SHIFT = 2**8
_MXFP8_HASH_RANDOM_LEVELS = 2**24
_LOGGED_KEYS: set[tuple[Any, ...]] = set()
_LOG_LOCK = threading.Lock()
_FAKE_QUANT_LAYER_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "mxfp8_fake_quant_layer_override", default=None
)


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


def get_mxfp8_fake_quant_config() -> MXFP8FakeQuantConfig:
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


def is_mxfp8_fake_quant_enabled() -> bool:
    override = _FAKE_QUANT_LAYER_OVERRIDE.get()
    if override is not None:
        return override
    return get_mxfp8_fake_quant_config().enable


@contextmanager
def fake_quant_layer_context(enabled: bool):
    token = _FAKE_QUANT_LAYER_OVERRIDE.set(enabled)
    try:
        yield
    finally:
        _FAKE_QUANT_LAYER_OVERRIDE.reset(token)


def should_fake_quantize_layer(layer_name: str | None) -> bool:
    config = get_mxfp8_fake_quant_config()
    if not config.enable:
        return False
    if not layer_name:
        return True
    return not any(
        re.match(pattern[3:], layer_name) is not None if pattern.startswith("re:") else pattern in layer_name
        for pattern in config.ignore_patterns
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
    rounding_mode = normalize_mxfp8_rounding_mode(rounding_mode)
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
    max_norm = torch.finfo(torch.float8_e4m3fn).max
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

    quant = quantized.reshape(original_shape).to(torch.float8_e4m3fn)
    shared_exp_fixed = torch.nan_to_num(shared_exp, nan=-127.0)
    scale = torch.clamp(shared_exp_fixed + 127.0, 0, 255).round().to(torch.uint8)
    return quant, scale


def _dequantize_mxfp8(tensor_q: torch.Tensor, tensor_scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    tensor_fp32 = tensor_q.to(torch.float32)
    num_blocks = tensor_q.shape[-1] // _MXFP8_BLOCK_SIZE
    blocked = tensor_fp32.reshape(*tensor_q.shape[:-1], num_blocks, _MXFP8_BLOCK_SIZE)
    scale_shape = blocked.shape[:-1]
    descale = torch.exp2(tensor_scale.to(torch.float32).reshape(scale_shape) - 127.0)
    dequant = blocked * descale.unsqueeze(-1)
    return dequant.reshape_as(tensor_q).to(dtype)


def _fake_quantize_mxfp8_tensor(tensor: torch.Tensor, config: MXFP8FakeQuantConfig) -> torch.Tensor:
    original_shape = tensor.shape
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    tensor_q, tensor_scale = _quantize_mxfp8_torch(tensor_2d, config.rounding_mode)
    return _dequantize_mxfp8(tensor_q, tensor_scale, tensor.dtype).reshape(original_shape)


def _maybe_rotate(tensor: torch.Tensor, config: MXFP8FakeQuantConfig) -> torch.Tensor:
    if not config.rotation.enable:
        return tensor
    return apply_mxfp8_block_rotation(tensor, config.rotation)


def fake_quant_mxfp8_weight(tensor: torch.Tensor) -> torch.Tensor:
    config = get_mxfp8_fake_quant_config()
    if not config.enable:
        return tensor
    tensor = _maybe_rotate(tensor, config)
    _log_once(
        ("fake_quant_weight", config.mode, config.rounding_mode, config.rotation.enable),
        "MXFP8 fake quant enabled for rollout weights: mode=%s, rounding_mode=%s, rotation_enable=%s",
        config.mode,
        config.rounding_mode,
        config.rotation.enable,
    )
    return _fake_quantize_mxfp8_tensor(tensor, config)


def fake_quant_mxfp8_activation(tensor: torch.Tensor) -> torch.Tensor:
    config = get_mxfp8_fake_quant_config()
    if not config.enable:
        return tensor
    tensor = _maybe_rotate(tensor, config)
    if not config.quantize_activation:
        return tensor
    _log_once(
        ("fake_quant_activation", config.mode, config.rounding_mode, config.rotation.enable),
        "MXFP8 fake quant enabled for rollout activations: mode=%s, rounding_mode=%s, rotation_enable=%s",
        config.mode,
        config.rounding_mode,
        config.rotation.enable,
    )
    return _fake_quantize_mxfp8_tensor(tensor, config)


def fake_quant_mxfp8_transposed_moe_weight(weight: torch.Tensor) -> torch.Tensor:
    if not is_mxfp8_fake_quant_enabled():
        return weight
    return fake_quant_mxfp8_weight(weight.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


def fake_quant_mxfp8_transposed_moe_weight_list(weights: Any) -> Any:
    if not is_mxfp8_fake_quant_enabled() or not isinstance(weights, list):
        return weights
    return [fake_quant_mxfp8_transposed_moe_weight(weight) for weight in weights]


__all__ = [
    "MXFP8FakeQuantConfig",
    "fake_quant_mxfp8_activation",
    "fake_quant_mxfp8_transposed_moe_weight",
    "fake_quant_mxfp8_transposed_moe_weight_list",
    "fake_quant_mxfp8_weight",
    "fake_quant_layer_context",
    "get_mxfp8_fake_quant_config",
    "is_mxfp8_fake_quant_enabled",
    "normalize_mxfp8_fake_quant_mode",
    "normalize_mxfp8_rounding_mode",
    "should_fake_quantize_layer",
]
