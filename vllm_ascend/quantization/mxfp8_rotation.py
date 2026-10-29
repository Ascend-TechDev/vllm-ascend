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
"""Block-wise MXFP8 activation rotation helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

MXFP8_ROTATION_ENABLE_KEY = "mxfp8_rotation_enable"
MXFP8_ROTATION_KIND_KEY = "mxfp8_rotation_kind"
MXFP8_ROTATION_BLOCK_SIZE_KEY = "mxfp8_rotation_block_size"
MXFP8_ROTATION_SEED_KEY = "mxfp8_rotation_seed"

MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN = "block_hadamard_sign"
_MXFP8_ROTATION_KIND_ALIASES = {
    MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN: MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
    "hadamard_sign": MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
    "block_hadamard": MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN,
}
_DEFAULT_BLOCK_SIZE = 32
_DEFAULT_SEED = 0
_MAX_TORCH_SEED = 2**63 - 1
_ROTATION_MATRIX_CACHE: dict[tuple[str, int, int, str, str], torch.Tensor] = {}


@dataclass(frozen=True)
class MXFP8RotationConfig:
    enable: bool = False
    kind: str = MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN
    block_size: int = _DEFAULT_BLOCK_SIZE
    seed: int = _DEFAULT_SEED


def normalize_mxfp8_rotation_kind(kind: str) -> str:
    normalized = str(kind).lower()
    if normalized not in _MXFP8_ROTATION_KIND_ALIASES:
        raise ValueError(
            f"Unsupported MXFP8 rotation kind: {kind}. "
            f"Supported kinds: {sorted(_MXFP8_ROTATION_KIND_ALIASES)}"
        )
    return _MXFP8_ROTATION_KIND_ALIASES[normalized]


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_mxfp8_rotation_config(config: Mapping[str, Any] | None) -> MXFP8RotationConfig:
    if config is None:
        return MXFP8RotationConfig()

    enable = _coerce_bool(config.get(MXFP8_ROTATION_ENABLE_KEY, False))
    kind = normalize_mxfp8_rotation_kind(
        config.get(MXFP8_ROTATION_KIND_KEY, MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN)
    )
    return MXFP8RotationConfig(
        enable=enable,
        kind=kind,
        block_size=int(config.get(MXFP8_ROTATION_BLOCK_SIZE_KEY, _DEFAULT_BLOCK_SIZE)),
        seed=int(config.get(MXFP8_ROTATION_SEED_KEY, _DEFAULT_SEED)),
    )


def validate_mxfp8_rotation_config(config: MXFP8RotationConfig, group_size: int = _DEFAULT_BLOCK_SIZE):
    if not config.enable:
        return
    normalize_mxfp8_rotation_kind(config.kind)
    if config.block_size != group_size:
        raise ValueError(
            f"MXFP8 rotation block_size must match group_size={group_size}, got: {config.block_size}"
        )
    if config.block_size <= 0 or config.block_size & (config.block_size - 1):
        raise ValueError(f"MXFP8 Hadamard rotation requires a power-of-two block_size, got: {config.block_size}")


def _normalize_seed(seed: int) -> int:
    return int(seed) % _MAX_TORCH_SEED


def _build_hadamard_matrix(size: int) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a positive power of two, got: {size}")

    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix * (1.0 / math.sqrt(size))


def _build_block_rotation_matrix(config: MXFP8RotationConfig) -> torch.Tensor:
    if config.kind != MXFP8_ROTATION_KIND_BLOCK_HADAMARD_SIGN:
        raise ValueError(f"Unsupported MXFP8 rotation kind: {config.kind}")

    matrix = _build_hadamard_matrix(config.block_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_normalize_seed(config.seed))
    signs = torch.randint(0, 2, (config.block_size,), generator=generator, dtype=torch.int8)
    signs = signs.to(torch.float32).mul_(2.0).sub_(1.0)
    return matrix * signs


def _get_block_rotation_matrix(config: MXFP8RotationConfig, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cache_key = (
        normalize_mxfp8_rotation_kind(config.kind),
        int(config.block_size),
        _normalize_seed(config.seed),
        str(device),
        str(dtype),
    )
    cached = _ROTATION_MATRIX_CACHE.get(cache_key)
    if cached is None:
        cached = _build_block_rotation_matrix(config).to(device=device, dtype=dtype)
        _ROTATION_MATRIX_CACHE[cache_key] = cached
    return cached


def _rotation_work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def apply_mxfp8_block_rotation(
    tensor: torch.Tensor,
    config: MXFP8RotationConfig | Mapping[str, Any],
    *,
    transpose: bool = False,
) -> torch.Tensor:
    if isinstance(config, Mapping):
        config = get_mxfp8_rotation_config(config)
    if not config.enable:
        return tensor

    validate_mxfp8_rotation_config(config)
    last_dim = tensor.shape[-1]
    if last_dim % config.block_size != 0:
        raise ValueError(
            f"MXFP8 block rotation requires last_dim divisible by block_size={config.block_size}, "
            f"got shape={tuple(tensor.shape)}"
        )

    fp8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dtype is not None
    )
    if tensor.dtype in fp8_dtypes:
        raise ValueError("MXFP8 block rotation must run before FP8 quantization")

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    work_dtype = _rotation_work_dtype(original_dtype)
    matrix = _get_block_rotation_matrix(config, device=tensor.device, dtype=work_dtype)
    if transpose:
        matrix = matrix.t()

    tensor_2d = tensor.reshape(-1, last_dim).to(work_dtype)
    tensor_blocked = tensor_2d.reshape(tensor_2d.shape[0], last_dim // config.block_size, config.block_size)
    rotated = torch.matmul(tensor_blocked, matrix)
    return rotated.reshape(tensor_2d.shape).to(original_dtype).reshape(original_shape)


def is_mxfp8_rotation_enabled(config: MXFP8RotationConfig | Mapping[str, Any] | None) -> bool:
    if config is None:
        return False
    if isinstance(config, Mapping):
        config = get_mxfp8_rotation_config(config)
    return bool(config.enable)
