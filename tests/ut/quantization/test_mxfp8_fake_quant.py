import os
import unittest
from unittest.mock import patch

import torch

import vllm_ascend.quantization.mxfp8_fake_quant as fake_quant


class TestMXFP8FakeQuantRotation(unittest.TestCase):
    def setUp(self):
        self._saved_state = {
            name: getattr(fake_quant, name)
            for name in (
                "_RUNTIME_FAKE_QUANT_ENABLED",
                "_RUNTIME_MODE",
                "_RUNTIME_FAKE_QUANT_TARGETS",
                "_RUNTIME_QUANT_BACKEND",
                "_RUNTIME_ROUNDING_MODE",
                "_RUNTIME_GROUP_SIZE",
                "_RUNTIME_ROTATION_ENABLED",
                "_RUNTIME_ROTATION",
                "_RUNTIME_ROTATION_MATRIX",
                "_RUNTIME_IGNORE_PATTERNS",
                "_RUNTIME_FALLBACK_LAYERS",
                "_RUNTIME_ROTATION_DEVICE_MATRICES",
                "_WEIGHT_FAKE_QUANT_CACHE_EPOCH",
                "_WEIGHT_FAKE_QUANT_REFRESHED_LAYERS",
                "_WEIGHT_FAKE_QUANT_REFRESHED_LINEAR_LAYERS",
                "_WEIGHT_FAKE_QUANT_REFRESHED_MOE_LAYERS",
            )
        }

    def tearDown(self):
        for name, value in self._saved_state.items():
            setattr(fake_quant, name, value)

    def test_rotation_matrix_is_cached_before_npu_forward_and_reused(self):
        env = {
            "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
            "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": "w8a8_mxfp8",
            "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": "torch",
            "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": "rint",
            "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": "true",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND": "block_hadamard_sign",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": "7",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
            "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": "",
        }
        with patch.dict(os.environ, env, clear=False):
            fake_quant.initialize_mxfp8_fake_quant_config()
            tensor = torch.randn(2, 32)

            with self.assertRaisesRegex(RuntimeError, "not warmed up"):
                fake_quant._get_runtime_rotation_matrix(torch.device("cuda:0"), torch.float32)

            matrix = fake_quant.warmup_mxfp8_rotation_matrix("cpu")
            self.assertIs(matrix, fake_quant.warmup_mxfp8_rotation_matrix("cpu"))
            self.assertEqual(matrix.device.type, "cpu")

            rotated = fake_quant._maybe_rotate(tensor)
            expected = torch.matmul(tensor.view(2, 1, 32), matrix).view(2, 32)
            self.assertTrue(torch.allclose(rotated, expected))

    def test_backward_only_targets_do_not_rotate_rollout_activation(self):
        env = {
            "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
            "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": "w8a8_mxfp8",
            "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": "torch",
            "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": "rint",
            "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": "true",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND": "block_hadamard_sign",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": "7",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": '["Dgrad", "Wgrad"]',
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
            "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": "",
        }
        with patch.dict(os.environ, env, clear=False):
            fake_quant.initialize_mxfp8_fake_quant_config()
            tensor = torch.randn(2, 32)

            self.assertIs(fake_quant._maybe_rotate(tensor), tensor)
            self.assertIsNone(fake_quant.warmup_mxfp8_rotation_matrix("cpu"))

    def test_fallback_layers_skip_fake_quant_for_linear_and_moe_prefixes(self):
        env = {
            "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
            "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": "w8a8_mxfp8",
            "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": "torch",
            "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": "rint",
            "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": "false",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND": "block_hadamard_sign",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": "7",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
            "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": "0-3,37-48",
        }
        with patch.dict(os.environ, env, clear=False):
            config = fake_quant.initialize_mxfp8_fake_quant_config()

            self.assertEqual(config.fallback_layers, ((0, 3), (37, 48)))
            self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.0.self_attn.q_proj"))
            self.assertTrue(fake_quant.should_fake_quantize_layer("model.layers.4.self_attn.q_proj"))
            self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.37.mlp.experts"))
            self.assertFalse(fake_quant.should_fake_quantize_layer("model.layer48.mlp.down_proj"))
            self.assertTrue(fake_quant.should_fake_quantize_layer("model.layers.49.mlp.down_proj"))

    def _configure_mxfp4(self, mode, **overrides):
        env = {
            "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
            "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": mode,
            "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": "torch",
            "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": "hash",
            "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": "false",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_KIND": "block_hadamard_sign",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_BLOCK_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": "7",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
            "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": "0-1",
        }
        env.update(overrides)
        with patch.dict(os.environ, env, clear=False):
            return fake_quant.initialize_mxfp8_fake_quant_config()

    def test_mxfp4_rollout_selects_weight_and_activation_formats(self):
        tensor = torch.linspace(-1.7, 2.1, 64).reshape(2, 32)
        for mode in ("w4a8_mxfp4", "w4a4_mxfp4"):
            with self.subTest(mode=mode):
                config = self._configure_mxfp4(mode)
                self.assertTrue(config.quantize_activation)
                expected_weight = fake_quant._fake_quantize_mxfp4_tensor(tensor, "hash")
                torch.testing.assert_close(fake_quant.fake_quant_mxfp8_weight(tensor), expected_weight)
                activation_quantizer = (
                    fake_quant._fake_quantize_mxfp4_tensor
                    if mode == "w4a4_mxfp4"
                    else fake_quant._fake_quantize_mxfp8_tensor
                )
                expected_input = activation_quantizer(tensor, "rint")
                torch.testing.assert_close(fake_quant.fake_quant_mxfp8_activation(tensor), expected_input)
                self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.1.mlp.experts"))
                self.assertTrue(fake_quant.should_fake_quantize_layer("model.layers.2.mlp.experts"))

    def test_mxfp4_rotation_and_transposed_moe_weight_cache(self):
        from types import SimpleNamespace

        for mode in ("w4a8_mxfp4", "w4a4_mxfp4"):
            with self.subTest(mode=mode):
                self._configure_mxfp4(mode, VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE="true")
                layer = SimpleNamespace(w13_weight=torch.randn(2, 32, 64), w2_weight=torch.randn(2, 32, 32))
                original = layer.w13_weight.clone()
                original_down = layer.w2_weight.clone()
                canonical = fake_quant._maybe_rotate(original.transpose(-1, -2).contiguous())
                expected = fake_quant._fake_quantize_mxfp4_tensor(canonical, "hash").transpose(-1, -2)
                self.assertTrue(fake_quant.ensure_mxfp8_moe_weight_fake_quantized(layer))
                torch.testing.assert_close(layer.w13_weight, expected)
                cached = layer.w13_weight.clone()
                self.assertFalse(fake_quant.ensure_mxfp8_moe_weight_fake_quantized(layer))
                self.assertTrue(torch.equal(layer.w13_weight, cached))
                # A new actor weight load supplies high-precision weights before invalidation.
                layer.w13_weight.copy_(original)
                layer.w2_weight.copy_(original_down)
                fake_quant.invalidate_mxfp8_weight_fake_quant_cache("unit test weight sync")
                self.assertTrue(fake_quant.ensure_mxfp8_moe_weight_fake_quantized(layer))
                torch.testing.assert_close(layer.w13_weight, expected)

    def test_mxfp4_backward_only_fake_targets_keep_rollout_high_precision(self):
        self._configure_mxfp4("w4a4_mxfp4", VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS='["Dgrad", "Wgrad"]')
        tensor = torch.randn(2, 32)
        self.assertIs(fake_quant.fake_quant_mxfp8_weight(tensor), tensor)
        self.assertIs(fake_quant.fake_quant_mxfp8_activation(tensor), tensor)

    def test_mxfp4_midpoint_and_zero_block_scales(self):
        tensor = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7] * 4).reshape(1, 32)
        payload, scale = fake_quant.quantize_mxfp4_tensor(tensor)
        self.assertTrue(torch.equal(payload, torch.tensor([0, 1, 1, 2, 2, 4, 4, 6] * 4).reshape(1, 32)))
        self.assertTrue(torch.equal(scale, torch.tensor([[127.0]])))
        payload, scale = fake_quant.quantize_mxfp4_tensor(torch.zeros(2, 96))
        self.assertTrue(torch.equal(scale, torch.zeros(2, 3)))
        self.assertTrue(torch.equal(fake_quant.dequantize_mxfp4_tensor(payload, scale, torch.float32), payload))

    def test_mxfp4_rejects_native_backend_and_invalid_shapes(self):
        with self.assertRaisesRegex(ValueError, "quant_backend='torch'"):
            self._configure_mxfp4("w4a4_mxfp4", VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND="npu")
        with self.assertRaises(ValueError):
            fake_quant.quantize_mxfp4_tensor(torch.zeros(2, 31))

    def test_mxfp4_handles_empty_token_rows(self):
        tensor = torch.empty(0, 96)
        payload, scale = fake_quant.quantize_mxfp4_tensor(tensor)
        self.assertEqual(tuple(scale.shape), (0, 3))
        self.assertEqual(fake_quant.dequantize_mxfp4_tensor(payload, scale, tensor.dtype).shape, tensor.shape)
