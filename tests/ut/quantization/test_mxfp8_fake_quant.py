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
                "_RUNTIME_QUANT_BACKEND",
                "_RUNTIME_ROUNDING_MODE",
                "_RUNTIME_GROUP_SIZE",
                "_RUNTIME_ROTATION_ENABLED",
                "_RUNTIME_ROTATION",
                "_RUNTIME_ROTATION_MATRIX",
                "_RUNTIME_IGNORE_PATTERNS",
                "_RUNTIME_ROTATION_DEVICE_MATRICES",
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
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
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
