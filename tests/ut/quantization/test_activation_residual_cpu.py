"""CPU numerical checks for the actual rollout compensation function bodies.

The quantizer and rotation modules are imported normally. Dense/MoE/TP entry
points otherwise import the full vLLM/NPU stack, so these CPU-only tests compile
their actual AST nodes with explicit hardware/communication stubs. This does not
validate their full-module import wiring, NPU kernels, graph capture, or HCCL.
"""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from vllm_ascend.quantization import mxfp8_fake_quant as fake_quant

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_function(relative_path, name, namespace, class_name=None):
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def cpu_grouped_matmul(*, x, weight, bias=None, group_list, group_list_type, **kwargs):
    """Hardware-boundary substitute, including empty experts and both list forms."""
    counts = group_list if group_list_type == 1 else torch.diff(group_list, prepend=group_list.new_zeros(1))
    outputs = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        output = x[0][start : start + count] @ weight[0][expert]
        if bias is not None:
            output = output + bias[0][expert]
        outputs.append(output)
        start += count
    return [torch.cat(outputs, dim=0)]


class TestActivationResidualCPU(unittest.TestCase):
    def setUp(self):
        self.state = {name: value for name, value in vars(fake_quant).items() if name.startswith("_RUNTIME_")}
        torch.manual_seed(71)

    def tearDown(self):
        for name, value in self.state.items():
            setattr(fake_quant, name, value)

    def configure(self, enabled=True, **overrides):
        env = {
            "VLLM_ASCEND_QAT_FAKE_QUANT": "1",
            "VLLM_ASCEND_QAT_FAKE_QUANT_MODE": "w4a8_mxfp4",
            "VLLM_ASCEND_QAT_MXFP8_QUANT_BACKEND": "torch",
            "VLLM_ASCEND_QAT_MXFP8_ROUNDING_MODE": "rint",
            "VLLM_ASCEND_QAT_MXFP8_GROUP_SIZE": "32",
            "VLLM_ASCEND_QAT_MXFP4_SCALE_ALG": "standard",
            "VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE": "false",
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_TARGETS": '["Fprop"]',
            "VLLM_ASCEND_QAT_MXFP8_ROTATION_SEED": "19",
            "VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS": "[]",
            "VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS": "",
            "VLLM_ASCEND_QAT_MXFP8_ACTIVATION_RESIDUAL_FPROP": str(enabled).lower(),
        }
        env.update(overrides)
        with patch.dict("os.environ", env):
            return fake_quant.initialize_mxfp8_fake_quant_config()

    def dense_apply(self):
        return load_function(
            "vllm_ascend/ops/linear.py",
            "apply",
            {"torch": torch, "prepare_mxfp8_activation": fake_quant.prepare_mxfp8_activation},
            "AscendUnquantizedLinearMethod",
        )

    def test_default_false_and_configuration_refresh(self):
        with patch.dict("os.environ", {}, clear=True):
            config = fake_quant.initialize_mxfp8_fake_quant_config()
        self.assertFalse(config.activation_residual_fprop)
        self.configure(True)
        self.assertTrue(fake_quant.get_mxfp8_fake_quant_config().activation_residual_fprop)
        self.assertTrue(fake_quant.is_mxfp8_activation_residual_enabled())
        self.configure(False)
        self.assertFalse(fake_quant.get_mxfp8_fake_quant_config().activation_residual_fprop)

    def test_residual_is_full_and_uses_rotated_coordinates(self):
        for rotation in ("false", "true"):
            with self.subTest(rotation=rotation):
                self.configure(VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE=rotation)
                activation = torch.randn(3, 64)
                quantized, residual = fake_quant.prepare_mxfp8_activation(activation)
                self.assertIsNotNone(residual)
                torch.testing.assert_close(quantized, fake_quant.fake_quant_mxfp8_activation(activation))
                torch.testing.assert_close(residual, fake_quant._maybe_rotate(activation) - quantized)
                self.assertEqual(residual.dtype, activation.dtype)
        for dtype in (torch.float16, torch.bfloat16):
            self.configure()
            _, residual = fake_quant.prepare_mxfp8_activation(torch.randn(2, 32).to(dtype))
            self.assertEqual(residual.dtype, dtype)

    def test_no_compensation_if_fake_quant_or_activation_quantization_is_disabled(self):
        for overrides in (
            {"VLLM_ASCEND_QAT_FAKE_QUANT": "0"},
            {"VLLM_ASCEND_QAT_FAKE_QUANT_MODE": "w8a16_mxfp8"},
        ):
            with self.subTest(overrides=overrides):
                self.configure(**overrides)
                activation = torch.randn(2, 32)
                _, residual = fake_quant.prepare_mxfp8_activation(activation)
                self.assertIsNone(residual)

    def test_no_fprop_qdq_still_executes_zero_residual_gemm(self):
        self.configure(VLLM_ASCEND_QAT_MXFP8_FAKE_QUANT_TARGETS='["Dgrad", "Wgrad"]')
        activation = torch.randn(2, 32)
        quantized, residual = fake_quant.prepare_mxfp8_activation(activation)
        self.assertIsNotNone(residual)
        self.assertTrue(torch.equal(quantized, activation))
        self.assertTrue(torch.equal(residual, torch.zeros_like(activation)))
        layer = SimpleNamespace(weight=torch.randn(8, 32), _mxfp8_fake_quant_enabled=True)
        with patch.object(torch.ops.vllm, "unquantized_gemm", side_effect=F.linear, create=True) as gemm:
            result = self.dense_apply()(None, layer, activation)
        self.assertEqual(gemm.call_count, 2)
        torch.testing.assert_close(result, F.linear(activation, layer.weight), rtol=0, atol=0)

    def test_dense_two_gemms_restore_only_activation_error_bias_once(self):
        apply = self.dense_apply()
        activation, weight, bias = torch.randn(4, 64), torch.randn(32, 64), torch.randn(32)
        for rotation in ("false", "true"):
            with self.subTest(rotation=rotation):
                self.configure(VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE=rotation)
                quantized_weight = fake_quant.fake_quant_mxfp8_weight(weight)
                layer = SimpleNamespace(weight=quantized_weight, _mxfp8_fake_quant_enabled=True)
                with patch.object(torch.ops.vllm, "unquantized_gemm", side_effect=F.linear, create=True) as gemm:
                    result = apply(None, layer, activation, bias)
                expected = F.linear(fake_quant._maybe_rotate(activation), quantized_weight, bias)
                torch.testing.assert_close(result, expected, atol=3e-5, rtol=1e-5)
                self.assertFalse(torch.allclose(result, F.linear(activation, weight, bias)))
                self.assertEqual(gemm.call_count, 2)
                self.assertIs(gemm.call_args_list[0].args[1], gemm.call_args_list[1].args[1])
                self.assertIsNone(gemm.call_args_list[1].args[2])

    def test_zero_residual_keeps_second_gemm(self):
        self.configure()
        layer = SimpleNamespace(weight=torch.randn(8, 32), _mxfp8_fake_quant_enabled=True)
        with patch.object(torch.ops.vllm, "unquantized_gemm", side_effect=F.linear, create=True) as gemm:
            result = self.dense_apply()(None, layer, torch.zeros(2, 32))
        self.assertEqual(gemm.call_count, 2)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_dense_disabled_or_ignored_layer_keeps_one_gemm(self):
        activation, weight = torch.randn(3, 32), torch.randn(8, 32)
        for enabled, layer_enabled in ((False, True), (True, False)):
            self.configure(enabled)
            layer = SimpleNamespace(weight=weight, _mxfp8_fake_quant_enabled=layer_enabled)
            expected_activation = fake_quant.fake_quant_mxfp8_activation(activation) if layer_enabled else activation
            with patch.object(torch.ops.vllm, "unquantized_gemm", side_effect=F.linear, create=True) as gemm:
                result = self.dense_apply()(None, layer, activation)
            torch.testing.assert_close(result, F.linear(expected_activation, weight), rtol=0, atol=0)
            self.assertEqual(gemm.call_count, 1)

    def test_fallback_and_ignore_filter_remain_effective(self):
        self.configure(
            VLLM_ASCEND_QAT_MXFP8_IGNORE_PATTERNS='["lm_head", "router"]',
            VLLM_ASCEND_QAT_MXFP8_FALLBACK_LAYERS="3-5",
        )
        self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.3.mlp.experts"))
        self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.4.self_attn.o_proj"))
        self.assertFalse(fake_quant.should_fake_quantize_layer("model.layers.7.router"))
        self.assertTrue(fake_quant.should_fake_quantize_layer("model.layers.6.mlp.experts"))

    def moe_apply(self, gmm):
        activation = SimpleNamespace(SWIGLUOAI="swigluoai", SWIGLUSTEP="swiglustep", GELU="gelu", GELU_TANH="gelutanh")
        npu = SimpleNamespace(
            npu_grouped_matmul=gmm,
            npu_swiglu=lambda x: F.silu(x.chunk(2, dim=-1)[0]) * x.chunk(2, dim=-1)[1],
        )
        return load_function(
            "vllm_ascend/ops/fused_moe/moe_mlp.py",
            "unquant_apply_mlp",
            {
                "torch": torch,
                "torch_npu": npu,
                "MoEActivation": activation,
                "prepare_mxfp8_activation": fake_quant.prepare_mxfp8_activation,
            },
        )

    def test_moe_compensates_before_nonlinearity_and_down_with_same_topk_scales(self):
        activation = torch.randn(3, 32) / 3
        original_w1, original_w2 = torch.randn(3, 32, 64) / 4, torch.randn(3, 32, 32) / 4
        bias1, bias2 = torch.randn(3, 64), torch.randn(3, 32)
        scales = torch.tensor([[0.3], [0.8], [0.6]])
        for rotation in ("false", "true"):
            for list_type in (0, 1):
                with self.subTest(rotation=rotation, group_list_type=list_type):
                    self.configure(VLLM_ASCEND_QAT_MXFP8_ROTATION_ENABLE=rotation)
                    w1 = fake_quant.fake_quant_mxfp8_transposed_moe_weight(original_w1)
                    w2 = fake_quant.fake_quant_mxfp8_transposed_moe_weight(original_w2)
                    groups = torch.tensor([2, 0, 1] if list_type == 1 else [2, 2, 3])
                    gmm = Mock(side_effect=cpu_grouped_matmul)
                    result, event = self.moe_apply(gmm)(
                        activation,
                        w1,
                        w2,
                        groups,
                        bias1,
                        bias2,
                        group_list_type=list_type,
                        topk_scales=scales,
                        need_trans=False,
                        fake_quant_enabled=True,
                    )
                    gate_up = cpu_grouped_matmul(
                        x=[fake_quant._maybe_rotate(activation)],
                        weight=[w1],
                        bias=[bias1],
                        group_list=groups,
                        group_list_type=list_type,
                    )[0]
                    gate, up = gate_up.chunk(2, dim=-1)
                    down_input = fake_quant._maybe_rotate(F.silu(gate) * up) * scales
                    expected = cpu_grouped_matmul(
                        x=[down_input],
                        weight=[w2],
                        bias=[bias2],
                        group_list=groups,
                        group_list_type=list_type,
                    )[0]
                    torch.testing.assert_close(result, expected, rtol=2e-5, atol=3e-5)
                    self.assertIsNone(event)
                    self.assertEqual(gmm.call_count, 4)
                    calls = gmm.call_args_list
                    for main, residual in ((calls[0], calls[1]), (calls[2], calls[3])):
                        self.assertIs(main.kwargs["weight"][0], residual.kwargs["weight"][0])
                        self.assertIs(main.kwargs["group_list"], residual.kwargs["group_list"])
                        self.assertIsNone(residual.kwargs["bias"])

    def test_moe_disabled_and_empty_tokens(self):
        for enabled, count in ((False, 2), (True, 4)):
            self.configure(enabled)
            gmm = Mock(side_effect=cpu_grouped_matmul)
            result, _ = self.moe_apply(gmm)(
                torch.empty(0, 32),
                torch.randn(2, 32, 64),
                torch.randn(2, 32, 32),
                torch.tensor([0, 0]),
                need_trans=False,
                fake_quant_enabled=True,
            )
            self.assertEqual(result.shape, (0, 32))
            self.assertEqual(gmm.call_count, count)

    def test_moe_disabled_retains_staged_qdq_and_transposed_weights(self):
        self.configure(False)
        activation = torch.randn(3, 32) / 3
        w1 = fake_quant.fake_quant_mxfp8_transposed_moe_weight(torch.randn(2, 32, 64) / 4)
        w2 = fake_quant.fake_quant_mxfp8_transposed_moe_weight(torch.randn(2, 32, 32) / 4)
        groups = torch.tensor([2, 1])
        gmm = Mock(side_effect=cpu_grouped_matmul)
        result, _ = self.moe_apply(gmm)(
            activation,
            w1.transpose(1, 2),
            w2.transpose(1, 2),
            groups,
            need_trans=True,
            fake_quant_enabled=True,
        )
        gate_up = cpu_grouped_matmul(
            x=[fake_quant.fake_quant_mxfp8_activation(activation)],
            weight=[w1],
            group_list=groups,
            group_list_type=1,
        )[0]
        gate, up = gate_up.chunk(2, dim=-1)
        expected = cpu_grouped_matmul(
            x=[fake_quant.fake_quant_mxfp8_activation(F.silu(gate) * up)],
            weight=[w2],
            group_list=groups,
            group_list_type=1,
        )[0]
        self.assertEqual(gmm.call_count, 2)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_allreduce_fusion_uses_quant_method_and_existing_collective(self):
        for enabled, layer_enabled in ((False, False), (False, True), (True, True)):
            self.configure(enabled)
            quant = SimpleNamespace(apply=Mock(return_value=torch.ones(2, 8)))
            collective = Mock(side_effect=lambda x: x * 2)
            fused = Mock(return_value=torch.full((2, 8), 7.0))
            op = SimpleNamespace(
                layer=SimpleNamespace(_mxfp8_fake_quant_enabled=layer_enabled, weight=torch.randn(8, 32)),
                quant_method=quant,
                get_input_parallel=lambda x: x,
                tp_rank=0,
                tp_size=2,
                skip_bias_add=False,
                bias=None,
                reduce_results=True,
                hcomm_info="test",
                comm_group=SimpleNamespace(all_reduce=collective),
            )
            apply = load_function(
                "vllm_ascend/ops/linear_op.py",
                "apply_impl",
                {
                    "torch_npu": SimpleNamespace(npu_mm_all_reduce_base=fused),
                    "is_mxfp8_activation_residual_enabled": fake_quant.is_mxfp8_activation_residual_enabled,
                },
                "MatmulAllreduceRowParallelOp",
            )
            output, _ = apply(op, torch.randn(2, 32))
            self.assertEqual(quant.apply.call_count, int(layer_enabled))
            self.assertEqual(collective.call_count, int(layer_enabled))
            self.assertEqual(fused.call_count, int(not layer_enabled))
            self.assertEqual(output.shape, (2, 8))

    def test_reduce_scatter_fusion_does_not_bypass_compensation(self):
        class UnquantizedLinearMethod:
            apply = Mock(return_value=torch.ones(4, 8))

        modules = {}
        for name, attributes in (
            ("vllm.model_executor.layers.linear", {"UnquantizedLinearMethod": UnquantizedLinearMethod}),
            ("vllm_ascend.quantization.method_adapters", {"AscendLinearMethod": type("Adapter", (), {})}),
            ("vllm_ascend.quantization.methods", {"AscendW8A8LinearMethod": type("W8A8", (), {})}),
        ):
            module = types.ModuleType(name)
            vars(module).update(attributes)
            modules[name] = module
        for enabled, layer_enabled in ((False, False), (False, True), (True, True)):
            self.configure(enabled)
            quant = UnquantizedLinearMethod()
            quant.apply.reset_mock()
            fused, collective = Mock(return_value=torch.ones(2, 8)), Mock(return_value=torch.ones(2, 8))
            layer = SimpleNamespace(
                prefix="model.layers.1.mlp.down_proj",
                quant_method=quant,
                _mxfp8_fake_quant_enabled=layer_enabled,
                weight=torch.randn(8, 32),
                tp_size=2,
                tp_rank=0,
            )
            op = SimpleNamespace(layer=layer, quant_method=quant)
            namespace = {
                "torch": torch,
                "F": F,
                "torch_npu": SimpleNamespace(npu_mm_reduce_scatter_base=fused),
                "_EXTRA_CTX": SimpleNamespace(flash_comm_v1_enabled=True, mmrs_fusion=True, pad_size=0),
                "enable_dsa_cp": lambda: False,
                "get_tp_group": lambda: SimpleNamespace(
                    device_group=SimpleNamespace(
                        _get_backend=lambda device: SimpleNamespace(get_hccl_comm_name=lambda rank: "test")
                    )
                ),
                "tensor_model_parallel_reduce_scatter": collective,
                "is_mxfp8_activation_residual_enabled": fake_quant.is_mxfp8_activation_residual_enabled,
            }
            apply = load_function(
                "vllm_ascend/ops/linear_op.py",
                "matmul_and_reduce",
                namespace,
                "SequenceRowParallelOp",
            )
            with patch.dict(sys.modules, modules), patch.object(torch, "device", return_value="npu"):
                output = apply(op, torch.randn(4, 32), None)
            self.assertEqual(quant.apply.call_count, int(layer_enabled))
            self.assertEqual(collective.call_count, int(layer_enabled))
            self.assertEqual(fused.call_count, int(not layer_enabled))
            self.assertEqual(output.shape, (2, 8))
