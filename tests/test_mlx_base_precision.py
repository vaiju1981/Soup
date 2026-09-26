"""``training.quantization`` on the MLX backend.

``load_mlx_model`` used to discard the setting (``del quantization``) and load
whatever the checkpoint held, while the setup panel printed ``Quant: 4bit``.
Pointing ``base`` at a full-precision model therefore trained in bf16 at about
1.7x the memory of the 4-bit run the config and the panel described, measured
on Qwen2.5-0.5B-Instruct: 1.87 GB peak against 1.08 GB for its mlx-community
4-bit checkpoint.

The setting now means what it means on the transformers backend: a
full-precision base is quantized at load (mlx-lm's QLoRA path, the same call
``mlx_lm.convert -q`` makes). An already-quantized checkpoint trains at its own
precision and says so when that differs from the setting.
"""

from __future__ import annotations

import json
import sys
import types
from io import StringIO

import pytest

from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx

_BASE = "org/model"
_MLX_4BIT = {"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}


def _plan(model_config: dict, quantization: str):
    from soup_cli.utils.mlx import plan_mlx_base_precision

    return plan_mlx_base_precision(model_config, quantization, base=_BASE)


class TestAFullPrecisionBaseIsQuantizedToTheSetting:
    @pytest.mark.parametrize("setting, bits", [("4bit", 4), ("8bit", 8)])
    def test_4bit_and_8bit_quantize_at_load(self, setting, bits):
        precision = _plan({}, setting)
        assert precision.source == "load_time"
        assert (precision.bits, precision.group_size, precision.mode) == (bits, 64, "affine")
        assert precision.mismatch is None

    def test_none_trains_it_as_stored(self):
        precision = _plan({}, "none")
        assert precision.source == "full_precision"
        assert precision.bits is None
        assert precision.mismatch is None

    def test_the_description_says_it_was_quantized_here(self):
        assert "quantized from full precision at load" in _plan({}, "4bit").describe()


class TestAQuantizedCheckpointTrainsAtItsOwnPrecision:
    def test_a_matching_setting_is_silent(self):
        precision = _plan(_MLX_4BIT, "4bit")
        assert precision.source == "checkpoint"
        assert precision.bits == 4
        assert precision.mismatch is None

    @pytest.mark.parametrize(
        "model_config, setting, wanted",
        [
            (_MLX_4BIT, "none", "full precision"),
            ({"quantization": {"group_size": 64, "bits": 8}}, "4bit", "4-bit"),
            ({"quantization": {"group_size": 64, "bits": 6}}, "4bit", "4-bit"),
        ],
        ids=["none-on-4bit", "default-4bit-on-8bit", "4bit-on-6bit"],
    )
    def test_a_different_setting_is_named_not_refused(self, model_config, setting, wanted):
        """A warning, not a refusal: the schema has no value for 6-bit, so a
        refusal would leave that checkpoint with no config that trains it."""
        precision = _plan(model_config, setting)
        assert precision.source == "checkpoint"
        assert precision.bits == model_config["quantization"]["bits"]
        assert f"training.quantization: {setting} is not applied" in precision.mismatch
        assert _BASE in precision.mismatch
        assert f"to train at {wanted}" in precision.mismatch

    def test_mxfp4_counts_as_4bit(self):
        precision = _plan({"quantization": {"group_size": 32, "bits": 4, "mode": "mxfp4"}}, "4bit")
        assert precision.mismatch is None
        assert "4-bit mxfp4" in precision.describe()

    def test_per_layer_overrides_are_reported(self):
        """mlx-lm's mixed recipes keep per-layer entries beside the defaults;
        calling such a checkpoint plain 4-bit would understate it."""
        mixed = {"quantization": {"group_size": 64, "bits": 4, "model.layers.0.mlp": {"bits": 6}}}
        precision = _plan(mixed, "4bit")
        assert precision.per_layer_overrides
        assert "per-layer overrides" in precision.describe()
        assert not _plan(_MLX_4BIT, "4bit").per_layer_overrides

    def test_bitnet_is_read_from_the_legacy_key(self):
        """mlx-lm applies bitnet without writing ``quantization``."""
        precision = _plan({"quantization_config": {"quant_method": "bitnet"}}, "4bit")
        assert precision.source == "checkpoint"
        assert precision.mode == "bitnet"
        assert precision.mismatch is not None


def test_a_setting_mlx_cannot_express_is_refused():
    with pytest.raises(ValueError, match="has no MLX equivalent"):
        _plan({}, "gptq")


class TestLoadMlxModelAppliesThePlan:
    """``load_mlx_model`` against recording fakes, so it runs without MLX."""

    @pytest.fixture
    def fakes(self, monkeypatch):
        calls: list = []
        loaded = {"model": object(), "config": {}}

        mlx_core = types.ModuleType("mlx.core")
        mlx_core.eval = lambda params: calls.append(("eval",))
        mlx_lm = types.ModuleType("mlx_lm")

        def load(path, **kwargs):
            calls.append(("load", path, kwargs))
            return loaded["model"], "tokenizer", loaded["config"]

        mlx_lm.load = load
        mlx_lm_utils = types.ModuleType("mlx_lm.utils")

        def quantize_model(model, config, **kwargs):
            calls.append(("quantize", kwargs))
            return model, config

        mlx_lm_utils.quantize_model = quantize_model
        # `import mlx.core as mx` resolves `core` as an attribute of `mlx`.
        mlx_root = types.ModuleType("mlx")
        mlx_root.core = mlx_core
        for name, module in {
            "mlx": mlx_root,
            "mlx.core": mlx_core,
            "mlx_lm": mlx_lm,
            "mlx_lm.utils": mlx_lm_utils,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)
        loaded["model"] = types.SimpleNamespace(parameters=lambda: {})
        return calls, loaded

    def test_a_full_precision_base_is_quantized_before_it_is_evaluated(self, fakes):
        from soup_cli.utils.mlx import load_mlx_model

        calls, _ = fakes
        _, tokenizer, precision = load_mlx_model(_BASE, quantization="4bit")

        assert tokenizer == "tokenizer"
        assert precision.source == "load_time"
        assert calls == [
            ("load", _BASE, {"lazy": True, "return_config": True}),
            ("quantize", {"group_size": 64, "bits": 4, "mode": "affine"}),
            ("eval",),
        ]

    def test_control_a_quantized_checkpoint_is_not_quantized_again(self, fakes):
        from soup_cli.utils.mlx import load_mlx_model

        calls, loaded = fakes
        loaded["config"] = _MLX_4BIT
        load_mlx_model(_BASE, quantization="4bit")
        assert [call[0] for call in calls] == ["load", "eval"]


class TestTheWrapperReportsAndRecordsThePrecision:
    def _run(self, tmp_path, monkeypatch, precision) -> tuple[str, dict]:
        from rich.console import Console

        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer import mlx_sft
        from soup_cli.utils import mlx as mlx_utils

        _install_fake_mlx(monkeypatch)
        monkeypatch.setattr(
            mlx_utils, "load_mlx_model", lambda *a, **k: (_FakeMlxModel(), object(), precision)
        )
        buffer = StringIO()
        monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=300))
        cfg = SoupConfig(
            base=_BASE,
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(epochs=1, batch_size=1),
            output=str(tmp_path),
        )
        wrapper = mlx_sft.MLXSFTTrainerWrapper(cfg)
        wrapper.setup({"train": [{"text": "hi"}], "val": []})
        wrapper.train()
        adapter_config = json.loads((tmp_path / "adapter_config.json").read_text())
        return buffer.getvalue(), adapter_config

    def test_the_precision_is_printed_and_written_to_the_adapter(self, tmp_path, monkeypatch):
        precision = _plan({}, "4bit")
        out, adapter_config = self._run(tmp_path, monkeypatch, precision)
        assert f"Base precision: {precision.describe()}" in out
        assert adapter_config["base_quantization"] == precision.as_metadata()
        assert adapter_config["base_quantization"]["source"] == "load_time"

    def test_a_mismatch_is_printed(self, tmp_path, monkeypatch):
        precision = _plan({"quantization": {"group_size": 64, "bits": 8}}, "4bit")
        out, adapter_config = self._run(tmp_path, monkeypatch, precision)
        assert "training.quantization: 4bit is not applied" in out
        assert adapter_config["base_quantization"]["bits"] == 8

    def test_8bit_is_no_longer_reported_as_ignored(self, monkeypatch):
        from rich.console import Console

        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer import mlx_sft

        buffer = StringIO()
        monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=300))
        cfg = SoupConfig(
            base=_BASE,
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(quantization="8bit"),
        )
        mlx_sft.MLXSFTTrainerWrapper(cfg)._check_unsupported()
        assert "quantization" not in buffer.getvalue()


class TestAgainstRealMlx:
    """QLoRA on a base quantized at load, with the real mlx and mlx-lm."""

    @pytest.fixture(autouse=True)
    def _need_mlx(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")

    @staticmethod
    def _load(monkeypatch, quantization: str):
        import mlx_lm
        from mlx_lm.models import llama

        from soup_cli.utils.mlx import load_mlx_model

        def tiny_full_precision_load(path, **kwargs):
            model = llama.Model(
                llama.ModelArgs(
                    model_type="llama",
                    hidden_size=64,
                    num_hidden_layers=2,
                    intermediate_size=128,
                    num_attention_heads=4,
                    rms_norm_eps=1e-5,
                    vocab_size=128,
                    num_key_value_heads=4,
                )
            )
            return model, object(), {"model_type": "llama"}

        monkeypatch.setattr(mlx_lm, "load", tiny_full_precision_load)
        return load_mlx_model(_BASE, quantization=quantization)

    @staticmethod
    def _linear_bits(model) -> set:
        import mlx.nn as nn

        return {
            module.bits if isinstance(module, nn.QuantizedLinear) else None
            for _, module in model.named_modules()
            if isinstance(module, (nn.Linear, nn.QuantizedLinear))
        }

    @pytest.mark.parametrize("setting, bits", [("4bit", 4), ("8bit", 8)])
    def test_every_linear_layer_is_quantized_to_the_setting(self, monkeypatch, setting, bits):
        model, _, precision = self._load(monkeypatch, setting)
        assert precision.source == "load_time"
        assert self._linear_bits(model) == {bits}

    def test_control_none_leaves_the_base_unquantized(self, monkeypatch):
        model, _, precision = self._load(monkeypatch, "none")
        assert precision.source == "full_precision"
        assert self._linear_bits(model) == {None}

    def test_lora_trains_on_top_of_the_quantized_base(self, monkeypatch):
        """Only the LoRA matrices are trainable, over QuantizedLinear bases."""
        from mlx.utils import tree_flatten

        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        model, _, _ = self._load(monkeypatch, "4bit")
        from soup_cli.config.schema import DataConfig, SoupConfig

        cfg = SoupConfig(
            base=_BASE,
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
        )
        MLXSFTTrainerWrapper(cfg)._apply_lora(model)
        trainable = [name for name, _ in tree_flatten(model.trainable_parameters())]
        assert trainable
        assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)
