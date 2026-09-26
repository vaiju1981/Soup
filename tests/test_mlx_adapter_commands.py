"""Commands given an adapter trained with ``backend: mlx``.

The MLX trainer writes mlx-lm's adapter format (``fine_tune_type``,
``lora_parameters``, base under ``model``, ``adapters.safetensors``). After an
MLX run the training summary suggested ``soup chat`` / ``push`` / ``merge`` /
``export`` on that directory, and every one of them failed: the PEFT-based
commands with "Cannot detect base model" and ``push`` with "does not look like
a valid model". ``soup chat`` and ``soup merge`` now load it through mlx-lm, the
rest refuse it naming the merge to run first, and the summary suggests only
commands that take it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

runner = CliRunner()

_BASE = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _mlx_adapter(directory: Path, **overrides) -> Path:
    """The adapter_config.json the MLX trainer writes, trimmed to what matters."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "fine_tune_type": "lora",
        "model": _BASE,
        "num_layers": 24,
        "lora_parameters": {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj"]},
    }
    config.update(overrides)
    (directory / "adapter_config.json").write_text(json.dumps(config))
    (directory / "adapters.safetensors").write_bytes(b"")
    return directory


def _peft_adapter(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/base", "r": 8})
    )
    return directory


class TestFindMlxAdapter:
    def test_an_mlx_adapter_is_recognised_with_its_base(self, tmp_path):
        from soup_cli.utils.mlx_adapter import find_mlx_adapter

        adapter = find_mlx_adapter(_mlx_adapter(tmp_path / "out"))
        assert adapter is not None
        assert adapter.base == _BASE
        assert adapter.path == tmp_path / "out"

    def test_a_missing_base_is_none_not_a_different_format(self, tmp_path):
        from soup_cli.utils.mlx_adapter import find_mlx_adapter

        adapter = find_mlx_adapter(_mlx_adapter(tmp_path / "out", model=""))
        assert adapter is not None
        assert adapter.base is None

    def test_control_a_peft_adapter_is_not_mlx(self, tmp_path):
        from soup_cli.utils.mlx_adapter import find_mlx_adapter

        assert find_mlx_adapter(_peft_adapter(tmp_path / "peft")) is None

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
    def test_an_unreadable_config_is_not_mlx(self, tmp_path, content):
        from soup_cli.utils.mlx_adapter import find_mlx_adapter

        (tmp_path / "adapter_config.json").write_text(content)
        assert find_mlx_adapter(tmp_path) is None

    def test_a_directory_without_a_config_is_not_mlx(self, tmp_path):
        from soup_cli.utils.mlx_adapter import find_mlx_adapter

        assert find_mlx_adapter(tmp_path) is None


class TestTheRefusalNamesTheWayForward:
    @pytest.mark.parametrize(
        "argv",
        [
            ["export", "--model", "out"],
            ["infer", "--model", "out", "--input", "p.jsonl", "--output", "o.jsonl"],
            ["serve", "--model", "out"],
            ["infer", "--model", "out", "--input", "p.jsonl", "--output", "o.jsonl",
             "--task", "asr"],
            ["push", "--model", "out", "--repo", "me/model"],
        ],
        ids=["export", "infer", "serve", "infer-asr", "push"],
    )
    def test_each_command_refuses_with_the_merge_step(self, tmp_path, monkeypatch, argv):
        monkeypatch.chdir(tmp_path)
        _mlx_adapter(tmp_path / "out")
        (tmp_path / "p.jsonl").write_text('{"prompt": "hi"}\n')

        result = runner.invoke(app, argv)

        out = " ".join(strip_ansi(result.output).split())
        assert result.exit_code == 1, out
        assert "is an MLX adapter" in out
        assert "soup merge --adapter out --output out-merged" in out
        assert f"soup {argv[0]} --model out-merged" in out
        assert "soup chat --model out" in out

    def test_control_a_peft_adapter_passes_the_check(self, tmp_path):
        from rich.console import Console

        from soup_cli.utils.mlx_adapter import exit_if_mlx_adapter

        exit_if_mlx_adapter(_peft_adapter(tmp_path / "peft"), "export", Console())

    def test_the_check_exits_with_code_1(self, tmp_path):
        from io import StringIO

        from rich.console import Console

        from soup_cli.utils.mlx_adapter import exit_if_mlx_adapter

        with pytest.raises(typer.Exit) as raised:
            exit_if_mlx_adapter(_mlx_adapter(tmp_path / "out"), "export", Console(file=StringIO()))
        assert raised.value.exit_code == 1


class TestChatLoadsAnMlxAdapterThroughMlxLm:
    @pytest.fixture
    def calls(self, monkeypatch):
        from soup_cli.commands import chat

        calls: dict = {"generated": []}

        def load_mlx(adapter_path, base_model, trust_remote_code=False):
            calls["mlx_load"] = (Path(adapter_path), base_model)
            calls["trust"] = trust_remote_code
            return "model", "tokenizer"

        def generate_mlx(model, tokenizer, messages, **kwargs):
            calls["generated"].append((messages[-1]["content"], kwargs))
            return "Quellport."

        def refuse_peft(**kwargs):
            raise AssertionError("an MLX adapter must not reach the PEFT loader")

        monkeypatch.setattr(chat, "_load_mlx_adapter", load_mlx)
        monkeypatch.setattr(chat, "_generate_mlx", generate_mlx)
        monkeypatch.setattr(chat, "_load_model", refuse_peft)
        return calls

    def test_the_base_comes_from_the_mlx_config(self, tmp_path, calls):
        adapter = _mlx_adapter(tmp_path / "out")

        result = runner.invoke(
            app, ["chat", "--model", str(adapter), "-t", "0"], input="capital?\n/quit\n"
        )

        assert result.exit_code == 0, result.output
        assert calls["mlx_load"] == (adapter, _BASE)
        assert calls["generated"] == [("capital?", {"max_tokens": 512, "temperature": 0.0})]
        assert "Quellport." in result.output
        assert "Device: mlx" in strip_ansi(result.output)

    def test_an_explicit_base_wins(self, tmp_path, calls):
        adapter = _mlx_adapter(tmp_path / "out")
        result = runner.invoke(
            app, ["chat", "--model", str(adapter), "--base", "./local-base"], input="/quit\n"
        )
        assert result.exit_code == 0, result.output
        assert calls["mlx_load"] == (adapter, "./local-base")

    def test_trust_remote_code_reaches_the_mlx_loader(self, tmp_path, calls):
        adapter = _mlx_adapter(tmp_path / "out")
        result = runner.invoke(
            app, ["chat", "--model", str(adapter), "--trust-remote-code"], input="/quit\n"
        )
        assert result.exit_code == 0, result.output
        assert calls["trust"] is True

    def test_a_non_mlx_device_is_refused(self, tmp_path, calls):
        adapter = _mlx_adapter(tmp_path / "out")
        result = runner.invoke(app, ["chat", "--model", str(adapter), "--device", "cuda"])
        assert result.exit_code == 1
        assert "runs on --device mlx only" in " ".join(strip_ansi(result.output).split())
        assert "mlx_load" not in calls

    def test_without_mlx_lm_the_install_hint_keeps_its_extra(self, tmp_path, monkeypatch):
        """Rich reads an unescaped ``[mlx]`` as a tag and drops it, which would
        print an install command without the extra."""
        import sys

        monkeypatch.setitem(sys.modules, "mlx_lm", None)
        adapter = _mlx_adapter(tmp_path / "out")
        result = runner.invoke(app, ["chat", "--model", str(adapter)])
        assert result.exit_code == 1
        assert 'pip install "soup-cli[mlx]"' in strip_ansi(result.output)

    def test_control_a_peft_adapter_still_goes_through_peft(self, tmp_path, monkeypatch):
        from soup_cli.commands import chat

        seen = {}

        def load_peft(**kwargs):
            seen.update(kwargs)
            raise typer.Exit(0)

        monkeypatch.setattr(chat, "_load_model", load_peft)
        monkeypatch.setattr(
            chat, "_load_mlx_adapter", lambda *a: pytest.fail("PEFT adapter reached MLX")
        )
        adapter = _peft_adapter(tmp_path / "peft")
        result = runner.invoke(app, ["chat", "--model", str(adapter), "--device", "cpu"])
        assert result.exit_code == 0, result.output
        assert seen["base_model"] == "org/base"
        assert seen["is_adapter"] is True


class TestMergeFusesAnMlxAdapterThroughMlxLm:
    def test_the_adapter_base_and_dtype_reach_the_mlx_merge(self, tmp_path, monkeypatch):
        from soup_cli.commands import merge

        monkeypatch.chdir(tmp_path)
        _mlx_adapter(tmp_path / "out")
        seen = {}

        def fake_merge(adapter, base, output_dir, dtype, trust_remote_code):
            seen.update(adapter=adapter.path, base=base, output=output_dir, dtype=dtype)
            output_dir.mkdir()
            (output_dir / "model.safetensors").write_bytes(b"x")

        monkeypatch.setattr(merge, "merge_mlx_adapter", fake_merge)
        result = runner.invoke(
            app, ["merge", "--adapter", "out", "--output", "merged", "--dtype", "bfloat16"]
        )

        assert result.exit_code == 0, result.output
        assert seen == {
            "adapter": Path("out"),
            "base": _BASE,
            "output": Path("merged"),
            "dtype": "bfloat16",
        }
        assert "Merge Complete!" in result.output

    def test_a_bitsandbytes_save_format_is_refused(self, tmp_path, monkeypatch):
        from soup_cli.commands import merge

        monkeypatch.chdir(tmp_path)
        _mlx_adapter(tmp_path / "out")
        monkeypatch.setattr(merge, "merge_mlx_adapter", lambda *a: pytest.fail("merged"))
        result = runner.invoke(app, ["merge", "--adapter", "out", "--save-format", "4bit"])
        assert result.exit_code == 2
        assert "merges to --dtype only" in " ".join(strip_ansi(result.output).split())
        assert "Merge Plan" not in result.output, "refuse before announcing a plan"

    def test_a_merge_error_is_reported_not_raised(self, tmp_path, monkeypatch):
        from soup_cli.commands import merge

        monkeypatch.chdir(tmp_path)
        _mlx_adapter(tmp_path / "out")

        def broken(*args):
            raise ValueError("no LoRA layers")

        monkeypatch.setattr(merge, "merge_mlx_adapter", broken)
        result = runner.invoke(app, ["merge", "--adapter", "out"])
        assert result.exit_code == 1
        assert "Merge failed: no LoRA layers" in result.output

    def test_without_mlx_lm_the_install_hint_keeps_its_extra(self, tmp_path, monkeypatch):
        from soup_cli.commands import merge

        monkeypatch.chdir(tmp_path)
        _mlx_adapter(tmp_path / "out")

        def no_mlx(*args):
            raise ImportError("No module named 'mlx_lm'")

        monkeypatch.setattr(merge, "merge_mlx_adapter", no_mlx)
        result = runner.invoke(app, ["merge", "--adapter", "out"])
        assert result.exit_code == 1
        assert 'pip install "soup-cli[mlx]"' in strip_ansi(result.output)

    def test_control_a_peft_adapter_does_not_take_the_mlx_route(self, tmp_path, monkeypatch):
        from soup_cli.commands import merge

        monkeypatch.chdir(tmp_path)
        _peft_adapter(tmp_path / "peft")
        monkeypatch.setattr(merge, "merge_mlx_adapter", lambda *a: pytest.fail("MLX route"))
        result = runner.invoke(app, ["merge", "--adapter", "peft", "--base", "/no/such/base"])
        assert result.exit_code == 1
        assert "Merge failed" in result.output


class TestTheTrainingSummarySuggestsOnlyWhatWorks:
    def test_mlx_suggests_chat_and_merge_only(self):
        from soup_cli.commands.train import _next_steps

        text = _next_steps("./output", backend="mlx")
        assert "soup chat --model ./output" in text
        assert "soup merge --adapter ./output" in text
        assert "soup push --model ./output" not in text
        assert "soup export --model ./output" not in text

    @pytest.mark.parametrize("backend", ["transformers", "unsloth"])
    def test_control_other_backends_keep_all_four(self, backend):
        from soup_cli.commands.train import _next_steps

        text = _next_steps("./output", backend=backend)
        for command in ("soup chat", "soup push", "soup merge", "soup export"):
            assert command in text


class TestMergeAgainstRealMlx:
    """The fuse itself, on a tiny Llama saved to disk: no download."""

    @pytest.fixture(autouse=True)
    def _need_mlx(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")
        pytest.importorskip("tokenizers")

    @staticmethod
    def _tiny_base(directory: Path, quantized: bool) -> None:
        import mlx.core as mx
        from mlx_lm.models import llama
        from mlx_lm.utils import quantize_model, save_config, save_model
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast

        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "vocab_size": 128,
            "torch_dtype": "bfloat16",
        }
        model = llama.Model(llama.ModelArgs.from_dict(config))
        mx.eval(model.parameters())
        if quantized:
            model, config = quantize_model(model, config, group_size=64, bits=4)
        directory.mkdir()
        save_model(directory, model)
        save_config(config, config_path=directory / "config.json")
        (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
        (directory / "custom_modeling.py").write_text("# custom code\n")
        vocab = {f"t{i}": i for i in range(128)}
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0").save_pretrained(
            str(directory)
        )

    @staticmethod
    def _trained_adapter(base: Path, directory: Path) -> None:
        """A LoRA adapter with non-zero lora_b, saved the way the MLX trainer saves it."""
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers

        model, _ = load(str(base))
        model.freeze()
        parameters = {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj"]}
        linear_to_lora_layers(model, 2, parameters)
        weights = {
            name: mx.ones_like(value) * 0.01 if name.endswith("lora_b") else value
            for name, value in tree_flatten(model.trainable_parameters())
        }
        directory.mkdir()
        mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
        (directory / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "model": str(base),
                    "num_layers": 2,
                    "lora_parameters": parameters,
                }
            )
        )

    @pytest.mark.parametrize("quantized", [False, True], ids=["full-precision", "4bit"])
    def test_the_merged_model_is_dequantized_and_carries_the_lora_delta(self, tmp_path, quantized):
        import mlx.core as mx
        from mlx_lm import load

        from soup_cli.utils.mlx_adapter import find_mlx_adapter, merge_mlx_adapter

        self._tiny_base(tmp_path / "base", quantized=quantized)
        self._trained_adapter(tmp_path / "base", tmp_path / "adapter")

        merge_mlx_adapter(
            find_mlx_adapter(tmp_path / "adapter"), str(tmp_path / "base"),
            tmp_path / "merged", "float16",
        )

        config = json.loads((tmp_path / "merged" / "config.json").read_text())
        generation = json.loads((tmp_path / "merged" / "generation_config.json").read_text())
        assert generation["eos_token_id"] == [1, 2], "end-of-turn ids must survive the merge"
        assert (tmp_path / "merged" / "custom_modeling.py").is_file()
        assert "quantization" not in config and "quantization_config" not in config
        assert config["torch_dtype"] == "float16"
        merged, _ = load(str(tmp_path / "merged"))
        with_adapter, _ = load(str(tmp_path / "base"), adapter_path=str(tmp_path / "adapter"))
        tokens = mx.array([[1, 2, 3, 4]])
        assert merged.model.layers[0].self_attn.q_proj.weight.dtype == mx.float16
        assert mx.allclose(merged(tokens), with_adapter(tokens), atol=5e-2).item()

    def test_the_merged_model_loads_in_transformers(self, tmp_path):
        """The point of merging: the PEFT-free commands load it through transformers."""
        pytest.importorskip("torch")
        from transformers import AutoModelForCausalLM

        from soup_cli.utils.mlx_adapter import find_mlx_adapter, merge_mlx_adapter

        self._tiny_base(tmp_path / "base", quantized=True)
        self._trained_adapter(tmp_path / "base", tmp_path / "adapter")
        merge_mlx_adapter(
            find_mlx_adapter(tmp_path / "adapter"), str(tmp_path / "base"),
            tmp_path / "merged", "float16",
        )
        model = AutoModelForCausalLM.from_pretrained(str(tmp_path / "merged"))
        assert model.config.model_type == "llama"
