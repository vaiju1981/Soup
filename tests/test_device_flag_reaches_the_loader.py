"""``--device`` decides where ``infer`` / ``chat`` / ``diff`` / ``serve`` load the model.

Each command resolved ``--device`` (or detected one), printed it in its panel,
and then loaded with a hard-coded ``device_map="auto"``, which picks an
accelerator whenever one exists. ``soup infer --device cpu`` on a Mac therefore
loaded onto MPS while the panel said cpu. An explicit device now pins the whole
model there; without ``--device`` the load is ``"auto"`` as before.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

runner = CliRunner()


class _LoadedError(Exception):
    """Raised by the fake loader: the device_map is all these tests need."""


class _Tokenizer:
    pad_token = None
    eos_token = "<eos>"


@pytest.fixture
def load_calls(monkeypatch):
    """Record every ``from_pretrained`` call, then stop the command."""
    import transformers

    calls: list = []

    def fake_model(name, **kwargs):
        calls.append((name, kwargs))
        raise _LoadedError

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Tokenizer()
    )
    return calls


def _model_dir(path, kind: str):
    path.mkdir()
    if kind == "adapter":
        (path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/base"})
        )
    else:
        (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return path.name


def _argv(command: str, model: str) -> list[str]:
    return {
        "infer": ["infer", "--model", model, "--input", "p.jsonl", "--output", "o.jsonl"],
        "chat": ["chat", "--model", model],
        "diff": ["diff", "--model-a", model, "--model-b", model, "--prompt", "hi"],
        "serve": ["serve", "--model", model],
    }[command]


@pytest.mark.parametrize("command", ["infer", "chat", "diff", "serve"])
@pytest.mark.parametrize("kind", ["full", "adapter"])
@pytest.mark.parametrize(
    "device_args, expected",
    [
        ([], "auto"),
        (["--device", "cpu"], {"": "cpu"}),
        (["--device", "mps"], {"": "mps"}),
        (["--device", "cuda:1"], {"": "cuda:1"}),
    ],
    ids=["default", "cpu", "mps", "cuda-1"],
)
def test_the_requested_device_is_the_device_map(
    tmp_path, monkeypatch, load_calls, command, kind, device_args, expected
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p.jsonl").write_text('{"prompt": "hi"}\n')
    model = _model_dir(tmp_path / "model", kind)

    result = runner.invoke(app, _argv(command, model) + device_args)

    assert isinstance(result.exception, _LoadedError), result.output
    loaded, kwargs = load_calls[0]
    assert loaded == ("org/base" if kind == "adapter" else model)
    assert kwargs["device_map"] == expected


def test_the_panel_device_is_where_it_loads(tmp_path, monkeypatch, load_calls):
    """The panel always showed the requested device; now the load agrees with it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "p.jsonl").write_text('{"prompt": "hi"}\n')
    model = _model_dir(tmp_path / "model", "full")
    result = runner.invoke(app, _argv("infer", model) + ["--device", "cpu"])
    assert "Device:   cpu" in result.output
    assert load_calls[0][1]["device_map"] == {"": "cpu"}


class TestLoadDeviceMap:
    @pytest.mark.parametrize("requested", [None, ""])
    def test_no_device_keeps_auto(self, requested):
        from soup_cli.utils.gpu import load_device_map

        assert load_device_map(requested) == "auto"

    @pytest.mark.parametrize("requested", ["cpu", "mps", "cuda", "cuda:1"])
    def test_an_explicit_device_pins_the_whole_model(self, requested):
        from soup_cli.utils.gpu import load_device_map

        assert load_device_map(requested) == {"": requested}


class TestServeDraftModel:
    """Speculative decoding needs the draft beside the main model."""

    @pytest.fixture
    def draft_kwargs(self, monkeypatch):
        from unittest.mock import MagicMock

        import transformers

        seen: dict = {}

        def fake(name, **kwargs):
            seen.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake)
        return seen

    def test_the_main_models_device_map_is_followed(self, draft_kwargs):
        import torch

        from soup_cli.commands.serve import _load_draft_model

        _load_draft_model("draft", "mps", {"": "mps"})
        assert draft_kwargs["device_map"] == {"": "mps"}
        assert draft_kwargs["torch_dtype"] == torch.float16

    @pytest.mark.parametrize("device, expected", [("cpu", "cpu"), ("cuda", "auto")])
    def test_control_without_a_map_the_device_decides(self, draft_kwargs, device, expected):
        from soup_cli.commands.serve import _load_draft_model

        _load_draft_model("draft", device)
        assert draft_kwargs["device_map"] == expected
