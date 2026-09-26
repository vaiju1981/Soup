"""Loading a bf16 checkpoint onto MPS as float16 segfaulted every Soup command.

transformers 5.x materialises checkpoint tensors on a four-thread pool, and
with MPS as the target and a dtype cast (``soup infer`` / ``soup chat`` load
float16; most checkpoints are bfloat16) that pool segfaults. ``soup infer
--model Qwen/Qwen2.5-0.5B-Instruct`` exited 139 at "Loading weights: 0/290",
and ``--device cpu`` did not help because the load uses ``device_map="auto"``,
which still picks MPS. The CLI now defaults ``HF_DEACTIVATE_ASYNC_LOAD`` on
Apple Silicon, so transformers loads on one thread.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_ENV = "HF_DEACTIVATE_ASYNC_LOAD"


def _on(monkeypatch, system: str, machine: str) -> None:
    from soup_cli.utils import gpu

    monkeypatch.setattr(gpu.sys, "platform", system)
    monkeypatch.setattr(gpu.platform, "machine", lambda: machine)
    monkeypatch.delenv(_ENV, raising=False)


class TestTheDefault:
    def test_apple_silicon_loads_on_one_thread(self, monkeypatch):
        from soup_cli.utils.gpu import serialize_weight_loading_on_apple_silicon

        _on(monkeypatch, "darwin", "arm64")
        serialize_weight_loading_on_apple_silicon()
        assert os.environ[_ENV] == "1"

    def test_an_explicit_value_wins(self, monkeypatch):
        """Someone who has measured their own setup can turn parallel loading back on."""
        from soup_cli.utils.gpu import serialize_weight_loading_on_apple_silicon

        _on(monkeypatch, "darwin", "arm64")
        monkeypatch.setenv(_ENV, "0")
        serialize_weight_loading_on_apple_silicon()
        assert os.environ[_ENV] == "0"

    @pytest.mark.parametrize(
        "system, machine", [("linux", "x86_64"), ("win32", "AMD64"), ("darwin", "x86_64")]
    )
    def test_control_no_mps_no_change(self, monkeypatch, system, machine):
        """Only Apple Silicon has MPS; CUDA hosts keep transformers' parallel loader."""
        from soup_cli.utils.gpu import serialize_weight_loading_on_apple_silicon

        _on(monkeypatch, system, machine)
        serialize_weight_loading_on_apple_silicon()
        assert _ENV not in os.environ


def test_the_cli_sets_it_before_any_command_runs():
    """Wired at ``soup_cli.cli`` import, so every command and trainer inherits it."""
    import platform

    env = {k: v for k, v in os.environ.items() if k != _ENV}
    probe = "import os, soup_cli.cli; print(os.environ.get('HF_DEACTIVATE_ASYNC_LOAD'))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, check=True
    )
    apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
    assert result.stdout.strip() == ("1" if apple_silicon else "None")


@pytest.mark.smoke
class TestOnRealMps:
    """Downloads SmolLM2-135M-Instruct (~270 MB, bfloat16 on disk)."""

    _MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

    @pytest.fixture(autouse=True)
    def _need_mps(self):
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("needs an Apple Silicon MPS device")

    def _infer(self, tmp_path, **env_overrides) -> subprocess.CompletedProcess:
        prompts = tmp_path / "prompts.jsonl"
        prompts.write_text(json.dumps({"prompt": "hello"}) + "\n")
        env = {k: v for k, v in os.environ.items() if k != _ENV}
        env.update(env_overrides)
        return subprocess.run(
            [
                sys.executable, "-m", "soup_cli", "infer", "--model", self._MODEL,
                "--input", str(prompts), "--output", str(tmp_path / "out.jsonl"),
                "--temperature", "0", "--max-tokens", "8",
            ],
            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=600,
        )

    def test_soup_infer_loads_a_bf16_checkpoint_on_mps(self, tmp_path):
        result = self._infer(tmp_path)
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        rows = (tmp_path / "out.jsonl").read_text().splitlines()
        assert len(rows) == 1 and json.loads(rows[0])["response"]

    def test_control_parallel_loading_still_crashes(self, tmp_path):
        """What the default works around. If this starts passing, transformers
        fixed its parallel MPS loader: drop the default in
        ``utils/gpu.serialize_weight_loading_on_apple_silicon`` and this test."""
        result = self._infer(tmp_path, **{_ENV: "0"})
        assert result.returncode != 0, "parallel loading onto MPS no longer crashes"
