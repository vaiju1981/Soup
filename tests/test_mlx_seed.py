"""``training.seed`` / ``training.data_seed`` on the MLX backend.

#353 threaded the seed through every transformers task wrapper, and MLX was
left warning that it read neither field. The setup panel still printed
``Seed: 42 (unset default)`` for an MLX run, which seeded nothing, so two runs
of the same config drew different LoRA initialisations and batch orders.

MLX now seeds the two RNGs a run draws from, the way ``mlx_lm.lora --seed``
does: ``mx.random`` (LoRA's ``lora_a`` and dropout) and numpy's global RNG (the
batch order, which mlx-lm's ``iterate_batches`` and Soup's masked one shuffle
with ``np.random.permutation``). This replaces
``tests/test_issue353_mlx_seed_warning.py``, whose last test asked to be deleted
together with the warning once MLX seeded for real.
"""

from __future__ import annotations

import sys
import types
from io import StringIO

import pytest

from tests.test_issue634_mlx_resume import _FakeMlxModel, _install_fake_mlx


def _training(**fields):
    from soup_cli.config.schema import TrainingConfig

    return TrainingConfig(**fields)


def _mlx_config(**training):
    from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig

    return SoupConfig(
        base="mlx-community/tiny",
        task="sft",
        backend="mlx",
        data=DataConfig(train="./data/train.jsonl", format="chatml"),
        training=TrainingConfig(**training),
        output="./out",
    )


def _record_seeds(monkeypatch) -> dict:
    """Fake ``mlx.core.random.seed`` and patch ``numpy.random.seed`` to record calls."""
    import numpy as np

    seeds: dict = {}
    fake_core = types.ModuleType("mlx.core")
    fake_core.random = types.SimpleNamespace(seed=lambda seed: seeds.__setitem__("mx", seed))
    # The parent too: `import mlx.core as mx` resolves `core` as an attribute of
    # `mlx`, so a real mlx imported earlier in the session would bypass a fake
    # registered only as "mlx.core", and without mlx installed the import fails.
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(np.random, "seed", lambda seed: seeds.__setitem__("np", seed))
    return seeds


class TestBothRngsAreSeeded:
    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({}, {"mx": 42, "np": 42}),
            ({"seed": 7}, {"mx": 7, "np": 7}),
            ({"seed": 7, "data_seed": 3}, {"mx": 7, "np": 3}),
            ({"data_seed": 3}, {"mx": 42, "np": 3}),
        ],
        ids=["unset-is-42", "seed-drives-both", "data-seed-takes-order", "data-seed-alone"],
    )
    def test_the_configured_values_reach_both_rngs(self, monkeypatch, fields, expected):
        from soup_cli.utils.seeding import apply_mlx_training_seed

        seeds = _record_seeds(monkeypatch)
        applied = apply_mlx_training_seed(_training(**fields))
        assert seeds == expected
        assert applied == expected["mx"]

    def test_seed_zero_is_a_seed_not_unset(self, monkeypatch):
        """0 is the one value a truthiness check would turn back into 42."""
        from soup_cli.utils.seeding import apply_mlx_training_seed

        seeds = _record_seeds(monkeypatch)
        apply_mlx_training_seed(_training(seed=0, data_seed=0))
        assert seeds == {"mx": 0, "np": 0}


class TestTrainSeedsBeforeLoraIsCreated:
    def test_the_seed_is_applied_before_the_lora_layers_are_drawn(self, tmp_path, monkeypatch):
        """LoRA's ``lora_a`` is the run's first draw; a seed applied after it
        would leave the initialisation unseeded while the batch order was not."""
        from soup_cli.config.schema import DataConfig, SoupConfig, TrainingConfig
        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper

        _install_fake_mlx(monkeypatch)
        events: list = []
        sys.modules["mlx.core"].random = types.SimpleNamespace(
            seed=lambda seed: events.append(("seed", seed))
        )
        sys.modules["mlx_lm.tuner.utils"].linear_to_lora_layers = (
            lambda model, num_layers, config: events.append(("lora",))
        )
        cfg = SoupConfig(
            base="mlx-community/tiny",
            task="sft",
            backend="mlx",
            data=DataConfig(train="./data/train.jsonl", format="chatml"),
            training=TrainingConfig(epochs=1, batch_size=1, seed=7),
            output=str(tmp_path),
        )
        wrapper = MLXSFTTrainerWrapper(cfg)
        wrapper.model = _FakeMlxModel()
        wrapper.tokenizer = object()
        wrapper._dataset = {"train": [{"text": "hi"}], "val": []}

        wrapper.train()

        assert events == [("seed", 7), ("lora",)]


class TestTheRunNoLongerSaysTheSeedIsIgnored:
    def _warnings_for(self, monkeypatch, **training) -> str:
        from rich.console import Console

        from soup_cli.trainer import mlx_sft

        buffer = StringIO()
        monkeypatch.setattr(mlx_sft, "console", Console(file=buffer, width=200))
        mlx_sft.MLXSFTTrainerWrapper(_mlx_config(**training))._check_unsupported()
        return buffer.getvalue()

    def test_a_set_seed_and_data_seed_print_no_ignored_warning(self, monkeypatch):
        out = self._warnings_for(monkeypatch, seed=7, data_seed=3)
        assert "seed" not in out.lower()

    def test_control_the_other_unsupported_warnings_still_fire(self, monkeypatch):
        """CONTROL: the list lost two lines, not its purpose."""
        out = self._warnings_for(monkeypatch, seed=7, use_galore=True)
        assert "GaLore" in out
        assert "seed" not in out.lower()

    def test_doctor_no_longer_lists_the_seed_as_unread_on_mlx(self):
        from soup_cli.config.backend_support import check_config, unsupported_for

        assert check_config(_mlx_config(seed=7, data_seed=3)) == []
        declared = {entry.field for entry in unsupported_for("sft", "mlx")}
        assert not declared & {"training.seed", "training.data_seed"}


class TestAgainstRealMlx:
    """The seed has to make real MLX draws repeat, not merely reach a function."""

    @pytest.fixture(autouse=True)
    def _need_mlx(self):
        pytest.importorskip("mlx.core")
        pytest.importorskip("mlx_lm")

    @staticmethod
    def _lora_init(seed: int) -> dict:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.models import llama

        from soup_cli.trainer.mlx_sft import MLXSFTTrainerWrapper
        from soup_cli.utils.seeding import apply_mlx_training_seed

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
        mx.eval(model.parameters())
        wrapper = MLXSFTTrainerWrapper(_mlx_config(seed=seed))
        apply_mlx_training_seed(wrapper.config.training)
        wrapper._apply_lora(model)
        return dict(tree_flatten(model.trainable_parameters()))

    def test_the_same_seed_draws_the_same_lora_initialisation(self):
        import mlx.core as mx

        first, second = self._lora_init(7), self._lora_init(7)
        assert first.keys() == second.keys()
        assert any(name.endswith("lora_a") for name in first)
        assert all(mx.array_equal(first[name], second[name]).item() for name in first)

    def test_control_a_different_seed_draws_a_different_one(self):
        import mlx.core as mx

        first, second = self._lora_init(7), self._lora_init(8)
        lora_a = [name for name in first if name.endswith("lora_a")]
        assert not all(mx.array_equal(first[name], second[name]).item() for name in lora_a)

    @staticmethod
    def _batch_order(**training) -> list[int]:
        from mlx_lm.tuner.trainer import iterate_batches

        from soup_cli.utils.seeding import apply_mlx_training_seed

        # Sixteen rows of distinct lengths, one per batch: the lengths read back
        # in order ARE the shuffle, and a coincidental match is 1 in 16!.
        dataset = [([1] * length, 0) for length in range(1, 17)]
        apply_mlx_training_seed(_training(**training))
        return [
            int(lengths[0, 1].item())
            for _, lengths in iterate_batches(dataset, batch_size=1, max_seq_length=64)
        ]

    def test_mlx_lms_own_batch_order_follows_the_data_seed(self):
        assert self._batch_order(seed=7) == self._batch_order(seed=7)
        assert self._batch_order(seed=7, data_seed=3) == self._batch_order(seed=9, data_seed=3)

    def test_control_a_different_data_seed_reorders_the_batches(self):
        assert self._batch_order(seed=7, data_seed=3) != self._batch_order(seed=7, data_seed=4)
