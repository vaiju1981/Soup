"""Recognise an adapter directory written by ``backend: mlx``.

The MLX trainer writes mlx-lm's own ``adapter_config.json`` -- ``fine_tune_type``,
``lora_parameters`` and the base under ``model`` -- beside
``adapters.safetensors``. PEFT's format names the base
``base_model_name_or_path`` and stores ``adapter_model.safetensors`` with
different tensor names. ``soup chat`` / ``merge`` / ``export`` / ``infer`` /
``serve`` read the base from that key and load the adapter through PEFT, so on
an MLX directory each stopped at "Cannot detect base model", and naming the
base with ``--base`` only moved the failure to PEFT. ``soup push`` looks for
PEFT's files and called the directory not a model at all. The training summary
suggested these commands after an MLX run all the same.

``soup chat`` loads an MLX adapter through mlx-lm, and ``soup merge`` fuses one
into a dequantized model in the standard Hugging Face layout. The other
commands refuse it, naming that merge as the step to run first.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

#: Keys only mlx-lm's adapter config carries; a PEFT config has ``peft_type``.
_MLX_ADAPTER_KEYS = frozenset({"fine_tune_type", "lora_parameters"})


@dataclass(frozen=True)
class MlxAdapter:
    """An MLX adapter directory and the base model its config names, if any."""

    path: Path
    base: Optional[str]


def find_mlx_adapter(model_dir: "Path | str") -> Optional[MlxAdapter]:
    """Return the MLX adapter at ``model_dir``, or ``None`` when it is not one."""
    path = Path(model_dir)
    try:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict) or not _MLX_ADAPTER_KEYS <= config.keys():
        return None
    base = config.get("model")
    return MlxAdapter(path=path, base=base if isinstance(base, str) and base else None)


def unsupported_message(adapter: MlxAdapter, command: str) -> str:
    """Why ``soup <command>`` cannot take ``adapter``, and what to run instead."""
    merged = f"{adapter.path}-merged"
    return (
        f"{adapter.path} is an MLX adapter (trained with backend: mlx). "
        f"soup {command} takes a PEFT adapter or a full model, and this is neither.\n"
        f"Merge it into its base first, then give soup {command} the merged model:\n"
        f"  soup merge --adapter {adapter.path} --output {merged}\n"
        f"  soup {command} --model {merged} ...\n"
        f"To chat with the adapter as it is: soup chat --model {adapter.path}"
    )


def merge_mlx_adapter(
    adapter: MlxAdapter,
    base: str,
    output_dir: Path,
    dtype: str,
    trust_remote_code: bool = False,
) -> None:
    """Fuse ``adapter`` into ``base`` and save a dequantized model at ``output_dir``.

    These are ``mlx_lm.fuse --dequantize``'s steps. ``mlx_lm.utils.save`` is not
    called: given a repo id it looks the snapshot up again with
    ``local_files_only=True``, and huggingface_hub 1.x refuses the snapshot
    ``mlx_lm.load`` itself downloaded, because that download fetched only the
    files mlx-lm needs. ``mlx_lm.fuse`` fails the same way. The three public
    helpers ``save`` wraps are called directly instead, and the two kinds of
    file ``save`` also copies from the base are copied here: ``generation_config.json``
    (for instruct models it carries the end-of-turn ids, without which a
    transformers ``generate`` on the merged model runs past the turn) and any
    custom modeling ``*.py``.

    ``dtype`` is ``float16``, ``bfloat16`` or ``float32``, and is also written
    to the config so a loader's ``dtype="auto"`` gets what is on disk.
    """
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mlx_lm import load
    from mlx_lm.utils import dequantize_model, save_config, save_model

    model, tokenizer, config = load(
        base,
        adapter_path=str(adapter.path),
        return_config=True,
        tokenizer_config={"trust_remote_code": trust_remote_code},
    )
    fused = [
        (name, module.fuse(dequantize=True))
        for name, module in model.named_modules()
        if hasattr(module, "fuse")
    ]
    if not fused:
        raise ValueError(f"{adapter.path} added no LoRA layers to {base}; nothing to merge")
    model.update_modules(tree_unflatten(fused))
    model = dequantize_model(model)
    model.set_dtype(getattr(mx, dtype))

    config.pop("quantization", None)
    config.pop("quantization_config", None)
    for key in ("torch_dtype", "dtype"):
        if key in config:
            config[key] = dtype

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(output_dir, model, donate_model=True)
    save_config(config, config_path=output_dir / "config.json")
    tokenizer.save_pretrained(str(output_dir))
    for source in _base_files_to_copy(base):
        shutil.copy(source, output_dir / source.name)


def _base_files_to_copy(base: str) -> list[Path]:
    """``generation_config.json`` and ``*.py`` from ``base``, as mlx_lm's own save copies.

    A hub id resolves to its local snapshot. ``allow_patterns`` fetches only
    these files, usually already cached by ``mlx_lm.load``, and without
    ``local_files_only`` there is no snapshot-completeness check to trip.
    """
    root = Path(base)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(base, allow_patterns=["generation_config.json", "*.py"])
        )
    return sorted(
        path for pattern in ("generation_config.json", "*.py") for path in root.glob(pattern)
    )


def exit_if_mlx_adapter(model_path: "Path | str", command: str, console: "Console") -> None:
    """Stop ``soup <command>`` with :func:`unsupported_message` on an MLX adapter."""
    adapter = find_mlx_adapter(model_path)
    if adapter is None:
        return
    import typer

    from soup_cli.utils.terminal import for_terminal

    console.print(f"[red]{for_terminal(unsupported_message(adapter, command))}[/]")
    raise typer.Exit(1)
