"""soup chat — interactive chat with a fine-tuned model."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from soup_cli.utils.mlx_adapter import find_mlx_adapter
from soup_cli.utils.terminal import for_terminal

console = Console()


def chat(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to LoRA adapter directory or full model",
    ),
    base_model: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model ID. Auto-detected from adapter_config.json if not set.",
    ),
    device: Optional[str] = typer.Option(
        None,
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum number of tokens to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0 = greedy, 1.0 = creative)",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the conversation",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Source hub for the base model: hf (default) / modelscope / "
            "modelers. Non-HF hubs require the matching SDK; the base model "
            "is snapshotted to a cwd-contained cache before chat starts "
            "(v0.53.10 #152)."
        ),
    ),
):
    """Chat with a fine-tuned model in the terminal."""
    # v0.53.10 #152 — pre-fetch base from a non-HF hub before any path
    # resolution. Local paths and HF repo IDs are passed through unchanged.
    if hub and hub != "hf":
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        try:
            model, base_model = apply_hub_to_cli_model(
                model, base_model, hub, console=console
            )
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2) from exc
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    model_path = Path(model)

    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    # Detect if it's a LoRA adapter or full model
    adapter_config_path = model_path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()

    # An adapter trained with backend: mlx is in mlx-lm's format, not PEFT's,
    # and names its base under `model`. mlx-lm loads it; PEFT cannot.
    mlx_adapter = find_mlx_adapter(model_path) if is_adapter else None
    if mlx_adapter is not None and device not in (None, "mlx"):
        console.print(
            f"[red]{for_terminal(model_path)} is an MLX adapter and runs on "
            f"--device mlx only, not {for_terminal(device)}.[/]"
        )
        raise typer.Exit(1)

    # Resolve base model
    if is_adapter and not base_model:
        base_model = mlx_adapter.base if mlx_adapter else _detect_base_model(adapter_config_path)
        if not base_model:
            console.print(
                "[red]Cannot detect base model from adapter_config.json.[/]\n"
                "Please specify with [bold]--base[/] flag."
            )
            raise typer.Exit(1)

    # Detect device
    if mlx_adapter is not None:
        device = "mlx"
    elif not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    console.print(
        Panel(
            f"Model:  [bold]{model_path}[/]\n"
            + (f"Base:   [bold]{base_model}[/]\n" if is_adapter else "")
            + f"Device: [bold]{device}[/]\n"
            f"Type:   [bold]{'LoRA adapter' if is_adapter else 'Full model'}[/]",
            title="Loading model",
        )
    )

    # Resolve --trust-remote-code (v0.36.0 Part B). Uses the base model id
    # for LoRA adapters since that's what gets executed; otherwise the
    # local model path.
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    probe_target = base_model or str(model_path)
    requires = model_requires_trust_remote_code(str(model_path)) or False
    resolved_trust = resolve_trust_remote_code(
        probe_target,
        requested=trust_remote_code,
        console=console,
        requires_remote_code=requires,
    )

    # Load model + tokenizer
    if mlx_adapter is not None:
        model_obj, tokenizer = _load_mlx_adapter(mlx_adapter.path, base_model)
    else:
        model_obj, tokenizer = _load_model(
            model_path=str(model_path),
            base_model=base_model,
            is_adapter=is_adapter,
            device=device,
            trust_remote_code=resolved_trust,
        )

    console.print("[bold green]Model loaded![/] Type your message. Commands:")
    console.print("  [dim]/quit[/]  - exit chat")
    console.print("  [dim]/clear[/] - reset conversation history")
    console.print("  [dim]/system <text>[/] - set system prompt")
    console.print()

    # Chat loop
    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = console.input("[bold blue]You:[/] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye![/]")
            break
        elif user_input.lower() == "/clear":
            history = []
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
            console.print("[dim]Conversation cleared.[/]\n")
            continue
        elif user_input.lower().startswith("/system "):
            new_system = user_input[8:].strip()
            # Remove old system prompt if exists
            history = [msg for msg in history if msg["role"] != "system"]
            history.insert(0, {"role": "system", "content": new_system})
            console.print(f"[dim]System prompt set: {new_system}[/]\n")
            continue

        # Add user message
        history.append({"role": "user", "content": user_input})

        # Generate response
        if mlx_adapter is not None:
            response = _generate_mlx(
                model_obj, tokenizer, history,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            response = _generate(
                model_obj, tokenizer, history,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device,
            )

        history.append({"role": "assistant", "content": response})
        console.print(f"[bold green]Assistant:[/] {response}\n")


def _detect_base_model(adapter_config_path: Path) -> Optional[str]:
    """Read base_model_name_or_path from adapter_config.json."""
    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path")
    except (json.JSONDecodeError, OSError):
        return None


def _load_model(
    model_path: str,
    base_model: Optional[str],
    is_adapter: bool,
    device: str,
    trust_remote_code: bool = False,
):
    """Load model and tokenizer. Supports LoRA adapters and full models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.print("[dim]Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        console.print(f"[dim]Loading base model: {base_model}...[/]")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        console.print(f"[dim]Loading LoRA adapter: {model_path}...[/]")
        model_obj = PeftModel.from_pretrained(base, model_path)
    else:
        console.print(f"[dim]Loading model: {model_path}...[/]")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _load_mlx_adapter(adapter_path: Path, base_model: str):
    """Load ``base_model`` with the MLX adapter at ``adapter_path`` applied."""
    try:
        from mlx_lm import load
    except ImportError as exc:
        console.print(
            f"[red]{for_terminal(adapter_path)} is an MLX adapter; chatting with it "
            "needs mlx-lm on Apple Silicon.[/]\n"
            'Run: [bold]pip install "soup-cli\\[mlx]"[/]'
        )
        raise typer.Exit(1) from exc

    console.print(f"[dim]Loading {for_terminal(base_model)} with MLX adapter...[/]")
    return load(base_model, adapter_path=str(adapter_path))


def _generate_mlx(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response through mlx-lm, sampling as :func:`_generate` does."""
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # temp 0 is greedy in both; top_p 0.9 matches the transformers path when sampling.
    sampler = make_sampler(temp=temperature, top_p=0.9 if temperature > 0 else 0.0)
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
    ).strip()


def _generate(
    model,
    tokenizer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
) -> str:
    """Generate a response from the model given message history."""
    import torch

    from soup_cli.utils.vllm import encode_chat_prompt

    inputs = encode_chat_prompt(
        messages, tokenizer, fallback_on_error=False, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        outputs = model.generate(**gen_kwargs)

    # Decode only new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response
