"""MLX backend utilities — detection + hardware info + batch size estimation.

MLX is Apple's ML framework for Apple Silicon (M1-M4 chips). This module
provides feature detection and helpers so the rest of Soup can opportunistically
enable MLX training paths without hard-depending on the ``mlx`` package.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, replace
from typing import Any, Optional


def detect_mlx() -> bool:
    """Return True if the ``mlx`` package is importable on this machine.

    This does **not** check for Apple Silicon hardware — use ``get_mlx_info``
    for a full detection report.
    """
    try:
        import mlx  # noqa: F401
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def is_apple_silicon() -> bool:
    """Return True if the current machine is an Apple Silicon Mac."""
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def get_mlx_version() -> Optional[str]:
    """Return the MLX version string, or None if not installed.

    The top-level ``mlx`` package does not export ``__version__`` — it lives on
    ``mlx.core`` (and in distribution metadata), so reading ``mlx.__version__``
    always reported "unknown" (#659). Try the canonical surface first, fall
    back to importlib.metadata, and only report "unknown" as a last resort.
    """
    try:
        import mlx.core

        version = getattr(mlx.core, "__version__", None)
        if version:
            return str(version)
    except ImportError:
        pass

    try:
        from importlib.metadata import version as _dist_version

        return _dist_version("mlx")
    except Exception:  # noqa: BLE001 — no package, no metadata, or missing dist
        return None


def get_chip_info() -> dict[str, str]:
    """Best-effort chip detection for Apple Silicon Macs."""
    info: dict[str, str] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    if not is_apple_silicon():
        return info

    try:
        import subprocess  # noqa: S404 — used with list args only
        result = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            info["chip"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return info


def get_unified_memory_bytes() -> Optional[int]:
    """Return total unified memory in bytes, or None if unavailable."""
    if not is_apple_silicon():
        return None
    try:
        import subprocess  # noqa: S404
        result = subprocess.run(  # noqa: S603, S607
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def get_mlx_info() -> dict[str, Any]:
    """Return a full MLX detection report suitable for ``soup doctor``."""
    available = detect_mlx()
    info: dict[str, Any] = {
        "available": available,
        "version": get_mlx_version(),
        "apple_silicon": is_apple_silicon(),
        "chip": get_chip_info(),
        "unified_memory_bytes": get_unified_memory_bytes(),
    }
    return info


def estimate_mlx_batch_size(
    model_params_b: float,
    unified_memory_bytes: int,
    max_length: int,
    quantization: str = "4bit",
) -> int:
    """Rough batch-size estimator for MLX training.

    Uses the same heuristic as ``utils.gpu.estimate_batch_size`` but scaled
    for Apple Silicon unified memory. Returns at least 1.
    """
    bytes_per_param = {"4bit": 0.5, "8bit": 1.0, "none": 2.0}.get(quantization, 2.0)
    model_bytes = model_params_b * 1e9 * bytes_per_param
    activation_budget = max(0, unified_memory_bytes - model_bytes * 1.6)
    # Rough: 2 bytes per token × seq × hidden * 4 (q/k/v/o) + grads
    tokens_cost = max_length * 2.0 * 4096 * 4 * 1.5
    if tokens_cost <= 0:
        return 1
    batch = int(activation_budget / tokens_cost)
    return max(1, min(batch, 32))


#: The ``training.quantization`` values ``backend: mlx`` accepts, as bit widths
#: (``None`` is full precision). The schema refuses every other value for MLX.
_MLX_QUANTIZATION_BITS: dict[str, Optional[int]] = {"4bit": 4, "8bit": 8, "none": None}

#: Group size and mode for quantizing at load: mlx-lm's ``affine`` defaults,
#: which are what ``mlx_lm.convert -q`` used to build the ``mlx-community``
#: ``*-4bit`` / ``*-8bit`` checkpoints.
_LOAD_TIME_GROUP_SIZE = 64
_LOAD_TIME_MODE = "affine"

#: Keys of mlx-lm's ``quantization`` config that are not per-layer overrides.
_QUANTIZATION_DEFAULT_KEYS = frozenset({"group_size", "bits", "mode"})


@dataclass(frozen=True)
class MlxBasePrecision:
    """The precision an MLX run trains its frozen base at, and where it came from.

    ``source`` is ``"full_precision"`` (an unquantized checkpoint trained as
    stored), ``"load_time"`` (an unquantized checkpoint Soup quantized while
    loading it) or ``"checkpoint"`` (a checkpoint that was already quantized).
    """

    source: str
    bits: Optional[int] = None
    group_size: Optional[int] = None
    mode: Optional[str] = None
    per_layer_overrides: bool = False
    #: Set when ``training.quantization`` asked for a precision the run does not
    #: train at, which only happens with an already-quantized checkpoint.
    mismatch: Optional[str] = None

    def describe(self) -> str:
        if self.source == "full_precision":
            return "full precision, as stored in the checkpoint"
        width = f"{self.bits}-bit {self.mode}" if self.bits else str(self.mode)
        if self.source == "load_time":
            return f"{width}, quantized from full precision at load (QLoRA)"
        overrides = " with per-layer overrides" if self.per_layer_overrides else ""
        return f"{width}{overrides}, as quantized in the checkpoint"

    def as_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "per_layer_overrides": self.per_layer_overrides,
        }


def _checkpoint_quantization(model_config: dict) -> Optional[dict]:
    """The quantization ``mlx_lm.load`` applied, from the config it returned.

    ``mlx_lm.utils.load_model`` writes every format it converts (mxfp4, AWQ,
    GPTQ, compressed-tensors) into ``config["quantization"]``. ``bitnet`` is the
    one it applies without writing that key, so it is read from
    ``quantization_config`` instead.
    """
    quantization = model_config.get("quantization")
    if isinstance(quantization, dict):
        return quantization
    legacy = model_config.get("quantization_config")
    if isinstance(legacy, dict) and legacy.get("quant_method"):
        return {"bits": legacy.get("bits"), "mode": legacy["quant_method"]}
    return None


def plan_mlx_base_precision(
    model_config: dict, quantization: str, *, base: str, explicit: bool = True
) -> MlxBasePrecision:
    """Decide the precision the base trains at.

    An unquantized checkpoint is quantized at load to the width
    ``training.quantization`` names, which is mlx-lm's QLoRA path and the same
    meaning the setting has on the transformers backend. A checkpoint that is
    already quantized can be neither restored to full precision nor
    re-quantized without compounding the loss, so it trains at its own
    precision, and ``mismatch`` says so when that differs from the setting.
    That is a warning rather than a refusal because the schema has no value
    for most widths ``mlx-community`` ships (3-, 5- and 6-bit, mixed recipes),
    so a refusal would leave those checkpoints with no config that trains them.
    """
    if quantization not in _MLX_QUANTIZATION_BITS:
        raise ValueError(
            f"training.quantization: {quantization} has no MLX equivalent; "
            f"use one of {sorted(_MLX_QUANTIZATION_BITS)}"
        )
    requested_bits = _MLX_QUANTIZATION_BITS[quantization]

    checkpoint = _checkpoint_quantization(model_config)
    if checkpoint is None:
        if requested_bits is None:
            return MlxBasePrecision(source="full_precision")
        return MlxBasePrecision(
            source="load_time",
            bits=requested_bits,
            group_size=_LOAD_TIME_GROUP_SIZE,
            mode=_LOAD_TIME_MODE,
        )

    precision = MlxBasePrecision(
        source="checkpoint",
        bits=checkpoint.get("bits"),
        group_size=checkpoint.get("group_size"),
        mode=checkpoint.get("mode", "affine"),
        per_layer_overrides=any(key not in _QUANTIZATION_DEFAULT_KEYS for key in checkpoint),
    )
    if precision.bits == requested_bits:
        return precision
    # Only a setting the user wrote is worth a warning: the schema default is
    # ``4bit``, and an unset config pointed at an 8-bit checkpoint asked for nothing.
    if not explicit:
        return precision
    mismatch = (
        f"training.quantization: {quantization} is not applied: {base} is already "
        f"quantized ({precision.describe()}) and trains at that precision."
    )
    # bitnet and other schemes without a bit width have no full-precision sibling.
    if precision.bits is not None:
        wanted = "full precision" if requested_bits is None else f"{requested_bits}-bit"
        mismatch += (
            " On MLX the setting quantizes a full-precision base; point base at the "
            f"full-precision model to train at {wanted}."
        )
    return replace(precision, mismatch=mismatch)


def load_mlx_model(
    model_path: str, quantization: str = "4bit", *, explicit: bool = True,
) -> tuple[Any, Any, MlxBasePrecision]:
    """Load a model with ``mlx_lm`` at the precision ``quantization`` asks for.

    The weights load lazily and are evaluated once, after any quantization,
    which is the order ``mlx_lm.convert -q`` uses: an unquantized base is
    quantized layer by layer rather than first materialised in full.
    """
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import quantize_model

    model, tokenizer, model_config = load(model_path, lazy=True, return_config=True)
    precision = plan_mlx_base_precision(
        model_config, quantization, base=model_path, explicit=explicit
    )
    if precision.source == "load_time":
        model, _ = quantize_model(
            model,
            model_config,
            group_size=precision.group_size,
            bits=precision.bits,
            mode=precision.mode,
        )
    mx.eval(model.parameters())
    return model, tokenizer, precision
