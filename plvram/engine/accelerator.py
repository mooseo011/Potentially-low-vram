"""Accelerator detection across CUDA (NVIDIA), ROCm (AMD) and XPU (Intel).

DeepSpeed runs on several backends through its accelerator abstraction
(``deepspeed.accelerator.get_accelerator()``). Before DeepSpeed/torch are even
installed we still want to know what the user is targeting, so this module
sniffs the active accelerator from torch when present and otherwise lets the
user's configured choice drive setup.

torch reports each backend differently:
  - ROCm builds are HIP and masquerade as CUDA: ``torch.version.hip`` is set and
    ``torch.cuda.*`` works against AMD GPUs.
  - Intel GPUs use the XPU runtime: ``torch.xpu.is_available()``.
  - NVIDIA is plain CUDA: ``torch.version.cuda`` set, ``torch.version.hip`` None.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils import module_available

# Selectable accelerator targets. "auto" resolves to a detected/default kind.
KINDS = ("auto", "cuda", "rocm", "xpu", "cpu")
DEVICE_KINDS = ("cuda", "rocm", "xpu")  # actual GPU backends


@dataclass
class AcceleratorInfo:
    kind: str  # "cuda" | "rocm" | "xpu" | "cpu" | "none"
    label: str
    available: bool
    detail: str = ""


# Human labels used across the UI / docs.
LABELS = {
    "cuda": "NVIDIA CUDA",
    "rocm": "AMD ROCm/HIP",
    "xpu": "Intel XPU",
    "cpu": "CPU only",
    "auto": "Auto-detect",
    "none": "none",
}


def detect_accelerator() -> AcceleratorInfo:
    """Best-effort detection of the active accelerator via torch."""
    if not module_available("torch"):
        return AcceleratorInfo("none", "none", False, "torch not installed")
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import edge cases
        return AcceleratorInfo("none", "none", False, f"torch import error: {exc}")

    hip = getattr(torch.version, "hip", None)
    if hip:
        avail = bool(torch.cuda.is_available())
        return AcceleratorInfo(
            "rocm",
            f"AMD ROCm/HIP {hip}",
            avail,
            "AMD GPU via ROCm" if avail else "ROCm torch build but no GPU visible",
        )

    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                return AcceleratorInfo("xpu", "Intel XPU", True, "Intel GPU via XPU")
        except Exception:  # pragma: no cover - driver edge cases
            pass

    cuda = getattr(torch.version, "cuda", None)
    if cuda:
        avail = bool(torch.cuda.is_available())
        return AcceleratorInfo(
            "cuda",
            f"NVIDIA CUDA {cuda}",
            avail,
            "NVIDIA GPU" if avail else "CUDA torch build but no GPU visible",
        )

    return AcceleratorInfo("cpu", "CPU only", False, "CPU-only torch build")


def resolve_target(config_value: str, detected: AcceleratorInfo | None = None) -> str:
    """Resolve a configured accelerator choice to a concrete kind.

    ``auto`` -> detected GPU kind if any, else CUDA (the most common target for
    a fresh setup). An explicit choice is always honoured.
    """
    if config_value and config_value in DEVICE_KINDS + ("cpu",):
        return config_value
    detected = detected or detect_accelerator()
    if detected.kind in DEVICE_KINDS:
        return detected.kind
    return "cuda"
