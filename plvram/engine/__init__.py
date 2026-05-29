"""Inference engine: accelerator + VRAM detection, DeepSpeed config, runner."""

from .accelerator import AcceleratorInfo, detect_accelerator, resolve_target, KINDS
from .vram import GpuInfo, SystemMemory, SystemProbe, probe_system
from .ds_config import build_ds_config, recommend_offload

__all__ = [
    "AcceleratorInfo",
    "detect_accelerator",
    "resolve_target",
    "KINDS",
    "GpuInfo",
    "SystemMemory",
    "SystemProbe",
    "probe_system",
    "build_ds_config",
    "recommend_offload",
]
