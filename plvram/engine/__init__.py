"""Inference engine: VRAM detection, DeepSpeed config, ZeRO-Inference runner."""

from .vram import GpuInfo, SystemMemory, probe_system
from .ds_config import build_ds_config, recommend_offload

__all__ = [
    "GpuInfo",
    "SystemMemory",
    "probe_system",
    "build_ds_config",
    "recommend_offload",
]
