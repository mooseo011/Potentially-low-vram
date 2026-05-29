"""Probe the machine: GPUs / VRAM, system RAM, and free disk for offload.

Designed to degrade gracefully. If torch isn't installed we fall back to
``nvidia-smi``; if that's missing too we simply report no GPUs. Nothing here
raises on a CPU-only or fresh box.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from ..utils import module_available


@dataclass
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int | None = None


@dataclass
class SystemMemory:
    ram_total: int
    ram_available: int
    offload_path: str | None = None
    offload_free: int | None = None


@dataclass
class SystemProbe:
    gpus: list[GpuInfo] = field(default_factory=list)
    memory: SystemMemory | None = None
    cuda_available: bool = False
    source: str = "none"  # "torch", "nvidia-smi", or "none"

    @property
    def total_vram(self) -> int:
        return sum(g.total_bytes for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)


def _probe_gpus_torch() -> tuple[list[GpuInfo], bool] | None:
    if not module_available("torch"):
        return None
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    if not torch.cuda.is_available():
        return [], False
    gpus: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            free, total = None, props.total_memory
        gpus.append(
            GpuInfo(index=i, name=props.name, total_bytes=int(total), free_bytes=free)
        )
    return gpus, True


def _probe_gpus_smi() -> list[GpuInfo] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    gpus: list[GpuInfo] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            total = int(float(parts[2])) * 1024 * 1024  # MiB -> bytes
            free = int(float(parts[3])) * 1024 * 1024
        except ValueError:
            continue
        gpus.append(GpuInfo(index=idx, name=parts[1], total_bytes=total, free_bytes=free))
    return gpus


def _probe_memory(offload_path: str | None) -> SystemMemory:
    ram_total = ram_avail = 0
    if module_available("psutil"):
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        ram_total, ram_avail = vm.total, vm.available
    free = None
    if offload_path:
        try:
            from pathlib import Path

            target = Path(offload_path)
            # disk_usage needs an existing dir; walk up to the first that exists.
            while not target.exists() and target != target.parent:
                target = target.parent
            free = shutil.disk_usage(target).free
        except (OSError, ValueError):
            free = None
    return SystemMemory(
        ram_total=ram_total,
        ram_available=ram_avail,
        offload_path=offload_path,
        offload_free=free,
    )


def probe_system(offload_path: str | None = None) -> SystemProbe:
    """Best-effort snapshot of GPUs, RAM and offload disk space."""
    probe = SystemProbe()

    torch_result = _probe_gpus_torch()
    if torch_result is not None:
        probe.gpus, probe.cuda_available = torch_result
        probe.source = "torch"
    else:
        smi = _probe_gpus_smi()
        if smi is not None:
            probe.gpus = smi
            probe.cuda_available = bool(smi)
            probe.source = "nvidia-smi"

    probe.memory = _probe_memory(offload_path)
    return probe
