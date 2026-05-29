"""Probe the machine: GPUs / VRAM, system RAM, and free disk for offload.

Accelerator-aware: probes NVIDIA (CUDA), AMD (ROCm/HIP) and Intel (XPU) GPUs.
Designed to degrade gracefully — torch first, then the vendor ``*-smi`` tools,
then nothing. Never raises on a CPU-only or fresh box.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field

from ..utils import module_available
from .accelerator import AcceleratorInfo, detect_accelerator


@dataclass
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int | None = None
    backend: str = ""  # "cuda" | "rocm" | "xpu"


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
    cuda_available: bool = False  # generalised: any accelerator usable
    source: str = "none"  # "torch", "nvidia-smi", "rocm-smi", "xpu-smi", "none"
    accelerator: AcceleratorInfo | None = None

    @property
    def total_vram(self) -> int:
        return sum(g.total_bytes for g in self.gpus)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)


def _probe_torch(accel: AcceleratorInfo) -> tuple[list[GpuInfo], bool] | None:
    """Probe GPUs through torch for the detected accelerator kind."""
    if not module_available("torch"):
        return None
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None

    gpus: list[GpuInfo] = []

    if accel.kind == "xpu" and hasattr(torch, "xpu"):
        try:
            if not torch.xpu.is_available():
                return [], False
            for i in range(torch.xpu.device_count()):
                props = torch.xpu.get_device_properties(i)
                total = int(getattr(props, "total_memory", 0))
                free = None
                if hasattr(torch.xpu, "mem_get_info"):
                    try:
                        free, total = torch.xpu.mem_get_info(i)
                    except Exception:
                        pass
                gpus.append(
                    GpuInfo(i, getattr(props, "name", f"XPU {i}"),
                            int(total), free, backend="xpu")
                )
            return gpus, True
        except Exception:
            return [], False

    # CUDA and ROCm both go through torch.cuda (ROCm = HIP masquerading as cuda).
    backend = "rocm" if accel.kind == "rocm" else "cuda"
    try:
        if not torch.cuda.is_available():
            return [], False
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free, total = torch.cuda.mem_get_info(i)
            except Exception:
                free, total = None, props.total_memory
            gpus.append(
                GpuInfo(i, props.name, int(total), free, backend=backend)
            )
        return gpus, True
    except Exception:
        return None


def _probe_nvidia_smi() -> list[GpuInfo] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    code, out = _run(
        [exe, "--query-gpu=index,name,memory.total,memory.free",
         "--format=csv,noheader,nounits"]
    )
    if code != 0:
        return None
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])
            total = int(float(parts[2])) * 1024 * 1024
            free = int(float(parts[3])) * 1024 * 1024
        except ValueError:
            continue
        gpus.append(GpuInfo(idx, parts[1], total, free, backend="cuda"))
    return gpus or None


def _probe_rocm_smi() -> list[GpuInfo] | None:
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    # rocm-smi's JSON output is the most stable thing to parse across versions.
    code, out = _run([exe, "--showmeminfo", "vram", "--showproductname", "--json"])
    if code != 0 or not out.strip():
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    gpus: list[GpuInfo] = []
    for key, vals in data.items():
        if not key.lower().startswith("card"):
            continue
        try:
            idx = int("".join(ch for ch in key if ch.isdigit()) or "0")
        except ValueError:
            idx = len(gpus)
        name = (
            vals.get("Card series")
            or vals.get("Card model")
            or vals.get("Card SKU")
            or f"AMD GPU {idx}"
        )
        total = _to_int(vals.get("VRAM Total Memory (B)"))
        used = _to_int(vals.get("VRAM Total Used Memory (B)"))
        free = (total - used) if (total and used is not None) else None
        gpus.append(GpuInfo(idx, str(name), total or 0, free, backend="rocm"))
    return gpus or None


def _probe_xpu_smi() -> list[GpuInfo] | None:
    exe = shutil.which("xpu-smi") or shutil.which("xpumcli")
    if not exe:
        return None
    code, out = _run([exe, "discovery", "-j"])
    if code != 0 or not out.strip():
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    devices = data.get("device_list") or data.get("devices") or []
    gpus: list[GpuInfo] = []
    for d in devices:
        idx = _to_int(d.get("device_id")) or len(gpus)
        name = d.get("device_name") or f"Intel GPU {idx}"
        total_mib = _to_int(d.get("memory_physical_size_byte")) or _to_int(
            d.get("memory_physical_size")
        )
        total = total_mib if (total_mib and total_mib > 1 << 30) else (
            (total_mib or 0) * 1024 * 1024
        )
        gpus.append(GpuInfo(int(idx), str(name), int(total), None, backend="xpu"))
    return gpus or None


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, out.stdout or ""
    except (subprocess.SubprocessError, OSError) as exc:
        return 127, str(exc)


def _probe_memory(offload_path: str | None) -> SystemMemory:
    ram_total = ram_avail = 0
    if module_available("psutil"):
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        ram_total, ram_avail = vm.total, vm.available
    free = None
    if offload_path:
        try:
            from pathlib import Path  # noqa: PLC0415

            target = Path(offload_path)
            while not target.exists() and target != target.parent:
                target = target.parent
            free = shutil.disk_usage(target).free
        except (OSError, ValueError):
            free = None
    return SystemMemory(ram_total, ram_avail, offload_path, free)


def probe_system(offload_path: str | None = None) -> SystemProbe:
    """Best-effort snapshot of GPUs, RAM and offload disk space."""
    probe = SystemProbe()
    accel = detect_accelerator()
    probe.accelerator = accel

    torch_result = _probe_torch(accel)
    if torch_result is not None:
        probe.gpus, probe.cuda_available = torch_result
        probe.source = "torch"
    else:
        # No torch — fall back to whichever vendor tool is present.
        for fn, src in (
            (_probe_nvidia_smi, "nvidia-smi"),
            (_probe_rocm_smi, "rocm-smi"),
            (_probe_xpu_smi, "xpu-smi"),
        ):
            found = fn()
            if found:
                probe.gpus = found
                probe.cuda_available = True
                probe.source = src
                break

    probe.memory = _probe_memory(offload_path)
    return probe
