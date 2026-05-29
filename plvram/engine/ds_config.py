"""Build the DeepSpeed configuration for ZeRO-Inference with offloading.

The interesting part of this project: turning a user's offload choices into a
valid DeepSpeed ZeRO-3 config, including the DeepNVMe ``aio`` block used to
stream parameters off NVMe. References:
  - ZeRO-Inference: https://www.deepspeed.ai/2022/09/09/zero-inference.html
  - DeepNVMe / async-io: https://www.deepspeed.ai/tutorials/deepnvme/
"""

from __future__ import annotations

from ..config import AppConfig, OffloadConfig


def _dtype_block(dtype: str) -> dict:
    if dtype == "bf16":
        return {"bf16": {"enabled": True}}
    if dtype == "fp32":
        return {}
    return {"fp16": {"enabled": True}}


def build_ds_config(cfg: AppConfig, hidden_size: int | None = None) -> dict:
    """Return a DeepSpeed config dict for ZeRO-3 inference.

    ``hidden_size`` (if known) is used to autotune the ZeRO-3 bucket sizes the
    way the DeepSpeed docs recommend; otherwise we fall back to fixed values.
    """
    off: OffloadConfig = cfg.offload

    zero: dict = {
        "stage": 3,
        # Inference: persist nothing, keep zero live params so everything can
        # be evicted to CPU/NVMe. This is what lets the model exceed VRAM.
        "stage3_param_persistence_threshold": 0,
        "stage3_max_live_parameters": 0,
        "stage3_max_reuse_distance": 0,
    }

    if hidden_size:
        zero["stage3_prefetch_bucket_size"] = max(0, int(0.9 * hidden_size * hidden_size))
        zero["stage3_param_persistence_threshold"] = 10 * hidden_size
    else:
        zero["stage3_prefetch_bucket_size"] = 0

    config: dict = {
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 2_000_000_000,
        "zero_optimization": zero,
        # Inference only — make sure DeepSpeed never tries to build an optimizer.
        "wall_clock_breakdown": False,
    }
    config.update(_dtype_block(cfg.dtype))

    if off.param_device == "cpu":
        zero["offload_param"] = {
            "device": "cpu",
            "pin_memory": off.pin_memory,
        }
    elif off.param_device == "nvme":
        zero["offload_param"] = {
            "device": "nvme",
            "nvme_path": off.nvme_path,
            "pin_memory": off.pin_memory,
            "buffer_count": off.buffer_count,
            "buffer_size": off.buffer_size,
            "max_in_cpu": off.max_in_cpu,
        }
        # DeepNVMe async-io engine config.
        config["aio"] = {
            "block_size": off.aio_block_size,
            "queue_depth": off.aio_queue_depth,
            "thread_count": off.aio_thread_count,
            "single_submit": off.aio_single_submit,
            "overlap_events": off.aio_overlap_events,
        }
    # param_device == "none" -> no offload_param block; everything stays on GPU.

    return config


def estimate_model_bytes(num_params: float, dtype: str) -> int:
    bytes_per = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype, 2)
    return int(num_params * bytes_per)


def recommend_offload(
    total_vram: int,
    ram_available: int,
    model_bytes: int,
    has_nvme_space: bool,
) -> tuple[str, str]:
    """Suggest an offload tier given the hardware budget.

    Returns ``(device, human_reason)``. Heuristic, intentionally conservative:
    leaves headroom for activations and the CUDA/KV-cache footprint.
    """
    # Keep ~20% VRAM head-room for activations + KV cache.
    usable_vram = int(total_vram * 0.8)
    usable_ram = int(ram_available * 0.7)

    if total_vram and model_bytes <= usable_vram:
        return "none", "Model fits in VRAM with head-room; no offload needed."
    if model_bytes <= usable_ram:
        return "cpu", "Too big for VRAM but fits in CPU RAM; offloading params to RAM."
    if has_nvme_space:
        return (
            "nvme",
            "Exceeds both VRAM and RAM; streaming params from NVMe via DeepNVMe.",
        )
    return (
        "nvme",
        "Exceeds VRAM and RAM and no offload disk detected — set an NVMe path "
        "with free space to proceed.",
    )
