from plvram.config import AppConfig
from plvram.engine.ds_config import (
    build_ds_config,
    estimate_model_bytes,
    recommend_offload,
)


def test_cpu_offload_config():
    cfg = AppConfig()
    cfg.offload.param_device = "cpu"
    ds = build_ds_config(cfg, hidden_size=4096)
    zero = ds["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["offload_param"]["device"] == "cpu"
    assert "aio" not in ds
    assert ds["fp16"]["enabled"] is True


def test_nvme_offload_includes_aio_block():
    cfg = AppConfig()
    cfg.offload.param_device = "nvme"
    cfg.offload.nvme_path = "/tmp/offload"
    ds = build_ds_config(cfg)
    zero = ds["zero_optimization"]
    assert zero["offload_param"]["device"] == "nvme"
    assert zero["offload_param"]["nvme_path"] == "/tmp/offload"
    # DeepNVMe aio engine must be present for NVMe offload.
    assert "aio" in ds
    assert ds["aio"]["queue_depth"] == cfg.offload.aio_queue_depth


def test_none_offload_keeps_weights_on_gpu():
    cfg = AppConfig()
    cfg.offload.param_device = "none"
    ds = build_ds_config(cfg)
    assert "offload_param" not in ds["zero_optimization"]


def test_bf16_dtype_block():
    cfg = AppConfig()
    cfg.dtype = "bf16"
    ds = build_ds_config(cfg)
    assert ds["bf16"]["enabled"] is True
    assert "fp16" not in ds


def test_estimate_model_bytes():
    assert estimate_model_bytes(7e9, "fp16") == int(7e9 * 2)
    assert estimate_model_bytes(7e9, "fp32") == int(7e9 * 4)


def test_recommend_offload_tiers():
    gb = 1024**3
    # Fits in VRAM
    dev, _ = recommend_offload(24 * gb, 32 * gb, 10 * gb, True)
    assert dev == "none"
    # Too big for VRAM, fits RAM
    dev, _ = recommend_offload(8 * gb, 64 * gb, 20 * gb, True)
    assert dev == "cpu"
    # Too big for both -> nvme
    dev, _ = recommend_offload(8 * gb, 16 * gb, 80 * gb, True)
    assert dev == "nvme"
