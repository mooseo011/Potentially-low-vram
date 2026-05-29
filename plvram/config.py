"""User configuration: model selection, offload tiers, generation params.

Persisted as JSON under the per-user config directory. Everything has a
sane default so a fresh install runs without a config file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .utils import config_dir, default_offload_dir

OffloadDevice = Literal["none", "cpu", "nvme"]
Dtype = Literal["fp16", "bf16", "fp32"]


@dataclass
class OffloadConfig:
    """Where ZeRO-3 should park parameters that don't fit in VRAM.

    - ``none``: keep everything on GPU (only works if it fits).
    - ``cpu``:  offload params to pinned CPU RAM.
    - ``nvme``: offload params to NVMe via DeepNVMe async-io (largest models).
    """

    param_device: OffloadDevice = "cpu"
    nvme_path: str = field(default_factory=lambda: str(default_offload_dir()))
    pin_memory: bool = True
    # DeepNVMe (aio) tuning. Defaults are conservative and portable.
    aio_block_size: int = 1048576
    aio_queue_depth: int = 8
    aio_thread_count: int = 1
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    # ZeRO-3 buffering knobs for NVMe offload.
    buffer_count: int = 5
    buffer_size: int = 1_000_000_000
    max_in_cpu: int = 1_000_000_000


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class AppConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: Dtype = "fp16"
    trust_remote_code: bool = False
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # Recently used model ids, most-recent first.
    recent_models: list[str] = field(default_factory=list)

    # ---- persistence -------------------------------------------------

    @staticmethod
    def path() -> Path:
        return config_dir() / "config.json"

    @classmethod
    def load(cls) -> "AppConfig":
        p = cls.path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        offload = OffloadConfig(**(data.get("offload") or {}))
        generation = GenerationConfig(**(data.get("generation") or {}))
        known = {
            "model_id",
            "dtype",
            "trust_remote_code",
            "recent_models",
        }
        kwargs = {k: data[k] for k in known if k in data}
        return cls(offload=offload, generation=generation, **kwargs)

    def save(self) -> Path:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    def remember_model(self, model_id: str, keep: int = 10) -> None:
        model_id = model_id.strip()
        if not model_id:
            return
        self.model_id = model_id
        recents = [m for m in self.recent_models if m != model_id]
        recents.insert(0, model_id)
        self.recent_models = recents[:keep]
