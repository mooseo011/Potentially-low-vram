"""ZeRO-Inference engine wrapping a HuggingFace causal LM.

Heavy imports (torch / transformers / deepspeed) are deferred until
:meth:`InferenceEngine.load` so the TUI starts instantly and stays usable on a
box where the backend hasn't been built yet. When the backend is missing the
engine runs in *demo mode*: it streams back a canned echo so the interface can
be exercised and screenshotted without a GPU.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

from ..config import AppConfig
from ..utils import module_available
from .ds_config import build_ds_config

ProgressFn = Callable[[str], None]


@dataclass
class LoadResult:
    ok: bool
    demo: bool
    message: str
    num_parameters: int | None = None


def backend_available() -> bool:
    """True only if the full inference stack can be imported."""
    return all(module_available(m) for m in ("torch", "transformers", "deepspeed"))


class InferenceEngine:
    """Loads a model under DeepSpeed ZeRO-3 and streams generations."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.demo = not backend_available()
        self._model = None
        self._tokenizer = None
        self._ds_engine = None
        self._loaded_model_id: str | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load(self, progress: ProgressFn | None = None) -> LoadResult:
        say = progress or (lambda _m: None)
        if self.demo:
            say("Backend not installed — running in DEMO mode.")
            self._loaded_model_id = self.cfg.model_id
            return LoadResult(
                ok=True,
                demo=True,
                message=(
                    "Inference backend (torch/transformers/deepspeed) is not "
                    "installed. Open the Setup screen to build it. Chat runs in "
                    "demo mode until then."
                ),
            )
        try:
            return self._load_real(say)
        except Exception as exc:  # pragma: no cover - depends on heavy stack
            self.demo = True
            return LoadResult(
                ok=False,
                demo=True,
                message=f"Failed to load model under DeepSpeed: {exc}",
            )

    def _load_real(self, say: ProgressFn) -> LoadResult:  # pragma: no cover
        import torch  # noqa: PLC0415
        import deepspeed  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from transformers.integrations import HfDeepSpeedConfig  # noqa: PLC0415

        cfg = self.cfg
        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }[cfg.dtype]

        say(f"Loading tokenizer for {cfg.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )

        say("Reading model config to size ZeRO-3 buckets ...")
        from transformers import AutoConfig  # noqa: PLC0415

        hf_cfg = AutoConfig.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        hidden = getattr(hf_cfg, "hidden_size", None)

        ds_config = build_ds_config(cfg, hidden_size=hidden)
        if cfg.offload.param_device == "nvme":
            import os  # noqa: PLC0415

            os.makedirs(cfg.offload.nvme_path, exist_ok=True)
            say(f"NVMe offload directory ready: {cfg.offload.nvme_path}")

        # HfDeepSpeedConfig must exist *before* from_pretrained so the model is
        # sharded straight onto the ZeRO-3 / offload partitions.
        say("Activating ZeRO-3 partitioning (HfDeepSpeedConfig) ...")
        self._dschf = HfDeepSpeedConfig(ds_config)  # kept alive intentionally

        say(f"Loading weights ({cfg.dtype}) with offload — this can take a while ...")
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model.eval()

        say("Initializing DeepSpeed inference engine ...")
        self._ds_engine = deepspeed.initialize(
            model=self._model, config_params=ds_config
        )[0]
        self._ds_engine.module.eval()

        n_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in self._model.parameters()
        )
        self._loaded_model_id = cfg.model_id
        say("Model ready.")
        return LoadResult(
            ok=True,
            demo=False,
            message=f"Loaded {cfg.model_id} under ZeRO-3 "
            f"({cfg.offload.param_device} offload).",
            num_parameters=int(n_params),
        )

    @property
    def loaded_model_id(self) -> str | None:
        return self._loaded_model_id

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str, history: list[dict] | None = None) -> Iterator[str]:
        """Yield generated text incrementally (token chunks)."""
        if self.demo:
            yield from self._demo_generate(prompt)
            return
        yield from self._real_generate(prompt, history or [])

    def _demo_generate(self, prompt: str) -> Iterator[str]:
        import time

        reply = (
            "[demo mode] The inference backend isn't installed yet, so I can't "
            "run a real model. Once DeepSpeed is built via the Setup screen, "
            f"this is where {self.cfg.model_id} would respond to:\n\n"
            f"> {prompt.strip()}\n"
        )
        for word in reply.split(" "):
            yield word + " "
            time.sleep(0.01)

    def _real_generate(  # pragma: no cover - depends on heavy stack
        self, prompt: str, history: list[dict]
    ) -> Iterator[str]:
        import torch  # noqa: PLC0415
        from transformers import TextIteratorStreamer  # noqa: PLC0415

        tok = self._tokenizer
        model = self._ds_engine.module if self._ds_engine else self._model

        messages = list(history) + [{"role": "user", "content": prompt}]
        if getattr(tok, "chat_template", None):
            input_ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = tok(prompt, return_tensors="pt").input_ids

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        gen = self.cfg.generation
        streamer = TextIteratorStreamer(
            tok, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            synced_gpus=True,  # required for ZeRO-3 generation
        )

        errbox: queue.Queue = queue.Queue()

        def _run() -> None:
            try:
                with torch.no_grad():
                    model.generate(**gen_kwargs)
            except Exception as exc:
                errbox.put(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for chunk in streamer:
            yield chunk
        thread.join()
        if not errbox.empty():
            raise errbox.get()
