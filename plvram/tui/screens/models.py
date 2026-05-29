"""Models panel: pick a model, dtype, and offload configuration."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Label, Select, Static

from ...config import AppConfig


class ModelsPanel(VerticalScroll):
    """Form for model id + dtype + offload tier, persisted to config."""

    def compose(self) -> ComposeResult:
        cfg: AppConfig = self.app.config  # type: ignore[attr-defined]

        yield Static("[b]Model[/b]", classes="section-title")
        yield Label("HuggingFace model id (or local path)")
        yield Input(value=cfg.model_id, id="model-id", placeholder="org/model-name")

        yield Label("Weight dtype")
        yield Select(
            [("fp16", "fp16"), ("bf16", "bf16"), ("fp32", "fp32")],
            value=cfg.dtype,
            id="dtype",
            allow_blank=False,
        )

        yield Label("Accelerator backend")
        yield Select(
            [
                ("auto — detect from PyTorch", "auto"),
                ("cuda — NVIDIA", "cuda"),
                ("rocm — AMD", "rocm"),
                ("xpu — Intel", "xpu"),
                ("cpu — no GPU", "cpu"),
            ],
            value=cfg.accelerator,
            id="accelerator",
            allow_blank=False,
        )

        yield Static("[b]Offload (how to exceed VRAM)[/b]", classes="section-title")
        yield Label("Parameter offload device")
        yield Select(
            [
                ("none — keep all weights on GPU", "none"),
                ("cpu — offload params to system RAM", "cpu"),
                ("nvme — stream params from NVMe (DeepNVMe)", "nvme"),
            ],
            value=cfg.offload.param_device,
            id="offload-device",
            allow_blank=False,
        )

        yield Label("NVMe offload directory (used when device = nvme)")
        yield Input(value=cfg.offload.nvme_path, id="nvme-path")

        yield Static(
            "NVMe offload streams parameters from disk on demand — the only way "
            "to run models far larger than RAM. Point this at a fast local SSD.",
            classes="hint",
        )

        yield Static("[b]Recent models[/b]", classes="section-title")
        yield Static(id="recent", classes="hint")

        with Horizontal():
            yield Button("Save settings", id="save", variant="primary")
            yield Button("Load / reload model", id="load", variant="success")
        yield Static(id="models-status", classes="status-line")

    def on_mount(self) -> None:
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        cfg: AppConfig = self.app.config  # type: ignore[attr-defined]
        recent = cfg.recent_models or ["(none yet)"]
        self.query_one("#recent", Static).update("\n".join(f"• {m}" for m in recent))

    def _apply_to_config(self) -> AppConfig:
        cfg: AppConfig = self.app.config  # type: ignore[attr-defined]
        cfg.model_id = self.query_one("#model-id", Input).value.strip() or cfg.model_id
        cfg.dtype = self.query_one("#dtype", Select).value  # type: ignore[assignment]
        cfg.accelerator = self.query_one(  # type: ignore[assignment]
            "#accelerator", Select
        ).value
        cfg.offload.param_device = self.query_one(  # type: ignore[assignment]
            "#offload-device", Select
        ).value
        cfg.offload.nvme_path = (
            self.query_one("#nvme-path", Input).value.strip()
            or cfg.offload.nvme_path
        )
        return cfg

    def on_button_pressed(self, event: Button.Pressed) -> None:
        cfg = self._apply_to_config()
        if event.button.id == "save":
            cfg.remember_model(cfg.model_id)
            path = cfg.save()
            self._refresh_recent()
            self.query_one("#models-status", Static).update(
                f"[green]Saved[/] settings to {path}"
            )
        elif event.button.id == "load":
            cfg.remember_model(cfg.model_id)
            cfg.save()
            self._refresh_recent()
            self.query_one("#models-status", Static).update(
                "Loading model — switch to the Chat tab to watch progress."
            )
            self.app.load_model()  # type: ignore[attr-defined]
