"""System panel: GPUs/VRAM, RAM, offload disk, and an offload recommendation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, DataTable, Label, Static

from ...engine.vram import probe_system
from ...engine.ds_config import recommend_offload
from ...utils import human_bytes


class SystemPanel(VerticalScroll):
    """Read-only hardware snapshot with a suggested offload tier."""

    def compose(self) -> ComposeResult:
        yield Static("[b]Hardware[/b]", classes="section-title")
        yield DataTable(id="gpu-table", zebra_stripes=True)
        yield Static(id="mem-info", classes="status-line")
        yield Static("[b]Offload recommendation[/b]", classes="section-title")
        yield Static(id="reco", classes="status-line")
        yield Button("Re-scan hardware", id="rescan", variant="primary")

    def on_mount(self) -> None:
        table = self.query_one("#gpu-table", DataTable)
        table.add_columns("GPU", "Name", "Total VRAM", "Free VRAM")
        self.refresh_probe()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rescan":
            self.refresh_probe()

    def refresh_probe(self) -> None:
        cfg = self.app.config  # type: ignore[attr-defined]
        probe = probe_system(cfg.offload.nvme_path)

        table = self.query_one("#gpu-table", DataTable)
        table.clear()
        if probe.gpus:
            for g in probe.gpus:
                table.add_row(
                    str(g.index),
                    g.name,
                    human_bytes(g.total_bytes),
                    human_bytes(g.free_bytes),
                )
        else:
            table.add_row("-", "No CUDA GPU detected", "-", "-")

        mem = probe.memory
        ram_line = (
            f"RAM: {human_bytes(mem.ram_available)} free / "
            f"{human_bytes(mem.ram_total)} total"
            if mem
            else "RAM: unknown"
        )
        disk_line = (
            f"  •  Offload disk ({mem.offload_path}): "
            f"{human_bytes(mem.offload_free)} free"
            if mem and mem.offload_free is not None
            else ""
        )
        src = f"  •  source: {probe.source}"
        self.query_one("#mem-info", Static).update(ram_line + disk_line + src)

        # Recommendation needs a model-size estimate; if we don't have the model
        # loaded, base it on a rough param count parsed from the model id.
        model_bytes = _guess_model_bytes(cfg.model_id, cfg.dtype)
        device, reason = recommend_offload(
            total_vram=probe.total_vram,
            ram_available=mem.ram_available if mem else 0,
            model_bytes=model_bytes,
            has_nvme_space=bool(mem and mem.offload_free),
        )
        applies = "applied" if cfg.offload.param_device == device else "current: "
        self.query_one("#reco", Static).update(
            f"For ~{human_bytes(model_bytes)} model: suggest [b]{device}[/b] offload.\n"
            f"{reason}\n"
            f"(Config offload: [b]{cfg.offload.param_device}[/b] — "
            f"set it on the Models tab.)"
        )


def _guess_model_bytes(model_id: str, dtype: str) -> int:
    """Very rough size estimate from a '7b' / '13B' style hint in the id."""
    import re

    bytes_per = {"fp16": 2, "bf16": 2, "fp32": 4}.get(dtype, 2)
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id)
    if m:
        billions = float(m.group(1))
        return int(billions * 1e9 * bytes_per)
    # Unknown — assume a small 1B-ish model so we don't over-warn.
    return int(1e9 * bytes_per)
