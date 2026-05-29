"""Setup panel: detect the backend environment and run a guided build.

Shows every environment check, derives an ordered plan, and runs the steps one
at a time streaming their output. Privileged or system-wide steps are labelled
so nothing surprising happens without the user clicking through.
"""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, RichLog, Static

from ...setup import detect_environment, plan_setup
from ...setup.runner import run_step
from ...setup.steps import SetupStep
from ...utils import is_windows


class SetupPanel(VerticalScroll):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plan: list[SetupStep] = []
        self._running = False

    def compose(self) -> ComposeResult:
        plat = "Windows" if is_windows() else "Linux"
        yield Static(
            f"[b]Backend setup[/b] — {plat}. DeepSpeed + DeepNVMe (async_io) "
            "powers offloading models larger than VRAM.",
            classes="section-title",
        )
        yield DataTable(id="env-table", zebra_stripes=True)
        yield Static("[b]Plan[/b]", classes="section-title")
        yield Static(id="plan-summary", classes="hint")
        with Horizontal():
            yield Button("Re-detect", id="detect", variant="primary")
            yield Button("Run full plan", id="run-all", variant="success")
            yield Button("Run next step", id="run-next")
        yield Static(id="setup-status", classes="status-line")
        yield RichLog(id="setup-output", wrap=True, highlight=False, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#env-table", DataTable)
        table.add_columns("", "Check", "Detail", "How to fix")
        self.detect()

    # ------------------------------------------------------------------
    def detect(self) -> None:
        report = detect_environment()
        table = self.query_one("#env-table", DataTable)
        table.clear()
        for c in report.checks:
            cls = {"ok": "green", "warn": "yellow", "fail": "red"}.get(c.status, "white")
            table.add_row(
                f"[{cls}]{c.symbol}[/]",
                c.name,
                c.detail,
                c.fix or "—",
            )
        self._plan = plan_setup(report)
        self._render_plan(report.ready)

    def _render_plan(self, ready: bool) -> None:
        if not self._plan:
            summary = "Nothing to do — backend looks ready. ✓"
        else:
            lines = []
            for i, s in enumerate(self._plan, 1):
                tags = []
                if s.requires_admin:
                    tags.append("[red]admin[/]")
                if s.manual:
                    tags.append("[yellow]manual[/]")
                if s.optional:
                    tags.append("[dim]optional[/]")
                tag = (" " + " ".join(tags)) if tags else ""
                lines.append(f"{i}. {s.title}{tag}")
            summary = "\n".join(lines)
        self.query_one("#plan-summary", Static).update(summary)
        state = "ready ✓" if ready else "action needed"
        self.query_one("#setup-status", Static).update(f"Environment: {state}")

    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._running:
            self._log("[yellow]A step is already running…[/]")
            return
        if event.button.id == "detect":
            self.detect()
        elif event.button.id == "run-all":
            if self._plan:
                self._run_plan(list(self._plan))
        elif event.button.id == "run-next":
            if self._plan:
                self._run_plan([self._plan[0]])

    def _log(self, line: str) -> None:
        self.query_one("#setup-output", RichLog).write(line)

    @work(thread=True, exclusive=True, group="setup")
    def _run_plan(self, steps: list[SetupStep]) -> None:
        app = self.app
        self._running = True
        try:
            for step in steps:
                app.call_from_thread(
                    self._log, f"\n[b cyan]==> {step.title}[/]"
                )
                if step.detail:
                    app.call_from_thread(self._log, f"[dim]{step.detail}[/]")
                if step.requires_admin:
                    app.call_from_thread(
                        self._log, "[red]requires administrator privileges[/]"
                    )
                exit_code = None
                for line in run_step(step):
                    if line.startswith("__EXIT__:"):
                        exit_code = line.split(":", 1)[1]
                        continue
                    app.call_from_thread(self._log, line)
                if exit_code == "manual":
                    app.call_from_thread(
                        self._log, "[yellow]→ complete this step yourself, then Re-detect.[/]"
                    )
                elif exit_code not in (None, "0"):
                    app.call_from_thread(
                        self._log,
                        f"[red]✗ step failed (exit {exit_code}); stopping.[/]",
                    )
                    break
                else:
                    app.call_from_thread(self._log, "[green]✓ done[/]")
        finally:
            self._running = False
            app.call_from_thread(self._log, "[dim]— re-detecting environment —[/]")
            app.call_from_thread(self.detect)
