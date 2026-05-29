"""The plvram Textual application shell."""

from __future__ import annotations

from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from ..config import AppConfig
from ..engine.inference import InferenceEngine
from .screens import ChatPanel, ModelsPanel, SetupPanel, SystemPanel


class PLVRamApp(App):
    """Run local models larger than your VRAM, from the terminal."""

    CSS_PATH = "styles.tcss"
    TITLE = "plvram"
    SUB_TITLE = "run models larger than your VRAM"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("f1", "show_tab('chat')", "Chat"),
        ("f2", "show_tab('models')", "Models"),
        ("f3", "show_tab('system')", "System"),
        ("f4", "show_tab('setup')", "Setup"),
    ]

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config: AppConfig = config or AppConfig.load()
        self.engine: InferenceEngine | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="chat", id="tabs"):
            with TabPane("Chat", id="chat"):
                yield ChatPanel(id="chat-panel")
            with TabPane("Models", id="models"):
                yield ModelsPanel(id="models-panel", classes="panel")
            with TabPane("System", id="system"):
                yield SystemPanel(id="system-panel", classes="panel")
            with TabPane("Setup", id="setup"):
                yield SetupPanel(id="setup-panel", classes="panel")
        yield Footer()

    # ------------------------------------------------------------------
    def action_show_tab(self, tab: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab

    # ---- shared engine lifecycle ------------------------------------
    def ensure_engine(self) -> InferenceEngine:
        """Create the engine if needed (cheap; no model load)."""
        if self.engine is None:
            self.engine = InferenceEngine(self.config)
        return self.engine

    def load_model(self) -> None:
        """(Re)build the engine for the current config and load weights."""
        self.engine = InferenceEngine(self.config)
        self._load_worker()

    @work(thread=True, exclusive=True, group="load")
    def _load_worker(self) -> None:
        engine = self.engine
        if engine is None:
            return

        def progress(msg: str) -> None:
            try:
                panel = self.query_one("#chat-panel", ChatPanel)
                self.call_from_thread(panel._update_status, msg)
            except Exception:
                pass

        engine.load(progress=progress)
        try:
            panel = self.query_one("#chat-panel", ChatPanel)
            self.call_from_thread(panel._update_status)
        except Exception:
            pass


def run(config: AppConfig | None = None) -> None:
    PLVRamApp(config=config).run()
