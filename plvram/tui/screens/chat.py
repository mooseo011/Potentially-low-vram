"""Chat panel: send prompts, stream the model's reply."""

from __future__ import annotations

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Input, Markdown, Static


class ChatPanel(VerticalScroll):
    """A streaming chat transcript backed by the shared InferenceEngine."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.history: list[dict] = []
        self._streaming = False
        self._current_md: Markdown | None = None
        self._buffer = ""

    def compose(self) -> ComposeResult:
        yield Static(id="chat-status", classes="status-line")
        with VerticalScroll(id="chat-log"):
            yield Markdown(
                "Welcome to **plvram**. Pick a model on the *Models* tab, then "
                "chat here. If the inference backend isn't built yet, replies "
                "run in demo mode — see the *Setup* tab.",
            )
        with Horizontal(id="chat-input-row"):
            yield Input(
                placeholder="Type a message and press Enter…", id="chat-input"
            )
            yield Button("Send", id="send", variant="primary")

    def on_mount(self) -> None:
        self._update_status()

    def _update_status(self, extra: str = "") -> None:
        app = self.app  # type: ignore[assignment]
        engine = getattr(app, "engine", None)
        cfg = app.config  # type: ignore[attr-defined]
        if engine is None:
            state = "no model loaded"
        elif engine.demo:
            state = "DEMO mode (backend not installed)"
        elif engine.loaded_model_id:
            state = f"loaded: {engine.loaded_model_id} ({cfg.offload.param_device} offload)"
        else:
            state = "model not loaded"
        msg = f"Model: {cfg.model_id}  •  {state}"
        if extra:
            msg += f"  •  {extra}"
        self.query_one("#chat-status", Static).update(msg)

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._submit()

    def _submit(self) -> None:
        if self._streaming:
            return
        box = self.query_one("#chat-input", Input)
        text = box.value.strip()
        if not text:
            return
        box.value = ""
        self._append_message("user", text)
        self._start_generation(text)

    def _append_message(self, role: str, text: str) -> Markdown:
        log = self.query_one("#chat-log", VerticalScroll)
        prefix = "**You:** " if role == "user" else "**Model:** "
        md = Markdown(prefix + text)
        log.mount(md)
        md.scroll_visible()
        return md

    # ------------------------------------------------------------------
    # generation worker
    # ------------------------------------------------------------------
    @work(thread=True, exclusive=True, group="generate")
    def _start_generation(self, prompt: str) -> None:
        app = self.app  # type: ignore[assignment]
        # Lazily ensure an engine exists (without forcing a heavy load if the
        # backend is missing — the engine handles demo mode itself).
        if getattr(app, "engine", None) is None:
            app.call_from_thread(app.ensure_engine)  # type: ignore[attr-defined]

        engine = app.engine  # type: ignore[attr-defined]
        # If a real backend is present but no model is loaded yet, load it now.
        if not engine.demo and engine.loaded_model_id is None:
            app.call_from_thread(self._update_status, "loading model…")
            result = engine.load(
                progress=lambda m: app.call_from_thread(self._update_status, m)
            )
            if not result.ok:
                app.call_from_thread(
                    self._append_message, "model", f"_load failed: {result.message}_"
                )
                app.call_from_thread(self._update_status)
                return

        self._streaming = True
        app.call_from_thread(self._begin_stream)
        try:
            for chunk in engine.generate(prompt, history=self.history):
                self._buffer += chunk
                app.call_from_thread(self._render_stream)
        except Exception as exc:  # pragma: no cover - backend dependent
            self._buffer += f"\n\n_generation error: {exc}_"
            app.call_from_thread(self._render_stream)
        finally:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": self._buffer})
            self._streaming = False
            app.call_from_thread(self._end_stream)

    def _begin_stream(self) -> None:
        self._buffer = ""
        self._current_md = self._append_message("model", "…")
        self._update_status("generating…")

    def _render_stream(self) -> None:
        if self._current_md is not None:
            self._current_md.update("**Model:** " + self._buffer)
            self._current_md.scroll_visible()

    def _end_stream(self) -> None:
        self._render_stream()
        self._current_md = None
        self._update_status()
