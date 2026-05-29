import pytest

from plvram.config import AppConfig
from plvram.tui.app import PLVRamApp


@pytest.mark.asyncio
async def test_app_boots_and_tabs_present():
    app = PLVRamApp(config=AppConfig())
    async with app.run_test() as pilot:
        from textual.widgets import TabbedContent

        tabs = app.query_one("#tabs", TabbedContent)
        assert tabs.active == "chat"
        # Switch through tabs via the F-key bindings.
        for key, expected in (("f2", "models"), ("f3", "system"), ("f4", "setup")):
            await pilot.press(key)
            assert tabs.active == expected


@pytest.mark.asyncio
async def test_chat_demo_roundtrip():
    app = PLVRamApp(config=AppConfig())
    async with app.run_test() as pilot:
        from plvram.tui.screens.chat import ChatPanel

        panel = app.query_one("#chat-panel", ChatPanel)
        app.query_one("#chat-input").value = "ping"
        await pilot.click("#send")
        # Wait for the generation worker to finish.
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert panel.history, "expected a recorded exchange"
        assert panel.history[-1]["role"] == "assistant"
        assert "demo" in panel.history[-1]["content"].lower()
