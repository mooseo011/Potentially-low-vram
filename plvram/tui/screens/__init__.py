"""Tab panels for the plvram TUI."""

from .chat import ChatPanel
from .models import ModelsPanel
from .system import SystemPanel
from .setup_panel import SetupPanel

__all__ = ["ChatPanel", "ModelsPanel", "SystemPanel", "SetupPanel"]
