"""Small shared helpers with no heavy third-party dependencies."""

from __future__ import annotations

import platform
from pathlib import Path


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def human_bytes(n: float | int | None) -> str:
    """Format a byte count as a human-readable string."""
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024.0
    return f"{n:.1f} EiB"


def config_dir() -> Path:
    """Per-user config directory, respecting platform conventions."""
    if is_windows():
        import os

        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "plvram"
    base = Path.home() / ".config"
    import os

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    return base / "plvram"


def default_offload_dir() -> Path:
    """A sensible default directory for NVMe offload swap files."""
    return config_dir() / "offload"


def module_available(name: str) -> bool:
    """True if a module can be imported without importing it eagerly."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
