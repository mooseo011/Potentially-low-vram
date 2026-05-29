"""Shared setup-step primitives (kept separate to avoid import cycles)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


@dataclass
class SetupStep:
    title: str
    command: list[str]
    detail: str = ""
    optional: bool = False
    requires_admin: bool = False
    shell: bool = False
    # Steps the user must perform themselves (e.g. install Visual Studio); we
    # show instructions instead of a runnable command.
    manual: bool = False
    instructions: str = ""
    env: dict[str, str] = field(default_factory=dict)


def pip_step(packages: list[str], title: str, detail: str = "") -> SetupStep:
    return SetupStep(
        title=title,
        command=[sys.executable, "-m", "pip", "install", *packages],
        detail=detail or f"pip install {' '.join(packages)}",
    )
