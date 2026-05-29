"""Turn an environment report into an ordered list of setup steps."""

from __future__ import annotations

from ..utils import is_windows
from .environment import EnvReport
from .steps import SetupStep, pip_step

__all__ = ["SetupStep", "pip_step", "plan_setup"]


def plan_setup(report: EnvReport) -> list[SetupStep]:
    # Imported here to avoid a module-load cycle (linux/windows import steps).
    from . import linux as linux_setup
    from . import windows as windows_setup

    if is_windows():
        return windows_setup.plan(report)
    return linux_setup.plan(report)
