"""Environment detection and platform setup/build orchestration."""

from .environment import EnvReport, Check, detect_environment
from .steps import SetupStep, pip_step
from .planner import plan_setup

__all__ = [
    "EnvReport",
    "Check",
    "detect_environment",
    "SetupStep",
    "pip_step",
    "plan_setup",
]
