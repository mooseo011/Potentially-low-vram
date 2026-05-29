"""Execute setup steps, streaming combined stdout/stderr line by line."""

from __future__ import annotations

import os
import subprocess
from typing import Iterator

from .steps import SetupStep


def run_step(step: SetupStep) -> Iterator[str]:
    """Yield output lines from running ``step``. Final line encodes exit code.

    Manual steps yield their instructions and a non-runnable marker.
    """
    if step.manual or not step.command:
        yield "[manual step — nothing to run automatically]"
        for line in (step.instructions or step.detail).splitlines():
            yield line
        yield "__EXIT__:manual"
        return

    env = os.environ.copy()
    env.update(step.env)
    try:
        proc = subprocess.Popen(
            step.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            shell=step.shell,
        )
    except (OSError, ValueError) as exc:
        yield f"failed to launch: {exc}"
        yield "__EXIT__:127"
        return

    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"__EXIT__:{proc.returncode}"
