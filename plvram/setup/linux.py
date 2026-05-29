"""Linux setup plan.

On Linux DeepSpeed installs cleanly from a wheel; the only real prerequisite
for DeepNVMe is the ``libaio`` development headers so the ``async_io`` op can
compile/JIT. We surface the libaio install as a (privileged) optional step and
let pip handle the rest.
"""

from __future__ import annotations

import shutil

from .environment import EnvReport
from .steps import SetupStep, pip_step


def _libaio_install_cmd() -> tuple[list[str], str]:
    """Pick an install command for the detected package manager."""
    if shutil.which("apt-get"):
        return (["apt-get", "install", "-y", "libaio-dev"], "Debian/Ubuntu (apt)")
    if shutil.which("dnf"):
        return (["dnf", "install", "-y", "libaio-devel"], "Fedora/RHEL (dnf)")
    if shutil.which("yum"):
        return (["yum", "install", "-y", "libaio-devel"], "RHEL/CentOS (yum)")
    if shutil.which("pacman"):
        return (["pacman", "-S", "--noconfirm", "libaio"], "Arch (pacman)")
    if shutil.which("zypper"):
        return (["zypper", "install", "-y", "libaio-devel"], "openSUSE (zypper)")
    return ([], "")


def plan(report: EnvReport) -> list[SetupStep]:
    steps: list[SetupStep] = []

    def failing(name: str) -> bool:
        c = report.by_name(name)
        return c is None or c.status == "fail"

    def needs(name: str) -> bool:
        c = report.by_name(name)
        return c is None or c.status in ("fail", "warn")

    # 1. libaio (DeepNVMe prerequisite) — privileged, optional but recommended.
    if needs("libaio"):
        cmd, mgr = _libaio_install_cmd()
        if cmd:
            steps.append(
                SetupStep(
                    title="Install libaio dev headers (DeepNVMe prerequisite)",
                    command=["sudo", *cmd],
                    detail=f"Detected {mgr}. Needed to build the async_io op "
                    "used for NVMe offloading.",
                    requires_admin=True,
                    optional=True,
                )
            )
        else:
            steps.append(
                SetupStep(
                    title="Install libaio dev headers (manual)",
                    command=[],
                    manual=True,
                    optional=True,
                    instructions="Install your distro's libaio development "
                    "package (e.g. libaio-dev / libaio-devel). Required for "
                    "DeepNVMe NVMe offloading.",
                )
            )

    # 2. PyTorch (CUDA build). We can't know the user's CUDA version, so we
    # point at the official index rather than guessing a wheel.
    if failing("PyTorch"):
        steps.append(
            SetupStep(
                title="Install PyTorch (CUDA build)",
                command=[],
                manual=True,
                instructions="Install the CUDA build of PyTorch matching your "
                "driver from https://pytorch.org/get-started/locally/ , e.g.:\n"
                "  pip install torch --index-url "
                "https://download.pytorch.org/whl/cu121",
            )
        )

    # 3. transformers + accelerate.
    missing_hf = [
        pkg
        for pkg, name in (("transformers", "Transformers"), ("accelerate", "Accelerate"))
        if failing(name)
    ]
    if missing_hf:
        steps.append(pip_step(missing_hf, "Install Transformers + Accelerate"))

    # 4. DeepSpeed. Build with async_io enabled so NVMe offload works.
    if failing("DeepSpeed") or report.by_name("DeepNVMe (async_io op)") and report.by_name(
        "DeepNVMe (async_io op)"
    ).status in ("fail", "warn"):
        steps.append(
            SetupStep(
                title="Install DeepSpeed with async_io (DeepNVMe) prebuilt",
                command=["pip", "install", "deepspeed"],
                detail="Builds the async_io op at install time so NVMe "
                "offloading is ready without a JIT step at first run.",
                env={"DS_BUILD_AIO": "1"},
            )
        )

    # 5. Verify.
    steps.append(
        SetupStep(
            title="Verify DeepSpeed ops (ds_report)",
            command=["ds_report"],
            detail="Confirms async_io shows [OKAY] for NVMe offload support.",
            optional=True,
        )
    )
    return steps
