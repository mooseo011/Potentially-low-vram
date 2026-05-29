"""Linux setup plan, per accelerator (CUDA / ROCm / XPU / CPU).

DeepSpeed installs from a wheel on Linux; the DeepNVMe prerequisite (``libaio``)
is the same regardless of accelerator because async_io is CPU-side. What differs
is the PyTorch build and a couple of accelerator runtimes.
"""

from __future__ import annotations

import shutil
import sys

from .environment import EnvReport
from .steps import SetupStep, pip_step

# Per-accelerator PyTorch wheel index (override to match your driver/runtime).
TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu121",
    "rocm": "https://download.pytorch.org/whl/rocm6.2",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def _libaio_install_cmd() -> tuple[list[str], str]:
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


def _torch_step(target: str) -> SetupStep:
    index = TORCH_INDEX.get(target, TORCH_INDEX["cuda"])
    return SetupStep(
        title=f"Install PyTorch ({target} build)",
        command=[
            sys.executable, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", index,
        ],
        detail=f"Installs the {target} PyTorch wheel from {index}. Adjust the "
        "index URL to match your installed runtime/driver version.",
    )


def _accelerator_runtime_steps(target: str) -> list[SetupStep]:
    """Accelerator-specific prerequisites that aren't a plain pip install."""
    if target == "rocm":
        return [
            SetupStep(
                title="Install ROCm runtime + toolkit (manual)",
                command=[], manual=True,
                instructions="Install ROCm for your distro (provides hipcc / "
                "rocminfo), e.g. via amdgpu-install, then set ROCM_PATH:\n"
                "  https://rocm.docs.amd.com/projects/install-on-linux/\n"
                "Ensure your user is in the 'render'/'video' groups.",
            )
        ]
    if target == "xpu":
        return [
            SetupStep(
                title="Install Intel oneAPI Base Toolkit (manual)",
                command=[], manual=True,
                instructions="Install the oneAPI Base Toolkit (DPC++ + level-zero) "
                "and source its environment before building/running:\n"
                "  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html\n"
                "  source /opt/intel/oneapi/setvars.sh",
            ),
            pip_step(
                ["intel-extension-for-pytorch", "oneccl_bind_pt"],
                "Install IPEX + oneCCL bindings (Intel XPU)",
                detail="Intel Extension for PyTorch enables XPU; oneccl_bind_pt "
                "lets DeepSpeed select the XPU accelerator.",
            ),
        ]
    return []


def plan(report: EnvReport) -> list[SetupStep]:
    target = getattr(report, "target", "cuda")
    steps: list[SetupStep] = []

    def failing(name: str) -> bool:
        c = report.by_name(name)
        return c is None or c.status == "fail"

    def needs(name: str) -> bool:
        c = report.by_name(name)
        return c is None or c.status in ("fail", "warn")

    # 1. libaio (DeepNVMe prerequisite) — same for every accelerator.
    if needs("libaio"):
        cmd, mgr = _libaio_install_cmd()
        if cmd:
            steps.append(SetupStep(
                title="Install libaio dev headers (DeepNVMe prerequisite)",
                command=["sudo", *cmd],
                detail=f"Detected {mgr}. Needed to build the async_io op used "
                "for NVMe offloading.",
                requires_admin=True, optional=True,
            ))
        else:
            steps.append(SetupStep(
                title="Install libaio dev headers (manual)",
                command=[], manual=True, optional=True,
                instructions="Install your distro's libaio development package "
                "(libaio-dev / libaio-devel). Required for DeepNVMe NVMe offload.",
            ))

    # 2. Accelerator runtime (ROCm / oneAPI), if relevant and not satisfied.
    if target in ("rocm", "xpu") and needs("PyTorch"):
        steps.extend(_accelerator_runtime_steps(target))

    # 3. PyTorch for the target accelerator.
    if failing("PyTorch"):
        steps.append(_torch_step(target))

    # 4. transformers + accelerate.
    missing_hf = [
        pkg for pkg, name in
        (("transformers", "Transformers"), ("accelerate", "Accelerate"))
        if failing(name)
    ]
    if missing_hf:
        steps.append(pip_step(missing_hf, "Install Transformers + Accelerate"))

    # 5. DeepSpeed with async_io (DeepNVMe).
    aio = report.by_name("DeepNVMe (async_io op)")
    if failing("DeepSpeed") or (aio and aio.status in ("fail", "warn")):
        steps.append(SetupStep(
            title="Install DeepSpeed with async_io (DeepNVMe) prebuilt",
            command=[sys.executable, "-m", "pip", "install", "deepspeed"],
            detail="Builds the async_io op at install time so NVMe offloading "
            f"is ready for the {target} backend.",
            env={"DS_BUILD_AIO": "1"},
        ))

    # 6. Verify.
    steps.append(SetupStep(
        title="Verify DeepSpeed ops (ds_report)",
        command=["ds_report"],
        detail="Confirms async_io shows [OKAY] and which accelerator is active.",
        optional=True,
    ))
    return steps
