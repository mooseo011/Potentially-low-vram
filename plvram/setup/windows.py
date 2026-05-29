"""Windows setup plan — the reason this project exists.

Installing DeepSpeed on Windows is painful: it needs the MSVC C++ toolchain,
a matching CUDA toolkit, the right environment variables, and a source build
because most wheels don't ship the compiled ops. We detect what's present and
emit an ordered, mostly-automated plan. The heavy lifting (entering the MSVC
developer environment via ``vcvars64.bat`` and building DeepSpeed with the
``async_io`` op for DeepNVMe) is delegated to ``scripts/setup_windows.ps1``,
which lives in the repo and is invoked here.

Posture: "detect + guided build" — privileged or system-wide installs are
proposed as steps the user explicitly approves, never run silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .environment import EnvReport
from .steps import SetupStep, pip_step


def _ps1_path() -> Path:
    # repo_root/scripts/setup_windows.ps1 relative to this file.
    return Path(__file__).resolve().parents[2] / "scripts" / "setup_windows.ps1"


def plan(report: EnvReport) -> list[SetupStep]:
    steps: list[SetupStep] = []

    def status(name: str) -> str:
        c = report.by_name(name)
        return c.status if c else "fail"

    # 1. Visual C++ Build Tools — the usual blocker. winget can install it
    # unattended, but it's large and system-wide, so it's optional + flagged.
    msvc = status("MSVC (cl.exe)")
    if msvc == "fail":
        steps.append(
            SetupStep(
                title="Install Visual C++ Build Tools",
                command=[
                    "winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools",
                    "--override",
                    "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools "
                    "--includeRecommended",
                    "-e",
                ],
                detail="Installs the C++ build tools DeepSpeed needs to compile "
                "its CUDA/C++ ops. Large download; requires admin.",
                requires_admin=True,
                optional=True,
            )
        )
        steps.append(
            SetupStep(
                title="(If winget unavailable) install Build Tools manually",
                command=[],
                manual=True,
                optional=True,
                instructions="Download 'Build Tools for Visual Studio 2022' and "
                "select the 'Desktop development with C++' workload:\n"
                "  https://visualstudio.microsoft.com/downloads/",
            )
        )

    # 2. CUDA toolkit (nvcc) for compiling ops.
    if status("CUDA toolkit (nvcc)") != "ok":
        steps.append(
            SetupStep(
                title="Install / locate CUDA Toolkit",
                command=[],
                manual=True,
                instructions="Install a CUDA Toolkit matching your PyTorch build "
                "from https://developer.nvidia.com/cuda-downloads and ensure "
                "CUDA_HOME / PATH point at it (nvcc must be runnable).",
            )
        )

    # 3. Ninja + build helpers via pip (safe, in-venv).
    pip_helpers: list[str] = []
    if status("Ninja") != "ok":
        pip_helpers.append("ninja")
    pip_helpers += ["wheel", "setuptools", "psutil", "py-cpuinfo", "pydantic"]
    steps.append(
        pip_step(
            pip_helpers,
            "Install build helpers (ninja, wheel, deps)",
            detail="In-venv Python build prerequisites for DeepSpeed.",
        )
    )

    # 4. PyTorch CUDA build.
    if status("PyTorch") != "ok":
        steps.append(
            SetupStep(
                title="Install PyTorch (CUDA build)",
                command=[
                    sys.executable, "-m", "pip", "install", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                ],
                detail="CUDA 12.1 wheel; change the index URL to match your "
                "installed CUDA/driver if needed.",
            )
        )

    # 5. transformers + accelerate.
    missing_hf = [
        pkg
        for pkg, name in (("transformers", "Transformers"), ("accelerate", "Accelerate"))
        if status(name) != "ok"
    ]
    if missing_hf:
        steps.append(pip_step(missing_hf, "Install Transformers + Accelerate"))

    # 6. The main event: build DeepSpeed from source with async_io (DeepNVMe).
    # The PowerShell script enters the MSVC dev environment, sets the build
    # flags, and runs the source build — the part that's hard to get right by
    # hand on Windows.
    if status("DeepSpeed") != "ok" or status("DeepNVMe (async_io op)") != "ok":
        ps1 = _ps1_path()
        steps.append(
            SetupStep(
                title="Build & install DeepSpeed with DeepNVMe (async_io)",
                command=[
                    "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(ps1), "-BuildAio",
                ],
                detail="Runs setup_windows.ps1: locates vcvars64.bat, sets "
                "DISTUTILS_USE_SDK=1 and DS_BUILD_AIO=1, then builds DeepSpeed "
                "from source so the async_io op for NVMe offload is compiled in.",
            )
        )

    # 7. Verify.
    steps.append(
        SetupStep(
            title="Verify DeepSpeed ops (ds_report)",
            command=["ds_report"],
            detail="Confirms DeepSpeed imports and which ops compiled. Look for "
            "async_io to enable NVMe offload.",
            optional=True,
        )
    )
    return steps
