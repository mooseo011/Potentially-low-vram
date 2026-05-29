"""Windows setup plan, per accelerator — the reason this project exists.

Installing DeepSpeed on Windows is painful: it needs the MSVC C++ toolchain, a
matching accelerator runtime, the right environment variables, and a source
build so the ops (including ``async_io`` for DeepNVMe) are compiled in. We
detect what's present and emit an ordered, mostly-automated plan; the heavy
lifting is delegated to ``scripts/setup_windows.ps1``.

Accelerator notes on Windows:
  - CUDA  — fully supported (NVIDIA + CUDA Toolkit).
  - XPU   — supported for Intel Arc/Core Ultra via torch XPU wheels + IPEX and
            the oneAPI Base Toolkit.
  - ROCm  — experimental: official PyTorch ROCm wheels for Windows are not yet
            generally available, so this path is flagged as manual.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .environment import EnvReport
from .steps import SetupStep, pip_step

TORCH_INDEX = {
    "cuda": "https://download.pytorch.org/whl/cu121",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def _ps1_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "setup_windows.ps1"


def _toolchain_steps(report: EnvReport) -> list[SetupStep]:
    """MSVC build tools — needed to compile DeepSpeed ops on Windows."""
    steps: list[SetupStep] = []
    msvc = report.by_name("MSVC (cl.exe)")
    if msvc and msvc.status == "fail":
        steps.append(SetupStep(
            title="Install Visual C++ Build Tools",
            command=[
                "winget", "install", "--id",
                "Microsoft.VisualStudio.2022.BuildTools", "--override",
                "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools "
                "--includeRecommended", "-e",
            ],
            detail="Installs the C++ build tools DeepSpeed needs to compile its "
            "ops. Large download; requires admin.",
            requires_admin=True, optional=True,
        ))
        steps.append(SetupStep(
            title="(If winget unavailable) install Build Tools manually",
            command=[], manual=True, optional=True,
            instructions="Download 'Build Tools for Visual Studio 2022' and select "
            "the 'Desktop development with C++' workload:\n"
            "  https://visualstudio.microsoft.com/downloads/",
        ))
    return steps


def _accelerator_runtime_steps(target: str, report: EnvReport) -> list[SetupStep]:
    def needs(name: str) -> bool:
        c = report.by_name(name)
        return c is None or c.status != "ok"

    if target == "cuda":
        if needs("CUDA toolkit (nvcc)"):
            return [SetupStep(
                title="Install / locate CUDA Toolkit",
                command=[], manual=True,
                instructions="Install a CUDA Toolkit matching your PyTorch build "
                "from https://developer.nvidia.com/cuda-downloads and ensure "
                "CUDA_HOME / PATH point at it (nvcc must be runnable).",
            )]
        return []
    if target == "xpu":
        steps = [SetupStep(
            title="Install Intel oneAPI Base Toolkit (Windows)",
            command=[], manual=True,
            instructions="Install the oneAPI Base Toolkit (DPC++ + level-zero) and "
            "your Intel Arc/Graphics driver, then run setvars.bat in the build "
            "shell:\n  https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html",
        )]
        if needs("Intel Extension for PyTorch"):
            steps.append(pip_step(
                ["intel-extension-for-pytorch"],
                "Install Intel Extension for PyTorch (XPU)",
                detail="Enables the XPU accelerator for torch + DeepSpeed.",
            ))
        return steps
    if target == "rocm":
        return [SetupStep(
            title="ROCm on Windows is experimental (manual)",
            command=[], manual=True,
            instructions="Official PyTorch ROCm wheels for Windows aren't generally "
            "available yet. Options:\n"
            "  • Install the AMD HIP SDK for Windows and a Windows ROCm torch "
            "build if one matches your GPU, or\n"
            "  • Use WSL2 + the Linux ROCm path (recommended), or\n"
            "  • Switch the accelerator target to CUDA/XPU on the Models tab.",
        )]
    return []


def plan(report: EnvReport) -> list[SetupStep]:
    target = getattr(report, "target", "cuda")
    steps: list[SetupStep] = []

    def status(name: str) -> str:
        c = report.by_name(name)
        return c.status if c else "fail"

    # 1. MSVC build tools (all accelerators need to compile the ops).
    steps.extend(_toolchain_steps(report))

    # 2. Accelerator runtime (CUDA toolkit / oneAPI / ROCm note).
    steps.extend(_accelerator_runtime_steps(target, report))

    # 3. Build helpers (in-venv, safe).
    pip_helpers: list[str] = []
    if status("Ninja") != "ok":
        pip_helpers.append("ninja")
    pip_helpers += ["wheel", "setuptools", "psutil", "py-cpuinfo", "pydantic"]
    steps.append(pip_step(
        pip_helpers, "Install build helpers (ninja, wheel, deps)",
        detail="In-venv Python build prerequisites for DeepSpeed.",
    ))

    # 4. PyTorch for the target accelerator.
    if status("PyTorch") != "ok" and target in TORCH_INDEX:
        steps.append(SetupStep(
            title=f"Install PyTorch ({target} build)",
            command=[
                sys.executable, "-m", "pip", "install", "torch",
                "--index-url", TORCH_INDEX[target],
            ],
            detail=f"{target} wheel from {TORCH_INDEX[target]}; change the index "
            "URL to match your installed runtime if needed.",
        ))

    # 5. transformers + accelerate.
    missing_hf = [
        pkg for pkg, name in
        (("transformers", "Transformers"), ("accelerate", "Accelerate"))
        if status(name) != "ok"
    ]
    if missing_hf:
        steps.append(pip_step(missing_hf, "Install Transformers + Accelerate"))

    # 6. The main event: build DeepSpeed from source with async_io (DeepNVMe).
    if status("DeepSpeed") != "ok" or status("DeepNVMe (async_io op)") != "ok":
        ps1 = _ps1_path()
        steps.append(SetupStep(
            title="Build & install DeepSpeed with DeepNVMe (async_io)",
            command=[
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(ps1), "-BuildAio",
            ],
            detail="Runs setup_windows.ps1: locates vcvars64.bat, sets "
            "DISTUTILS_USE_SDK=1 and DS_BUILD_AIO=1, then builds DeepSpeed from "
            f"source so the async_io op for NVMe offload is compiled in "
            f"(targeting the {target} backend).",
        ))

    # 7. Verify.
    steps.append(SetupStep(
        title="Verify DeepSpeed ops (ds_report)",
        command=["ds_report"],
        detail="Confirms DeepSpeed imports, the active accelerator, and which "
        "ops compiled (look for async_io to enable NVMe offload).",
        optional=True,
    ))
    return steps
