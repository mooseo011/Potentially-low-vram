"""Detect everything the inference backend needs, per platform & accelerator.

Produces an :class:`EnvReport` of named checks for the *target* accelerator
(CUDA / ROCm / XPU). Each check is cheap, never raises, and reports an
``ok``/``warn``/``fail`` status plus a human hint. The planner turns failures
into an ordered list of build steps.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from ..engine.accelerator import (
    AcceleratorInfo,
    LABELS,
    detect_accelerator,
    resolve_target,
)
from ..utils import is_windows, module_available

Status = str  # "ok" | "warn" | "fail"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "!", "fail": "✗"}.get(self.status, "?")


@dataclass
class EnvReport:
    os_name: str
    target: str = "cuda"  # resolved accelerator kind the plan targets
    accelerator: AcceleratorInfo | None = None  # what torch currently reports
    checks: list[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(Check(*args, **kwargs))

    def by_name(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def ready(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        return 127, str(exc)


def _check_python(report: EnvReport) -> None:
    v = platform.python_version()
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) >= (3, 9):
        report.add("Python", "ok", f"Python {v}")
    else:
        report.add("Python", "fail", f"Python {v} is too old",
                   fix="Install Python 3.9+ and recreate the virtualenv.")


def _check_torch(report: EnvReport, target: str) -> None:
    if not module_available("torch"):
        report.add(
            "PyTorch", "fail", "torch is not installed",
            fix=f"Install a PyTorch build for {LABELS.get(target, target)}.",
        )
        return
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:
        report.add("PyTorch", "warn", f"torch import error: {exc}")
        return

    accel = report.accelerator or detect_accelerator()
    if not accel.available:
        report.add(
            "PyTorch", "warn",
            f"torch {torch.__version__} installed but no GPU accelerator usable",
            fix=f"Install/repair the {LABELS.get(target, target)} torch build and "
            "GPU drivers.",
        )
        return

    if accel.kind != target:
        report.add(
            "PyTorch", "warn",
            f"torch {torch.__version__} reports {accel.label}, but target is "
            f"{LABELS.get(target, target)}",
            fix=f"Install a torch build matching the {target} backend, or change "
            "the accelerator target on the Models tab.",
        )
    else:
        report.add("PyTorch", "ok", f"torch {torch.__version__} — {accel.label}")


def _check_toolkit(report: EnvReport, target: str) -> None:
    """Check the compiler/runtime needed to build DeepSpeed ops for `target`."""
    if target == "cuda":
        out = _which_version("nvcc", ["nvcc", "--version"], "release")
        if out:
            report.add("CUDA toolkit (nvcc)", "ok", out)
        else:
            report.add(
                "CUDA toolkit (nvcc)", "warn", "nvcc not found on PATH",
                fix="Install the CUDA Toolkit and set CUDA_HOME/PATH so DeepSpeed "
                "ops can build.",
            )
    elif target == "rocm":
        hipcc = shutil.which("hipcc") or shutil.which("hipconfig")
        rocminfo = shutil.which("rocminfo")
        if hipcc or rocminfo:
            detail = f"hipcc: {hipcc or 'n/a'}; rocminfo: {rocminfo or 'n/a'}"
            report.add("ROCm toolkit (hipcc)", "ok", detail)
        else:
            report.add(
                "ROCm toolkit (hipcc)", "warn", "hipcc/rocminfo not found",
                fix="Install ROCm (e.g. amdgpu-install) and set ROCM_PATH so "
                "DeepSpeed can hipify/build its ops.",
            )
    elif target == "xpu":
        # Intel oneAPI DPC++ compiler (icpx) + level-zero runtime.
        icpx = shutil.which("icpx") or shutil.which("dpcpp")
        if icpx:
            report.add("oneAPI DPC++ (icpx)", "ok", f"found at {icpx}")
        else:
            report.add(
                "oneAPI DPC++ (icpx)", "warn", "icpx/dpcpp not found",
                fix="Install the Intel oneAPI Base Toolkit and run its "
                "setvars script (provides DPC++ + level-zero for XPU).",
            )
        if module_available("intel_extension_for_pytorch"):
            report.add("Intel Extension for PyTorch", "ok", "ipex importable")
        else:
            report.add(
                "Intel Extension for PyTorch", "warn", "ipex not installed",
                fix="pip install intel-extension-for-pytorch (XPU acceleration "
                "+ DeepSpeed XPU support).",
            )


def _which_version(exe: str, cmd: list[str], needle: str) -> str | None:
    if not shutil.which(exe):
        return None
    code, out = _run(cmd)
    if code != 0:
        return None
    for line in out.splitlines():
        if needle in line:
            return line.strip()
    return out.strip().splitlines()[-1] if out.strip() else "found"


def _check_transformers(report: EnvReport) -> None:
    for mod, label in (("transformers", "Transformers"), ("accelerate", "Accelerate")):
        if module_available(mod):
            report.add(label, "ok", f"{mod} installed")
        else:
            report.add(label, "fail", f"{mod} not installed", fix=f"pip install {mod}")


def _check_deepspeed(report: EnvReport, target: str) -> None:
    if not module_available("deepspeed"):
        report.add("DeepSpeed", "fail", "deepspeed not installed",
                   fix="Build/install DeepSpeed (use the Build button).")
        report.add("DeepNVMe (async_io op)", "fail", "deepspeed missing — op unknown",
                   fix="async_io is needed for NVMe offload; built with DeepSpeed.")
        return

    # Confirm DeepSpeed's active accelerator matches the target where possible.
    accel_detail = "deepspeed importable"
    try:
        from deepspeed.accelerator import get_accelerator  # noqa: PLC0415

        name = get_accelerator().device_name()
        accel_detail = f"deepspeed accelerator: {name}"
        ds_kind = {"cuda": "cuda", "xpu": "xpu", "cpu": "cpu"}.get(name, name)
        # ROCm reports as 'cuda' inside deepspeed too, so only warn on real
        # mismatches (e.g. target xpu but deepspeed sees cuda).
        if target == "xpu" and ds_kind != "xpu":
            report.add(
                "DeepSpeed", "warn",
                f"{accel_detail}; target is xpu",
                fix="Install oneccl_bind_pt + ipex so DeepSpeed selects the XPU "
                "accelerator.",
            )
        else:
            report.add("DeepSpeed", "ok", accel_detail)
    except Exception:
        report.add("DeepSpeed", "ok", accel_detail)

    code, out = _run(["ds_report"], timeout=60)
    if code != 0:
        report.add("DeepNVMe (async_io op)", "warn",
                   "could not run ds_report to confirm async_io",
                   fix="Run `ds_report` manually to inspect compiled ops.")
        return
    ok = False
    for line in out.splitlines():
        if "async_io" in line.lower():
            ok = "okay" in line.lower()
            break
    if "async_io" in out.lower() and ok:
        report.add("DeepNVMe (async_io op)", "ok",
                   "async_io available — NVMe offload supported")
    else:
        report.add(
            "DeepNVMe (async_io op)", "warn",
            "async_io op not available (NVMe offload disabled)",
            fix="Install libaio dev headers (Linux) or rebuild DeepSpeed with "
            "DS_BUILD_AIO=1 (Windows).",
        )


def _check_libaio_linux(report: EnvReport) -> None:
    found = False
    code, out = _run(["ldconfig", "-p"])
    if code == 0 and "libaio" in out:
        found = True
    if not found:
        for p in ("/usr/lib/x86_64-linux-gnu", "/usr/lib", "/usr/lib64"):
            try:
                if any(f.startswith("libaio") for f in os.listdir(p)):
                    found = True
                    break
            except OSError:
                continue
    if found:
        report.add("libaio", "ok", "libaio present (DeepNVMe prerequisite)")
    else:
        report.add("libaio", "warn", "libaio not detected",
                   fix="sudo apt-get install libaio-dev   (or libaio-devel via dnf)")


def _check_windows_toolchain(report: EnvReport) -> None:
    cl = shutil.which("cl")
    if cl:
        report.add("MSVC (cl.exe)", "ok", "Visual C++ compiler on PATH")
    else:
        vswhere = os.path.expandvars(
            r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        )
        if os.path.exists(vswhere):
            report.add("MSVC (cl.exe)", "warn",
                       "Visual Studio found but cl.exe not on PATH",
                       fix="Run from an 'x64 Native Tools Command Prompt', or let "
                       "the build script call vcvars64.bat.")
        else:
            report.add("MSVC (cl.exe)", "fail", "Visual C++ Build Tools not found",
                       fix="Install 'Build Tools for Visual Studio' with the "
                       "'Desktop development with C++' workload.")
    if shutil.which("ninja"):
        report.add("Ninja", "ok", "ninja build tool present")
    else:
        report.add("Ninja", "warn", "ninja not found", fix="pip install ninja")
    if shutil.which("git"):
        report.add("git", "ok", "git present")
    else:
        report.add("git", "warn", "git not found",
                   fix="Install Git for Windows (needed to fetch DeepSpeed).")


def detect_environment(accelerator: str = "auto") -> EnvReport:
    """Build an environment report for the resolved target accelerator."""
    detected = detect_accelerator()
    target = resolve_target(accelerator, detected)
    report = EnvReport(os_name=platform.platform(), target=target, accelerator=detected)

    report.add(
        "Accelerator target",
        "ok",
        f"target={LABELS.get(target, target)}  •  detected={detected.label}"
        f"  ({'available' if detected.available else 'not available'})",
    )
    _check_python(report)
    _check_torch(report, target)
    _check_toolkit(report, target)
    _check_transformers(report)
    _check_deepspeed(report, target)
    if is_windows():
        _check_windows_toolchain(report)
    else:
        _check_libaio_linux(report)
    return report
