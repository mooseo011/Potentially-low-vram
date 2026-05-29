"""Detect everything the inference backend needs, per platform.

Produces an :class:`EnvReport` of named checks. Each check is cheap, never
raises, and reports an ``ok``/``warn``/``fail`` status plus a human hint about
how to fix it. The Setup screen renders these and the planner turns the
failures into an ordered list of build steps.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

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
    checks: list[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(Check(*args, **kwargs))

    def by_name(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def ready(self) -> bool:
        """True when nothing is in a hard-fail state."""
        return all(c.status != "fail" for c in self.checks)


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        return 127, str(exc)


def _check_python(report: EnvReport) -> None:
    v = platform.python_version()
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) >= (3, 9):
        report.add("Python", "ok", f"Python {v}")
    else:
        report.add(
            "Python", "fail", f"Python {v} is too old",
            fix="Install Python 3.9+ and recreate the virtualenv.",
        )


def _nvcc_version() -> str | None:
    exe = shutil.which("nvcc")
    if not exe:
        return None
    code, out = _run([exe, "--version"])
    if code != 0:
        return None
    for line in out.splitlines():
        if "release" in line:
            return line.strip()
    return out.strip().splitlines()[-1] if out.strip() else "found"


def _check_torch_and_cuda(report: EnvReport) -> None:
    if not module_available("torch"):
        report.add(
            "PyTorch", "fail", "torch is not installed",
            fix="Install a CUDA build of PyTorch matching your driver.",
        )
    else:
        try:
            import torch  # noqa: PLC0415

            cuda_ok = torch.cuda.is_available()
            cu = getattr(torch.version, "cuda", None)
            if cuda_ok:
                report.add(
                    "PyTorch", "ok",
                    f"torch {torch.__version__} (CUDA {cu}, "
                    f"{torch.cuda.device_count()} GPU(s))",
                )
            else:
                report.add(
                    "PyTorch", "warn",
                    f"torch {torch.__version__} installed but CUDA unavailable",
                    fix="Install a CUDA-enabled torch build / check GPU drivers.",
                )
        except Exception as exc:
            report.add("PyTorch", "warn", f"torch import error: {exc}")

    nvcc = _nvcc_version()
    if nvcc:
        report.add("CUDA toolkit (nvcc)", "ok", nvcc)
    else:
        report.add(
            "CUDA toolkit (nvcc)", "warn",
            "nvcc not found on PATH",
            fix="Needed to JIT-build DeepSpeed ops. Install CUDA Toolkit "
            "and add it to PATH (set CUDA_HOME).",
        )


def _check_transformers(report: EnvReport) -> None:
    for mod, label in (("transformers", "Transformers"), ("accelerate", "Accelerate")):
        if module_available(mod):
            report.add(label, "ok", f"{mod} installed")
        else:
            report.add(label, "fail", f"{mod} not installed",
                       fix=f"pip install {mod}")


def _check_deepspeed(report: EnvReport) -> None:
    if not module_available("deepspeed"):
        report.add(
            "DeepSpeed", "fail", "deepspeed not installed",
            fix="Build/install DeepSpeed (use the Build button).",
        )
        report.add(
            "DeepNVMe (async_io op)", "fail", "deepspeed missing — op unknown",
            fix="async_io is needed for NVMe offload; built with DeepSpeed.",
        )
        return

    report.add("DeepSpeed", "ok", "deepspeed importable")
    # ds_report tells us which ops are compiled / JIT-loadable.
    code, out = _run(["ds_report"], timeout=60)
    if code != 0:
        report.add(
            "DeepNVMe (async_io op)", "warn",
            "could not run ds_report to confirm async_io",
            fix="Run `ds_report` manually to inspect compiled ops.",
        )
        return
    lowered = out.lower()
    if "async_io" in lowered:
        # ds_report prints [OKAY]/[NO] next to ops; check the async_io line.
        ok = False
        for line in out.splitlines():
            if "async_io" in line.lower():
                ok = "[okay]" in line.lower() or "okay" in line.lower()
                break
        if ok:
            report.add("DeepNVMe (async_io op)", "ok",
                       "async_io available — NVMe offload supported")
        else:
            report.add(
                "DeepNVMe (async_io op)", "warn",
                "async_io op not compatible (NVMe offload unavailable)",
                fix="Install libaio dev headers (Linux) or rebuild DeepSpeed "
                "with DS_BUILD_AIO=1 (Windows).",
            )
    else:
        report.add("DeepNVMe (async_io op)", "warn", "async_io status unknown")


def _check_libaio_linux(report: EnvReport) -> None:
    # async_io links against libaio. Probe common locations / ldconfig.
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
        report.add(
            "libaio", "warn", "libaio not detected",
            fix="sudo apt-get install libaio-dev   (or libaio-devel via dnf)",
        )


def _check_windows_toolchain(report: EnvReport) -> None:
    # DeepSpeed on Windows needs the MSVC build tools (cl.exe) + Ninja.
    cl = shutil.which("cl")
    if cl:
        report.add("MSVC (cl.exe)", "ok", "Visual C++ compiler on PATH")
    else:
        vswhere = os.path.expandvars(
            r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        )
        if os.path.exists(vswhere):
            report.add(
                "MSVC (cl.exe)", "warn",
                "Visual Studio found but cl.exe not on PATH",
                fix="Run the build from a 'x64 Native Tools Command Prompt', "
                "or let the build script call vcvars64.bat.",
            )
        else:
            report.add(
                "MSVC (cl.exe)", "fail",
                "Visual C++ Build Tools not found",
                fix="Install 'Build Tools for Visual Studio' with the "
                "'Desktop development with C++' workload.",
            )
    if shutil.which("ninja"):
        report.add("Ninja", "ok", "ninja build tool present")
    else:
        report.add("Ninja", "warn", "ninja not found", fix="pip install ninja")

    if shutil.which("git"):
        report.add("git", "ok", "git present")
    else:
        report.add("git", "warn", "git not found",
                   fix="Install Git for Windows (needed to fetch DeepSpeed).")


def detect_environment() -> EnvReport:
    report = EnvReport(os_name=platform.platform())
    _check_python(report)
    _check_torch_and_cuda(report)
    _check_transformers(report)
    _check_deepspeed(report)
    if is_windows():
        _check_windows_toolchain(report)
    else:
        _check_libaio_linux(report)
    return report
