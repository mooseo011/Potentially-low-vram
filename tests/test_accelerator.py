from plvram.engine.accelerator import (
    AcceleratorInfo,
    detect_accelerator,
    resolve_target,
)
from plvram.setup import detect_environment, plan_setup
from plvram.setup.environment import EnvReport
from plvram.setup import linux as linux_setup
from plvram.setup import windows as windows_setup


def test_detect_accelerator_runs():
    info = detect_accelerator()
    assert info.kind in ("none", "cpu", "cuda", "rocm", "xpu")


def test_resolve_target_explicit_and_auto():
    assert resolve_target("xpu") == "xpu"
    assert resolve_target("rocm") == "rocm"
    # auto with no GPU detected falls back to cuda
    none_info = AcceleratorInfo("none", "none", False)
    assert resolve_target("auto", none_info) == "cuda"
    # auto honours a detected GPU
    xpu_info = AcceleratorInfo("xpu", "Intel XPU", True)
    assert resolve_target("auto", xpu_info) == "xpu"


def test_detect_environment_sets_target():
    report = detect_environment("xpu")
    assert report.target == "xpu"
    assert report.by_name("Accelerator target") is not None


def _empty_report(target: str) -> EnvReport:
    # No checks -> everything reads as "missing", producing a full plan.
    return EnvReport(os_name="test", target=target)


def test_linux_plan_uses_accelerator_torch_index():
    for target, needle in (
        ("cuda", "cu121"),
        ("rocm", "rocm"),
        ("xpu", "xpu"),
        ("cpu", "cpu"),
    ):
        steps = linux_setup.plan(_empty_report(target))
        torch_steps = [s for s in steps if "PyTorch" in s.title]
        assert torch_steps, f"no torch step for {target}"
        joined = " ".join(torch_steps[0].command)
        assert needle in joined, f"{needle} not in torch step for {target}"


def test_linux_xpu_plan_includes_ipex():
    steps = linux_setup.plan(_empty_report("xpu"))
    titles = " ".join(s.title for s in steps)
    assert "IPEX" in titles or "intel-extension" in titles.lower()


def test_windows_plan_dispatches_on_accelerator():
    cuda_steps = windows_setup.plan(_empty_report("cuda"))
    assert any("CUDA Toolkit" in s.title for s in cuda_steps)

    xpu_steps = windows_setup.plan(_empty_report("xpu"))
    assert any("oneAPI" in s.title for s in xpu_steps)

    rocm_steps = windows_setup.plan(_empty_report("rocm"))
    assert any("ROCm" in s.title for s in rocm_steps)


def test_plan_setup_dispatches_by_platform():
    report = detect_environment("cuda")
    steps = plan_setup(report)
    assert isinstance(steps, list)
