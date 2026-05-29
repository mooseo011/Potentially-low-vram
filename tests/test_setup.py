from plvram.setup import detect_environment, plan_setup
from plvram.setup.runner import run_step
from plvram.setup.steps import SetupStep


def test_detect_environment_runs_without_backend():
    report = detect_environment()
    # Always reports Python; never raises on a bare box.
    names = {c.name for c in report.checks}
    assert "Python" in names
    assert "DeepSpeed" in names


def test_plan_setup_returns_steps_when_backend_missing():
    report = detect_environment()
    steps = plan_setup(report)
    # On a backend-less box there must be something to do.
    assert isinstance(steps, list)
    if not report.ready:
        assert len(steps) >= 1
        assert all(isinstance(s, SetupStep) for s in steps)


def test_run_step_manual_yields_instructions():
    step = SetupStep(
        title="manual", command=[], manual=True, instructions="do a thing"
    )
    out = list(run_step(step))
    assert any("do a thing" in line for line in out)
    assert out[-1] == "__EXIT__:manual"


def test_run_step_executes_command():
    import sys

    step = SetupStep(title="echo", command=[sys.executable, "-c", "print('hi')"])
    out = list(run_step(step))
    assert "hi" in out
    assert out[-1] == "__EXIT__:0"
