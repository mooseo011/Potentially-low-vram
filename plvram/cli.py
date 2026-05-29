"""Command-line entry point for plvram.

``plvram``            launch the TUI
``plvram doctor``     print the environment report (no TUI) and exit
``plvram plan``       print the setup plan for this machine and exit
``plvram setup``      run the platform setup script (scripts/setup_*.{sh,ps1})
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import AppConfig
from .utils import is_windows


def _cmd_doctor(_args) -> int:
    from .setup import detect_environment

    report = detect_environment()
    print(f"plvram {__version__} — environment report")
    print(f"OS: {report.os_name}\n")
    width = max(len(c.name) for c in report.checks)
    for c in report.checks:
        print(f"  {c.symbol}  {c.name.ljust(width)}  {c.detail}")
        if c.fix and c.status != "ok":
            print(f"     ↳ {c.fix}")
    print(f"\nReady: {'yes' if report.ready else 'no — see fixes above'}")
    return 0 if report.ready else 1


def _cmd_plan(_args) -> int:
    from .setup import detect_environment, plan_setup

    report = detect_environment()
    steps = plan_setup(report)
    if not steps:
        print("Nothing to do — backend looks ready.")
        return 0
    print(f"Setup plan for {'Windows' if is_windows() else 'Linux'}:\n")
    for i, s in enumerate(steps, 1):
        tags = []
        if s.requires_admin:
            tags.append("admin")
        if s.manual:
            tags.append("manual")
        if s.optional:
            tags.append("optional")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{i}. {s.title}{suffix}")
        if s.detail:
            print(f"     {s.detail}")
        if s.manual and s.instructions:
            for line in s.instructions.splitlines():
                print(f"       {line}")
        elif s.command:
            print(f"     $ {' '.join(s.command)}")
    return 0


def _cmd_setup(args) -> int:
    root = Path(__file__).resolve().parents[1] / "scripts"
    if is_windows():
        script = root / "setup_windows.ps1"
        cmd = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script),
        ]
        if not args.no_aio:
            cmd.append("-BuildAio")
    else:
        script = root / "setup_linux.sh"
        cmd = ["bash", str(script)]
    print(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except OSError as exc:
        print(f"Could not launch setup script: {exc}", file=sys.stderr)
        return 1


def _cmd_run(args) -> int:
    from .tui import run

    cfg = AppConfig.load()
    if args.model:
        cfg.remember_model(args.model)
    if args.offload:
        cfg.offload.param_device = args.offload
    run(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plvram",
        description="Run local AI models larger than your VRAM via DeepSpeed "
        "ZeRO-Inference with DeepNVMe offloading.",
    )
    p.add_argument("--version", action="version", version=f"plvram {__version__}")
    sub = p.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Launch the TUI (default).")
    run_p.add_argument("--model", help="Model id/path to preselect.")
    run_p.add_argument(
        "--offload", choices=["none", "cpu", "nvme"], help="Offload tier override."
    )
    run_p.set_defaults(func=_cmd_run)

    sub.add_parser("doctor", help="Print the environment report and exit.").set_defaults(
        func=_cmd_doctor
    )
    sub.add_parser("plan", help="Print the setup plan and exit.").set_defaults(
        func=_cmd_plan
    )
    setup_p = sub.add_parser("setup", help="Run the platform setup script.")
    setup_p.add_argument(
        "--no-aio", action="store_true", help="Skip building async_io (no NVMe offload)."
    )
    setup_p.set_defaults(func=_cmd_setup)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        # No subcommand -> launch the TUI.
        return _cmd_run(argparse.Namespace(model=None, offload=None))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
