# Potentially-low-vram (`plvram`)

> Great, low VRAM and I can't run my favourite models. **Or can I?**

`plvram` is a cross-platform **terminal UI for running local AI models that are
larger than your VRAM**. It drives **DeepSpeed ZeRO-Inference** with
**DeepNVMe** offloading so model weights that don't fit on the GPU are streamed
from CPU RAM and NVMe storage on demand — letting a modest GPU run models many
times the size of its memory.

It also tackles the *other* hard problem: **getting DeepSpeed (and the
`async_io` op DeepNVMe needs) built on Windows**. The Setup tab detects your
toolchain and runs a guided build so you don't have to fight `vcvars`, CUDA
paths and build flags by hand.

![Setup tab](docs/setup.svg)

## Highlights

- **Run models bigger than VRAM** — ZeRO-3 with `cpu` or `nvme` parameter
  offload. NVMe offload uses DeepNVMe (`async_io`) to stream weights from a
  fast SSD.
- **Multi-vendor GPUs** — targets **CUDA (NVIDIA)**, **ROCm (AMD)** and
  **XPU (Intel)** via DeepSpeed's accelerator abstraction. Detection, VRAM
  probing, and setup plans all adapt to the chosen backend.
- **TUI for everything** — chat with the model, pick models, accelerator &
  offload tiers, inspect your hardware, and run setup, all from the terminal.
- **Windows-friendly setup** — detects MSVC / CUDA / oneAPI / Ninja and builds
  DeepSpeed from source with `DS_BUILD_AIO=1` via `scripts/setup_windows.ps1`.
- **Linux setup** — installs `libaio` + the right PyTorch build + a
  DeepNVMe-enabled DeepSpeed for your accelerator.
- **Degrades gracefully** — the TUI runs (in *demo mode*) even before the
  backend is installed, so you can configure and drive setup first.

## Install

```bash
# Core TUI (works immediately, demo mode until the backend is built)
pip install -e .

# Inference backend (large; or use the in-app Setup tab)
pip install -e ".[backend]"
```

Python 3.9+ is required. A CUDA-capable NVIDIA GPU is needed for real
inference.

## Quick start

```bash
plvram            # launch the TUI
plvram doctor     # print the environment report and exit
plvram plan       # show the setup plan for this machine
plvram setup      # run the platform setup script (builds DeepSpeed + DeepNVMe)
```

In the TUI:

| Key | Tab | What it does |
|-----|-----|--------------|
| F1  | **Chat**   | Send prompts, stream replies. |
| F2  | **Models** | Choose model id, dtype, accelerator, and offload tier. |
| F3  | **System** | GPUs/VRAM, RAM, offload disk + a recommended offload tier. |
| F4  | **Setup**  | Detect the backend environment and run a guided build. |

Typical first run: open **Setup** → review the checks → **Run full plan** to
build DeepSpeed (with DeepNVMe). Then set your model and offload tier on
**Models**, **Load model**, and chat.

## How offloading works

`plvram` builds a DeepSpeed ZeRO-3 inference config from your choices
(`plvram/engine/ds_config.py`):

- **`none`** — keep all weights on the GPU (only if they fit).
- **`cpu`** — offload parameters to pinned CPU RAM (`offload_param.device =
  cpu`). Runs models larger than VRAM but bounded by system RAM.
- **`nvme`** — offload parameters to NVMe (`offload_param.device = nvme`) with
  a DeepNVMe `aio` block (`block_size`, `queue_depth`, `thread_count`, …).
  Runs models larger than RAM by streaming weights off a fast local SSD.

The **System** tab estimates your model's size and recommends a tier based on
available VRAM, RAM, and offload-disk space.

## Accelerators (NVIDIA / AMD / Intel)

DeepSpeed runs on multiple backends through its accelerator abstraction, and
`plvram` targets them all. Pick one on the **Models** tab, via
`plvram run --accelerator …`, or leave it on `auto` (detected from PyTorch).

| Target | GPUs | PyTorch build | Extra runtime | Notes |
|--------|------|---------------|---------------|-------|
| `cuda` | NVIDIA | `cu121` wheel | CUDA Toolkit (`nvcc`) | Fully supported, Linux + Windows. |
| `rocm` | AMD | `rocm6.2` wheel | ROCm runtime (`hipcc`) | Linux first-class; **Windows experimental** — use WSL2. |
| `xpu`  | Intel (Arc / Data Center / Core Ultra) | `xpu` wheel + IPEX | oneAPI Base Toolkit | Linux + Windows. Needs `intel-extension-for-pytorch` (+ `oneccl_bind_pt` for DeepSpeed). |
| `cpu`  | — | `cpu` wheel | — | No GPU; still supports `cpu`/`nvme` offload via DeepNVMe. |

`async_io` (DeepNVMe) is CPU-side I/O, so NVMe offloading works on **every**
accelerator — only the PyTorch build and GPU runtime differ. `plvram doctor`
and `plvram plan` accept `--accelerator {auto,cuda,rocm,xpu,cpu}` and tailor the
checks/plan accordingly (the correct compiler is checked per backend: `nvcc`
for CUDA, `hipcc`/`rocminfo` for ROCm, `icpx` + IPEX for XPU).

```bash
plvram doctor --accelerator xpu      # Intel-specific environment report
plvram plan   --accelerator rocm     # AMD setup plan
plvram run    --accelerator xpu      # launch targeting Intel XPU
```

## Windows setup (the hard part, automated)

DeepSpeed on Windows needs the MSVC C++ toolchain, a matching CUDA toolkit, and
a source build with the right flags. The Setup tab (and `plvram setup`) run
[`scripts/setup_windows.ps1`](scripts/setup_windows.ps1), which:

1. Locates Visual Studio via `vswhere` and imports the MSVC dev environment
   (`vcvars64.bat`) so `cl.exe` is on PATH.
2. Verifies the accelerator runtime for the chosen target (`nvcc` for CUDA,
   `icpx` for Intel XPU; ROCm-on-Windows is flagged as experimental).
3. Sets `DISTUTILS_USE_SDK=1` and (with `-BuildAio`) `DS_BUILD_AIO=1` so the
   `async_io` op for DeepNVMe is compiled in.
4. Builds DeepSpeed from source and installs it.
5. Runs `ds_report` to confirm `async_io` shows `[OKAY]`.

```powershell
# target Intel XPU instead of CUDA
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -BuildAio -Accelerator xpu
```

Prerequisites it can't install silently (Visual Studio Build Tools with the
"Desktop development with C++" workload, and the CUDA Toolkit / oneAPI Base
Toolkit) are surfaced as clearly-labelled steps with instructions.

## Linux setup

[`scripts/setup_linux.sh`](scripts/setup_linux.sh) installs `libaio` dev headers
for your package manager, the right PyTorch build, Transformers/Accelerate, and
a DeepSpeed built with `DS_BUILD_AIO=1`. Choose the accelerator with `ACCEL`:

```bash
ACCEL=cuda bash scripts/setup_linux.sh   # NVIDIA (default)
ACCEL=rocm bash scripts/setup_linux.sh   # AMD
ACCEL=xpu  bash scripts/setup_linux.sh   # Intel (adds IPEX + oneCCL)
```

For ROCm install the ROCm runtime first; for XPU install the oneAPI Base
Toolkit and `source setvars.sh`. See <https://pytorch.org/get-started/locally/>.

## Development

```bash
pip install -e ".[dev]"
pytest                 # 17 tests, no GPU required (TUI runs headless)
textual run --dev plvram.tui.app:PLVRamApp   # hot-reload the UI
```

The heavy ML stack is imported lazily, so the package imports and the test
suite run on a CPU-only box with no backend installed.

## Project layout

```
plvram/
  cli.py              entry point: run / doctor / plan / setup
  config.py           persisted config (model, dtype, offload tiers)
  engine/
    accelerator.py    CUDA / ROCm / XPU detection + target resolution
    vram.py           GPU/RAM/disk probing (torch -> nvidia/rocm/xpu-smi -> none)
    ds_config.py      DeepSpeed ZeRO-3 + DeepNVMe config builder
    inference.py      ZeRO-Inference engine (+ demo mode fallback)
  setup/
    environment.py    per-platform capability detection
    planner.py        report -> ordered setup steps
    linux.py          Linux plan (libaio + DeepSpeed)
    windows.py        Windows plan (MSVC/CUDA detect + guided build)
    runner.py         streaming step executor
  tui/                Textual app + Chat/Models/System/Setup tabs
scripts/
  setup_windows.ps1   guided DeepSpeed + DeepNVMe build for Windows
  setup_linux.sh      libaio + DeepSpeed install for Linux
```

## License

MIT — see [LICENSE](LICENSE).
