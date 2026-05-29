#!/usr/bin/env bash
# Install the plvram inference backend on Linux, including DeepNVMe (async_io).
#
# DeepSpeed installs from a wheel on Linux; the only real prerequisite for NVMe
# offloading is the libaio development headers so the async_io op can build.
set -euo pipefail

say() { printf '\n==> %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*" >&2; }

PYTHON="${PYTHON:-python3}"
BUILD_AIO="${BUILD_AIO:-1}"
# Accelerator: cuda (NVIDIA) | rocm (AMD) | xpu (Intel) | cpu
ACCEL="${ACCEL:-cuda}"

case "$ACCEL" in
    cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    rocm) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2" ;;
    xpu)  TORCH_INDEX="https://download.pytorch.org/whl/xpu" ;;
    cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *)    warn "Unknown ACCEL='$ACCEL'; defaulting to cuda"; TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
esac
say "Accelerator target: $ACCEL  (torch index: $TORCH_INDEX)"

say "Installing libaio dev headers (DeepNVMe prerequisite)"
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y libaio-dev
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y libaio-devel
elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y libaio-devel
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --noconfirm libaio
elif command -v zypper >/dev/null 2>&1; then
    sudo zypper install -y libaio-devel
else
    warn "Unknown package manager — install libaio development headers manually."
fi

say "Upgrading pip build tooling"
"$PYTHON" -m pip install --upgrade pip wheel setuptools ninja

if [ "$ACCEL" = "rocm" ]; then
    warn "ROCm: ensure the ROCm runtime/toolkit is installed (amdgpu-install) and ROCM_PATH is set."
elif [ "$ACCEL" = "xpu" ]; then
    warn "XPU: install the Intel oneAPI Base Toolkit and 'source /opt/intel/oneapi/setvars.sh' first."
fi

say "Installing PyTorch ($ACCEL) + Transformers + Accelerate"
"$PYTHON" -m pip install torch --index-url "$TORCH_INDEX" || \
    warn "torch install failed — adjust TORCH_INDEX to match your runtime version."
"$PYTHON" -m pip install "transformers>=4.40" "accelerate>=0.30"

if [ "$ACCEL" = "xpu" ]; then
    say "Installing Intel Extension for PyTorch + oneCCL bindings (XPU)"
    "$PYTHON" -m pip install intel-extension-for-pytorch oneccl_bind_pt || \
        warn "IPEX/oneccl install failed — see Intel's XPU install guide."
fi

say "Installing DeepSpeed (DS_BUILD_AIO=${BUILD_AIO} for NVMe offload)"
DS_BUILD_AIO="${BUILD_AIO}" "$PYTHON" -m pip install deepspeed

say "Verifying ops (ds_report)"
ds_report || warn "ds_report failed to run; check the install manually."

say "Done. Confirm async_io shows [OKAY] above for NVMe offloading."
