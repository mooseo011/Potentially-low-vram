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

say "Installing Transformers + Accelerate"
"$PYTHON" -m pip install "transformers>=4.40" "accelerate>=0.30"

cat <<'EOF'

NOTE: Install a CUDA build of PyTorch matching your driver, e.g.:
  pip install torch --index-url https://download.pytorch.org/whl/cu121
(See https://pytorch.org/get-started/locally/ )

EOF

say "Installing DeepSpeed (DS_BUILD_AIO=${BUILD_AIO} for NVMe offload)"
DS_BUILD_AIO="${BUILD_AIO}" "$PYTHON" -m pip install deepspeed

say "Verifying ops (ds_report)"
ds_report || warn "ds_report failed to run; check the install manually."

say "Done. Confirm async_io shows [OKAY] above for NVMe offloading."
