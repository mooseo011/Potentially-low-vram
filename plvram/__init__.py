"""plvram — run local AI models larger than your VRAM.

A cross-platform terminal UI that drives DeepSpeed ZeRO-Inference with
DeepNVMe (async-io / GDS) offloading so model weights that don't fit in GPU
memory are streamed from CPU RAM and NVMe storage on demand.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
