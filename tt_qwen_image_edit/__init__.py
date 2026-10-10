# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 + Lightning 4-step LoRA on one Tenstorrent Blackhole p100a.

The DiT (60 blocks) and the Qwen2.5-VL language model run on the card in TTNN; the vision
tower, the VAE and the scheduler run on the host. The reference behaviour is the ComfyUI
`qwen_edit` graph: ModelSamplingAuraFlow shift 3.1, euler/simple, 4 steps, cfg 1,
reference method `index_timestep_zero`.
"""
from .device import Timing, close_device, dram_stats, open_device
from .pipeline import QwenEditTT

__all__ = ["QwenEditTT", "Timing", "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"
