# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end edit on one p100a.

    python examples/quickstart.py ../media/source_1.png "Make it winter"

Set QWEN_EDIT_MODELS to the ComfyUI models directory that holds the four single files
(see tt_qwen_image_edit/config.py) and QWEN_EDIT_TOKENIZER to a Qwen2.5 tokenizer.
"""
import sys

from PIL import Image

from tt_qwen_image_edit import QwenEditTT, close_device, dram_stats, open_device

source = sys.argv[1] if len(sys.argv) > 1 else "../media/source_1.png"
prompt = sys.argv[2] if len(sys.argv) > 2 else "Make it winter"

dev = open_device()
try:
    model = QwenEditTT(dev)
    print(f"loaded in {model.load_s:.1f} s, device DRAM {dram_stats(dev)['allocated_bytes'] / 2**30:.1f} GiB")

    image, timing = model.generate(prompt, 1024, 1024, seed=0, images=[Image.open(source)])
    image.save("output.png")
    print({k: round(v, 2) for k, v in timing.values.items()})
finally:
    close_device(dev)
