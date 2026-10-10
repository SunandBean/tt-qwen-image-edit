# tt-qwen-image-edit

**What this port adds** — the single-chip port itself, the fp8 + Lightning-LoRA checkpoint loader, and the decision to measure parity against the ComfyUI `qwen_edit` graph the same weight files run on a GPU rather than against diffusers.
**What it builds on** — the [`changh95/qwen-image-2.1-p150`](https://huggingface.co/changh95/qwen-image-2.1-p150) `tt-dit-server` port and container (Apache-2.0).

Qwen-Image-Edit-2511 with the Lightning 4-step LoRA, ported to a single **Tenstorrent Blackhole p100a**.

Model card, demo images and the packaged release: **[sunandbean/qwen-image-edit-2511-p100a](https://huggingface.co/sunandbean/qwen-image-edit-2511-p100a)**

| Source | p100a (this port) | ComfyUI on RTX 5070 Ti |
|:---:|:---:|:---:|
| ![](media/source_1.png) | ![](media/result_1_p100a.png) | ![](media/result_1_comfyui.png) |

Same prompt, size and seed on both devices — 28.9 dB PSNR, PCC 0.985.

On time the GPU wins this one: 1024² with one reference image takes **16.0 s** on the card against
**10.2–12.2 s** in ComfyUI on the 5070 Ti. Nearly half of the card's time is the VAE decode (6.6 s)
against a 9.0 s sampler, so the decode is what would have to improve. Numbers on the model card.

## What it does

The 60-block MMDiT and the Qwen2.5-VL language model run on the card in TTNN; the vision tower, the VAE and
the scheduler run on the host. A 1024×1024 edit from one source image takes **19.1 s** warm.

The reference implementation is not diffusers but the **ComfyUI `qwen_edit` graph** that the same four weight
files run on an RTX 5070 Ti, which is what makes the parity comparison above meaningful.

## Install

```bash
pip install -e .    # inside a tt-metal / ttnn environment; ttnn is not on PyPI
```

Weights are not bundled. Point `QWEN_EDIT_MODELS` at a directory holding the four ComfyUI single files and
`QWEN_EDIT_TOKENIZER` at a Qwen2.5 tokenizer — see [`PYTHON.md`](PYTHON.md).

```python
from PIL import Image
from tt_qwen_image_edit import QwenEditTT, open_device, close_device

dev = open_device()
try:
    model = QwenEditTT(dev)
    image, timing = model.generate("Make it winter", 1024, 1024, seed=0,
                                   images=[Image.open("media/source_1.png")])
    image.save("output.png")
finally:
    close_device(dev)
```

## Layout

| Path | |
|---|---|
| `tt_qwen_image_edit/` | the port |
| `examples/quickstart.py` | runnable end-to-end example |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference |
| `PORTING_NOTES.md` | design decisions, the two problems that had to be solved, full verification tables |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## tt-metal version

Built and measured against tt-metal `v0.80.0-dev20260921-47-g2225ac6b4b`. That commit is **not present in
the public [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)** — the tag `v0.80.0-dev20260921`
exists, but the commit 47 ahead of it is not reachable from any public ref, so the image it came from cannot
be rebuilt from source. The nearest fetchable starting point is that tag.

## Licence

Apache-2.0. The upstream weights ([Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
[lightx2v/Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)) are
Apache-2.0 and are not redistributed here.

Built on the [`changh95/qwen-image-2.1-p150`](https://huggingface.co/changh95/qwen-image-2.1-p150)
`tt-dit-server` container image.
