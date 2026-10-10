# tt_qwen_image_edit — Python reference

Qwen-Image-Edit-2511 + Lightning 4-step LoRA on one Tenstorrent Blackhole p100a.

## Install

```bash
pip install -e .          # inside a tt-metal / ttnn environment; ttnn is not on PyPI
```

## Weights

Not bundled, and **downloaded on first use** into the HF cache. The four ComfyUI repackaged
single files come from four different Hub repos, all Apache-2.0:

| | Repo | File |
|---|---|---|
| DiT | `Comfy-Org/Qwen-Image-Edit_ComfyUI` | `split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` |
| LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` | `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` |
| Text encoder | `Comfy-Org/HunyuanVideo_1.5_repackaged` | `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| VAE | `Comfy-Org/Qwen-Image_ComfyUI` | `split_files/vae/qwen_image_vae.safetensors` |

About 31 GB in total. The tokenizer comes from `Qwen/Qwen2.5-VL-3B-Instruct` (the Qwen2.5
tokenizer is identical across the VL sizes).

`config.paths()` resolves each file in this order, so an existing ComfyUI install is used as-is
and nothing is downloaded twice:

1. an explicit `QWEN_EDIT_DIT` / `QWEN_EDIT_LORA` / `QWEN_EDIT_TE` / `QWEN_EDIT_VAE` path
2. the file under `$QWEN_EDIT_MODELS` in the ComfyUI layout, if it is there (`/comfy` by default)
3. `hf_hub_download` from the table above

`QWEN_EDIT_TOKENIZER` overrides the tokenizer the same way. `config.sha256_of(key)` returns the
checksum each file was validated against.

## API

### `open_device(trace_region_size=0, l1_small_size=98304)`

Opens a 1×1 mesh on the card with `DispatchCoreType.WORKER`. Pair with `close_device(dev)`.

### `QwenEditTT(dev, dit_prec=None, te_prec=None, host_threads=0, n_blocks=None)`

Loads the text encoder and the DiT onto the card and the vision tower and VAE onto the host.
`n_blocks` truncates the DiT for debugging. `.load_s` reports the load time.

### `QwenEditTT.generate(prompt, width, height, seed, images, steps=4) -> (PIL.Image, Timing)`

| Argument | Meaning |
|---|---|
| `prompt` | Edit instruction. English is what the model was trained on. |
| `width`, `height` | 512–1920, multiples of 8, at most 1920×1088 pixels. Non-multiples of 16 are generated at the next multiple of 16 and center-cropped. |
| `seed` | `torch.manual_seed(seed)` then `randn` float32, matching ComfyUI. |
| `images` | 1 to 3 `PIL.Image`. Image 1 is center-cropped and Lanczos-resized to the output size and sets the framing. |
| `steps` | Must be 4. |

`Timing.values` holds `resize_s`, `vision_s`, `reference_encode_s`, `text_encoder_s`, `context_s`, `dit_s`, `vae_decode_s`.

### `QwenEditTT.encode_prompt(prompt, vision, grids) -> torch.Tensor`

The text-encoder half on its own, if you want to cache conditioning across seeds.

### `dram_stats(dev) -> dict`

`total_bytes`, `allocated_bytes`, `free_bytes`, `largest_free_bytes_per_bank`.

## Caching

Per-image vision embeddings and reference latents are keyed on the pixel content, last 6 kept.
Re-using a source image across prompts skips the vision tower and the reference VAE encode
(about 4 s at 1024²). Changing the geometry releases the device buffers and clears the program
cache, so the first image at a new size pays the kernel compilation.

## Modules

| Module | Role |
|---|---|
| `pipeline.py` | `QwenEditTT`, the end-to-end loop |
| `host.py` | ComfyUI-equivalent host maths: prompt template, mrope positions, patching, rope ids, sigmas, euler step, latent normalization |
| `tt_dit.py` | 60-block MMDiT on the card |
| `tt_text_encoder.py` | Qwen2.5-VL language model (28 layers) on the card |
| `vision.py` | Qwen2.5-VL vision tower, host, bf16 storage |
| `vae.py` | Wan 2.1 VAE, host, `conv3d` executed as `conv2d` for a single frame |
| `weights.py` | `ComfyCheckpoint`: fp8 + per-tensor scale loading, LoRA merge |
| `ref_algo.py` | torch reference implementations used by the verification scripts |
| `config.py` | Static shapes, the edit template, weight paths |
| `device.py` | Open / close / DRAM stats / timing |
