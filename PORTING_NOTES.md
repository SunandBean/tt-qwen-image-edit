# Porting notes — Qwen-Image-Edit-2511 to one Blackhole p100a

Translated and condensed from the working notes kept during the port (2026-10-04).

## What the reference is

Not the diffusers pipeline: the reference is the **ComfyUI `qwen_edit` graph** that the same four
weight files run on an RTX 5070 Ti. Matching that graph is what makes the parity check meaningful.

| File (`$QWEN_EDIT_MODELS`) | Contents |
|---|---|
| `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | MMDiT, 60 blocks, dim 3072 (24×128). fp8 e4m3 + per-tensor `weight_scale`; some tensors (block 0 modulation and others) stay bf16 |
| `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | rank 64, 12 linears per block (not the modulation linears). Merged as `W += α/r · up@down` at load |
| `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | Qwen2.5-VL-7B: LM 28 layers (GQA 28/4, qkv bias, mrope [16,24,24]) + vision tower 32 blocks. `scale_weight` style fp8 |
| `vae/qwen_image_vae.safetensors` | Wan 2.1 VAE architecture, 16 latent channels. diffusers `convert_wan_vae_to_diffusers` maps every key onto `AutoencoderKLWan` |

## ComfyUI behaviour reproduced (batch 1, cfg 1)

- **Images.** Image 1 is center-cropped to the output size with PIL Lanczos (through uint8). Every image is
  then reduced twice: to 384² area for the vision tower, and to 1024² area on 8-px multiples for the VAE
  encode → `Wan21.process_in` → reference latent.
- **Prompt.** `"Picture k: <|vision_start|><|image_pad|><|vision_end|>"` per image, then the instruction,
  all placed in the edit system template and tokenized as one string. Each `<|image_pad|>` expands to that
  image's vision tokens (about 200 after the 2×2 merge). The conditioning is the final-norm hidden state
  from after the second `<|im_start|>` + `user\n`.
- **mrope positions.** ComfyUI `Qwen25_7BVLI.forward`: image spans are a (t, h, w) grid and the text after
  them skips ahead by `max(grid)//2`.
- **DiT tokens.** 2×2 patches (odd latents get a circular pad), id = `(index, h - h_len//2, w - w_len//2)`.
  Output latent is index 0, references are 1..k. Text starts at `max(w_len//2, h_len//2)` with all three
  axes equal. Rope is 16/56/56 dims per axis, theta 10000, adjacent pairs.
- **`index_timestep_zero`.** Modulation is computed for two rows, `t` and `0`: output tokens use `t`,
  reference tokens use `0`, text uses `t`. No tanh on the gate. The final `norm_out` applies scale first.
- **Sampling.** `ModelSamplingAuraFlow(shift 3.1, multiplier 1)`, "simple" 4 steps →
  sigmas `[1.0, 0.90291, 0.75610, 0.50820, 0]`. Noise is `torch.manual_seed(seed)` + `randn` float32
  `[1,16,1,h,w]`. Because sigma starts at 1, the input latent contributes 0 — no base-image VAE encode is
  needed. Euler in float32. cfg 1 means the negative branch and CFGNorm are never evaluated.

## Port layout

Text and image and reference streams are kept as separate tensors padded to multiples of 32 (image and
reference share weights but take different modulation rows); only attention concatenates them as
`[text; image; reference]`. The concatenated length is padded to a multiple of 256 via the text padding
to suit the 256-wide SDPA chunking, and padded keys are masked additively. MLPs use
`minimal_matmul(bias, fused GELU_TANH)`. `dit_minimal_matmul_addcmul_fused` is not used because it
requires the residual and the weights to share a tile format (bf16 vs bfp8).

**Modulation linears stay on the host.** Per block there are two 18432×3072 linears (img and txt), about
6.8 B parameters of the 20 B total, and they only ever consume the timestep embedding. The host computes
the 5 rows actually needed (4 sigmas plus 0) once and uploads those — about 0.85 GB on the card instead of
13 GB. Device weights end up at about 14.5 GB (bfp8) for the DiT and about 7 GB (bfp8) for the LM.

**Text encoder.** q/k rows are permuted into adjacent-pair order per head, and the mrope cos/sin tables are
built on the host per prompt. The sequence is right-padded to a multiple of 128, causal.

## Two problems worth recording

### 1. The SDPA kernel breaks this text encoder

Full text-encoder PCC was 0.981, with some text rows at cosine 0.72 from layer 0 onward. Op-by-op
comparison (`device_check.py te_ops`) put RMSNorm, qkv and rope at 0.99999 and SDPA alone at 0.98.

The cause is Qwen2.5's layer-0 k bias, which reaches 173 and drives attention logits to about 5791. The
SDPA kernel holds QKᵀ in bf16, whose spacing at that magnitude is 32, and the softmax ordering changes.
bf16 weights, HiFi4, `packer_l1_acc` off and different chunk sizes all left it unchanged.

Replacing it with an fp32-output matmul for QKᵀ plus an fp32 softmax gives 0.9957 — the same level a CPU
bf16 run reaches against float32 (0.9949).

### 2. Host OOM took down a neighbouring service

The first end-to-end run used im2col `conv3d` in the VAE and float32 vision weights, and the heap left
over after loading was never returned. RSS passed 15 GB and the machine-wide OOM killer chose ComfyUI in
another container (no work lost; it restarted cleanly).

Fixes, all of which keep the output identical:

- single-frame VAE `conv3d` → `conv2d` (PCC 1.0; encode 6.6 → 3.0 s, decode 10.3 → 5.8 s)
- vision tower weights stored bf16
- drop the checkpoint mapping after load and call `malloc_trim`

Resident host memory after load is now about 2.9 GB, and the development container runs under an explicit
memory cap.

## Verification (2026-10-04)

**`check_vs_comfy.py`** — ComfyUI modules on CPU in float32 as the reference:

| Check | Result |
|---|---|
| Token ids and `template_end`, 4 prompts | exact match |
| mrope positions | exact match |
| Small synthetic LM | PCC 1.0 |
| Real LM, 2 layers | PCC 0.999999999999492 |
| Vision tower | PCC 0.9999999999953 (0.99998 stored bf16) |
| DiT, 1 block, 280×216 (odd latent), 2 references | PCC 0.9999985 |

**`device_check.py`** — on the card: DiT 2 blocks 0.99998; DiT 60 blocks, first step (512² + 512²
reference) 0.9997; text encoder 0.9957.

**`comfy_parity.py`** — same input and seed as ComfyUI on the RTX 5070 Ti: winter edit 28.9 dB PSNR
(PCC 0.985); two-image composite 33.4 dB (PCC 0.998).

**`stage_service.py`** — through the model service, holding the card by lease next to the running
production service: 1024² one image 19.2 s, 1360×768 18.7 s, three images 50.7 s, missing image and wrong
step count both 422, startup 168 s, device DRAM 22.7 GB.
