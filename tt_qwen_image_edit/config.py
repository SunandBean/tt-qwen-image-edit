# SPDX-License-Identifier: Apache-2.0
"""Static configuration of Qwen-Image-Edit-2511 (Apache-2.0) with the Lightning 4-step LoRA (Apache-2.0), as the
ComfyUI repackaged single files that a ComfyUI GPU setup already runs:

  diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors   60-block MMDiT, fp8 e4m3 + per-tensor scale
  loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors   rank-64 LoRA on 12 linears per block
  text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors          Qwen2.5-VL-7B (LM + vision tower), scaled fp8
  vae/qwen_image_vae.safetensors                               Wan 2.1 VAE architecture, 16 latent channels

The reference behaviour is the ComfyUI `qwen_edit` graph this port was matched against: ModelSamplingAuraFlow
shift 3.1, Lightning LoRA strength 1, TextEncodeQwenImageEditPlus (up to 3 images), reference method
index_timestep_zero, KSampler euler / simple, 4 steps, cfg 1 (the negative prompt is never evaluated)."""
from __future__ import annotations

import os
from dataclasses import dataclass

TILE = 32


@dataclass(frozen=True)
class DiTConfig:
    dim: int = 3072  # 24 heads x 128
    heads: int = 24
    head_dim: int = 128
    n_layers: int = 60
    mlp_hidden: int = 12288  # FeedForward mult 4, GELU (tanh)
    in_channels: int = 64  # 16 latent channels x 2x2 patch
    patch: int = 2
    joint_attention_dim: int = 3584  # Qwen2.5-VL hidden size
    axes_dims: tuple = (16, 56, 56)  # (frame/index, h, w) rope dims
    rope_theta: float = 10000.0
    eps: float = 1e-6
    timestep_channels: int = 256


@dataclass(frozen=True)
class TextEncoderConfig:
    num_layers: int = 28
    hidden: int = 3584
    heads: int = 28
    kv_heads: int = 4
    head_dim: int = 128
    intermediate: int = 18944
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    mrope_section: tuple = (16, 24, 24)  # frequency pairs per (t, h, w) axis


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 32
    hidden: int = 1280
    heads: int = 16
    intermediate: int = 3420
    out_hidden: int = 3584
    patch: int = 14
    temporal_patch: int = 2
    merge: int = 2
    window: int = 112
    fullatt_blocks: tuple = (7, 15, 23, 31)
    mean: tuple = (0.48145466, 0.4578275, 0.40821073)
    std: tuple = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 4
    shift: float = 3.1  # ModelSamplingAuraFlow (multiplier 1)
    lora_strength: float = 1.0
    max_images: int = 3
    vl_area: int = 384 * 384  # images seen by the text encoder
    ref_area: int = 1024 * 1024  # reference latents


DIT = DiTConfig()
TE = TextEncoderConfig()
VISION = VisionConfig()
SAMPLER = SamplerConfig()

# Wan 2.1 latent normalization (ComfyUI latent_formats.Wan21, the same values as the VAE config)
LATENTS_MEAN = (-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921)
LATENTS_STD = (2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
               3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160)

# token ids of the Qwen2.5 tokenizer used by the edit template
IM_START, IMAGE_PAD, USER, NEWLINE = 151644, 151655, 872, 198
SYSTEM_PROMPT = ("Describe the key features of the input image (color, shape, size, texture, objects, background), "
                 "then explain how the user's text instruction should alter or modify the image. Generate a new image "
                 "that meets the user's requirements while maintaining consistency with the original input where "
                 "appropriate.")
TEMPLATE = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
IMAGE_PROMPT = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"

HF_REPO = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LICENSE = "apache-2.0"

# The four ComfyUI repackaged single files, with the HF repo each is published from and the
# sha256 this port was validated against. Every repo here is Apache-2.0.
COMFY_FILES = {
    "dit": ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
            "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
            "c9fdc158e46d3b61ef75f21ae866ca2fe808bf4a53643120d1c1e87c19280a4e"),
    "lora": ("lightx2v/Qwen-Image-Edit-2511-Lightning",
             "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
             "loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
             "22226e8d05d354bb356627d428809f5afd7819399b077238a2b70a82883a904f"),
    "te": ("Comfy-Org/HunyuanVideo_1.5_repackaged",
           "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
           "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
           "cb5636d852a0ea6a9075ab1bef496c0db7aef13c02350571e388aea959c5c0b4"),
    "vae": ("Comfy-Org/Qwen-Image_ComfyUI",
            "split_files/vae/qwen_image_vae.safetensors",
            "vae/qwen_image_vae.safetensors",
            "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f"),
}

# Qwen2.5 tokenizer; identical across the Qwen2.5-VL sizes, so the small one is enough.
TOKENIZER_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"


def comfy_models_dir() -> str:
    return os.environ.get("QWEN_EDIT_MODELS", "/comfy")


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename)


def paths(download: bool = True) -> dict:
    """Resolve the four weight files and the tokenizer.

    Order per file: an explicit QWEN_EDIT_<NAME> path, then the file under $QWEN_EDIT_MODELS
    (the ComfyUI layout) if it is there, then a download from the Hub into the HF cache.
    """
    root = comfy_models_dir()
    out = {}
    for key, (repo, remote, local, _sha) in COMFY_FILES.items():
        override = os.environ.get(f"QWEN_EDIT_{key.upper()}")
        if override:
            out[key] = override
            continue
        candidate = os.path.join(root, local)
        if os.path.exists(candidate) or not download:
            out[key] = candidate
            continue
        out[key] = _download(repo, remote)
    tokenizer = os.environ.get("QWEN_EDIT_TOKENIZER", "")
    out["tokenizer"] = tokenizer or (TOKENIZER_REPO if download else "")
    return out


def sha256_of(key: str) -> str:
    """The sha256 this port was validated against, for `weights.verify_checksums()`."""
    return COMFY_FILES[key][3]
