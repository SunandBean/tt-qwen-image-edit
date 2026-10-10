# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image VAE (Wan 2.1 architecture, 16 latent channels) on the host, bf16.

The ComfyUI file uses the original Wan key names; diffusers' converter maps them onto AutoencoderKLWan (default
config = Wan 2.1, all keys matched). One frame in, one frame out: every causal 3D convolution then sees its input
frame only through the last temporal slice of its kernel (the frames before it are causal zero padding), so it is
run as a 2D convolution with weight[:, :, -1]. Same result, and torch's CPU conv3d (im2col) would otherwise take
several GB of scratch at 1024^2."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from . import host


def _single_frame_forward(self, x, cache_x=None):
    if x.shape[2] == 1 and cache_x is None and self._padding[4] == self.kernel_size[0] - 1 and self.stride[0] == 1:
        y = F.conv2d(x[:, :, 0], self.weight[:, :, -1], self.bias, stride=self.stride[1:],
                     padding=(self._padding[2], self._padding[0]))
        return y.unsqueeze(2)
    return type(self).forward(self, x, cache_x)


def single_frame(vae):
    """Route every WanCausalConv3d through the 2D path for one-frame inputs (no frame cache)."""
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    for m in vae.modules():
        if isinstance(m, WanCausalConv3d) and m._padding[0] == m._padding[1] and m._padding[2] == m._padding[3]:
            m.forward = types.MethodType(_single_frame_forward, m)
    return vae


class QwenImageVAE:
    def __init__(self, path: str, dtype=torch.bfloat16, threads: int = 0):
        from diffusers import AutoencoderKLWan
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import load_file

        self.vae = AutoencoderKLWan()
        self.vae.load_state_dict(convert_wan_vae_to_diffusers(load_file(path)), strict=True)
        self.vae.eval().to(dtype)
        single_frame(self.vae)
        self.dtype = dtype
        self.threads = threads

    def _threads(self):
        prev = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        return prev

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """[1, H, W, 3] float in [0, 1] -> raw latent mean [1, 16, 1, H/8, W/8] float32 (ComfyUI VAEEncode)."""
        prev = self._threads()
        try:
            x = (image.movedim(-1, 1) * 2.0 - 1.0).unsqueeze(2).to(self.dtype)  # [1, 3, 1, H, W]
            return self.vae.encode(x).latent_dist.mode().float()
        finally:
            torch.set_num_threads(prev)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor):
        """Normalized sampler latent [1, 16, 1, h, w] -> PIL image (ComfyUI VAEDecode + SaveImage rounding)."""
        from PIL import Image
        import numpy as np

        prev = self._threads()
        try:
            z = host.latent_out(latent.float()).to(self.dtype)
            y = self.vae.decode(z).sample.float()  # [1, 3, 1, H, W] in ~[-1, 1]
        finally:
            torch.set_num_threads(prev)
        img = torch.clamp((y[0, :, 0] + 1.0) / 2.0, 0.0, 1.0).permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip(255.0 * img, 0, 255).astype(np.uint8))
