# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 + Lightning (4 steps) on one P100a: TT text encoder (Qwen2.5-VL LM) + TT DiT, host vision
tower, VAE and scheduler. Follows the reference ComfyUI `qwen_edit` graph (see host.py)."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from typing import List, Optional

import torch

from . import host
from .config import SAMPLER, paths
from .weights import ComfyCheckpoint

# device helpers are model-independent
from .device import Timing, close_device, dram_stats, open_device  # noqa: F401


def release_host_memory():
    """Return freed heap pages to the OS after loading (the weight conversions leave the glibc heap fragmented)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


class QwenEditTT:
    CACHE_SIZE = 6  # per-image vision embeddings + reference tokens (continuous runs reuse their images)

    def __init__(self, dev, dit_prec=None, te_prec=None, host_threads: int = 0, n_blocks: Optional[int] = None):
        import ttnn

        from .tt_dit import QwenEditDiT
        from .tt_text_encoder import QwenVLTextEncoder
        from .vae import QwenImageVAE
        from .vision import VisionTower

        p = paths()
        self.dev = dev
        t0 = time.perf_counter()
        self.tokenizer = _tokenizer(p["tokenizer"])
        te_ckpt = ComfyCheckpoint(p["te"])
        self.vision = VisionTower(te_ckpt, threads=host_threads)
        self.te = QwenVLTextEncoder(dev, te_ckpt, te_prec)
        dit_ckpt = ComfyCheckpoint(p["dit"], p["lora"], SAMPLER.lora_strength)
        self.dit = QwenEditDiT(dev, dit_ckpt, dit_prec, n_blocks=n_blocks)
        self.sigmas = host.sigmas()
        self.dit.load_rows(host.TimeConditioning(dit_ckpt).rows(self.sigmas[:-1]))
        self.vae = QwenImageVAE(p["vae"], threads=host_threads)
        ttnn.synchronize_device(dev)
        # only the token embeddings are read after loading: a fresh handle drops the mapped pages of the layers
        self.te.ckpt = ComfyCheckpoint(p["te"])
        del te_ckpt, dit_ckpt
        release_host_memory()
        self.load_s = time.perf_counter() - t0
        self._prep = None
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()

    # ------------------------------------------------------------------ images
    def _image(self, image: torch.Tensor, timing: Timing):
        """[1, H, W, 3] -> (vision embeddings [n, 3584], grid, raw-normalized reference latent [1, 16, 1, h, w])."""
        key = hashlib.sha256(repr(tuple(image.shape)).encode() + image.numpy().tobytes()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        t0 = time.perf_counter()
        patches, grid = host.vision_patches(host.area_resize(image, SAMPLER.vl_area))
        emb = self.vision(patches, grid)
        timing.mark("vision_s", t0)
        t0 = time.perf_counter()
        ref = host.latent_in(self.vae.encode(host.area_resize(image, SAMPLER.ref_area, multiple=8)))
        timing.mark("reference_encode_s", t0)
        self._cache[key] = (emb, grid, ref)
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return emb, grid, ref

    def prepared(self, geo: host.Geometry):
        if self._prep is None or self._prep.geo != geo:
            if self._prep is not None:
                self.dit.release(self._prep)
                old = self._prep.geo
                if (old.width, old.height, old.refs) != (geo.width, geo.height, geo.refs):
                    self.dev.clear_program_cache()
            self._prep = self.dit.prepare(geo)
        return self._prep

    # ------------------------------------------------------------------ public
    def encode_prompt(self, prompt: str, vision: List[torch.Tensor], grids) -> torch.Tensor:
        p = host.build_prompt(self.tokenizer, prompt, len(vision))
        layout, spans = host.expand(p, [v.shape[0] for v in vision])
        positions = host.mrope_positions(len(layout), spans, grids)
        embeds = self.te.embeddings(layout, vision)
        return self.te.encode(embeds, positions, keep_from=p.template_end)

    def generate(self, prompt: str, width: int, height: int, seed: int, images: List, steps: int = SAMPLER.steps):
        """images: 1-3 PIL images (the first is the one being edited and sets the framing). Returns (PIL, Timing)."""
        import ttnn

        if steps != SAMPLER.steps:
            raise ValueError(f"Lightning edit runs {SAMPLER.steps} steps")
        if not 1 <= len(images) <= SAMPLER.max_images:
            raise ValueError(f"1 to {SAMPLER.max_images} images")
        timing = Timing()
        t0 = time.perf_counter()
        floats = [host.to_float_image(im) for im in images]
        floats[0] = host.lanczos_center(floats[0], width, height)
        timing.mark("resize_s", t0)
        per_image = [self._image(f, timing) for f in floats]
        vision = [e for e, _, _ in per_image]
        grids = [g for _, g, _ in per_image]
        refs = [r for _, _, r in per_image]
        t0 = time.perf_counter()
        cond = self.encode_prompt(prompt, vision, grids)
        timing.mark("text_encoder_s", t0)
        geo = host.Geometry(width, height, tuple((r.shape[-2], r.shape[-1]) for r in refs), n_txt=cond.shape[0])
        t0 = time.perf_counter()
        prep = self.prepared(geo)
        ctx = self.dit.context(cond, geo)
        ref_emb = self.dit.embed(torch.cat([host.pack(r) for r in refs]), geo.n_ref_pad)
        timing.mark("context_s", t0)
        x = host.init_noise(geo, seed)
        s = self.sigmas
        x = x * s[0]  # noise_scaling with sigma 1 and denoise 1: the input latent contributes 0
        for i in range(len(s) - 1):
            t0 = time.perf_counter()
            v = self.dit.step(host.pack(x.to(torch.bfloat16)), ref_emb, ctx, s[i], prep)
            timing.mark("dit_s", t0)
            x = host.euler_step(x, host.unpack(v, geo.lat_h, geo.lat_w), s[i], s[i + 1])
        ttnn.deallocate(ctx)
        ttnn.deallocate(ref_emb)
        t0 = time.perf_counter()
        image = self.vae.decode(x)
        timing.mark("vae_decode_s", t0)
        return image, timing
