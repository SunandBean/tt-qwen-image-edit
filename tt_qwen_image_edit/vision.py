# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-VL vision tower on the host (CPU): ~0.67B parameters for ~200 merged tokens per image. Weights are
kept in bf16 (1.3 GB of host memory instead of 2.7 GB) and widened to float32 one block at a time; the math runs
in float32 like ComfyUI's (PCC 0.99999 vs its vision tower, bf16 math only reached 0.9991).

Same computation as transformers' Qwen2_5_VisionTransformerPretrainedModel / ComfyUI's Qwen2VLVisionTransformer:
3D patch embed (2 x 14 x 14), 32 blocks of RMSNorm + window attention (full attention in blocks 7, 15, 23, 31)
with 2D rotary embeddings, SwiGLU MLP, then the 2x2 patch merger to the LM width (3584). Tokens are processed in
window order and restored to raster order after the merger."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from .config import VISION, VisionConfig

PFX = "visual."


def _rms(x, w, eps=1e-6):
    xf = x.float()
    return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()).to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VisionTower:
    def __init__(self, ckpt, cfg: VisionConfig = VISION, threads: int = 0, dtype=torch.bfloat16):
        self.cfg = cfg
        g = lambda k: ckpt.get(PFX + k, dtype)
        self.patch_w = g("patch_embed.proj.weight").reshape(cfg.hidden, -1)  # Conv3d with stride = kernel
        self.blocks = []
        for i in range(cfg.depth):
            p = f"blocks.{i}."
            self.blocks.append({k: g(p + k) for k in (
                "norm1.weight", "norm2.weight", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                "mlp.gate_proj.weight", "mlp.gate_proj.bias", "mlp.up_proj.weight", "mlp.up_proj.bias",
                "mlp.down_proj.weight", "mlp.down_proj.bias")})
        self.ln_q = g("merger.ln_q.weight")
        self.mlp0 = (g("merger.mlp.0.weight"), g("merger.mlp.0.bias"))
        self.mlp2 = (g("merger.mlp.2.weight"), g("merger.mlp.2.bias"))
        self.threads = threads

    # ------------------------------------------------------------------ layout
    def _window_index(self, grid: Tuple[int, int, int]):
        c = self.cfg
        t, gh, gw = grid
        win = c.window // c.merge // c.patch
        lh, lw = gh // c.merge, gw // c.merge
        index = torch.arange(t * lh * lw).reshape(t, lh, lw)
        pad_h, pad_w = win - lh % win, win - lw % win
        nh, nw = (lh + pad_h) // win, (lw + pad_w) // win
        padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(t, nh, win, nw, win).permute(0, 1, 3, 2, 4).reshape(t, nh * nw, win, win)
        seqlens = (padded != -100).sum([2, 3]).reshape(-1)
        flat = padded.reshape(-1)
        window_index = flat[flat != -100]
        cu = (seqlens.cumsum(0) * c.merge * c.merge).tolist()
        cu_window = torch.unique_consecutive(torch.tensor([0] + cu))
        return window_index, cu_window

    def _rotary(self, grid):
        c = self.cfg
        t, h, w = grid
        m = c.merge
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        pos = torch.stack([hpos, wpos], dim=-1).repeat(t, 1)
        dim = (c.hidden // c.heads) // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        freqs = torch.outer(torch.arange(max(h, w), dtype=torch.float), inv_freq)
        return freqs[pos].flatten(1)  # [seq, head_dim / 2]

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def __call__(self, patches: torch.Tensor, grid: Tuple[int, int, int]) -> torch.Tensor:
        """[seq, 1176] patches (process_qwen2vl_images order) -> [seq / 4, 3584] float32 embeddings."""
        prev = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(self.threads)
        try:
            return self._forward(patches.float(), grid)
        finally:
            torch.set_num_threads(prev)

    def _forward(self, patches, grid):
        c = self.cfg
        unit = c.merge * c.merge
        x = patches @ self.patch_w.float().t()  # [seq, hidden]
        seq = x.shape[0]
        window_index, cu_window = self._window_index(grid)
        x = x.reshape(seq // unit, unit, -1)[window_index].reshape(seq, -1)
        rot = self._rotary(grid).reshape(seq // unit, unit, -1)[window_index].reshape(seq, -1)
        emb = torch.cat([rot, rot], dim=-1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        cu_full = torch.tensor([0, seq])
        heads, hd = c.heads, c.hidden // c.heads
        for i, stored in enumerate(self.blocks):
            b = {k: v.float() for k, v in stored.items()}
            cu = cu_full if i in c.fullatt_blocks else cu_window
            h = _rms(x, b["norm1.weight"])
            qkv = F.linear(h, b["attn.qkv.weight"], b["attn.qkv.bias"]).reshape(seq, 3, heads, hd).permute(1, 0, 2, 3)
            q, k, v = qkv.unbind(0)  # [seq, heads, hd]
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            outs: List[torch.Tensor] = []
            bounds = cu.tolist()
            for s0, s1 in zip(bounds[:-1], bounds[1:]):
                qs, ks, vs = (t[s0:s1].transpose(0, 1)[None] for t in (q, k, v))
                outs.append(F.scaled_dot_product_attention(qs, ks, vs)[0].transpose(0, 1).reshape(s1 - s0, -1))
            a = torch.cat(outs, dim=0)
            x = x + F.linear(a, b["attn.proj.weight"], b["attn.proj.bias"])
            h = _rms(x, b["norm2.weight"])
            m = F.silu(F.linear(h, b["mlp.gate_proj.weight"], b["mlp.gate_proj.bias"])) * F.linear(
                h, b["mlp.up_proj.weight"], b["mlp.up_proj.bias"])
            x = x + F.linear(m, b["mlp.down_proj.weight"], b["mlp.down_proj.bias"])
        y = _rms(x, self.ln_q).reshape(-1, c.hidden * unit)
        f = lambda wb: tuple(t.float() for t in wb)
        y = F.linear(F.gelu(F.linear(y, *f(self.mlp0))), *f(self.mlp2))
        return y[torch.argsort(window_index)]
