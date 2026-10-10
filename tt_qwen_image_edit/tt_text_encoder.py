# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen2.5-VL-7B language model prefill on one Blackhole, batch 1: the edit prompt (with the vision tokens of the
input images already spliced in on the host) -> final-norm hidden states [T, 3584].

Adapted from the sibling FLUX.2 klein port's text encoder (github.com/SunandBean/tt-flux2-klein, Qwen3). Differences: q/k/v projections carry biases, no q/k
norm, GQA 28/4, multimodal rope: every token has (t, h, w) positions and frequency pair j rotates with the axis
of its mrope section, so the cos/sin tables are built per prompt on the host (in the permuted adjacent-pair layout
the rope kernel expects, q/k rows permuted per head). The sequence is padded at the end to a multiple of 128 and
attention is causal, so pad rows never reach the valid ones. Embedding lookup stays on the host.

Attention is NOT the fused SDPA kernel: Qwen2.5's large q/k biases give logits up to ~6e3 in layer 0, and the SDPA
kernel keeps QK^T in bf16 (a step of 32 at that size), which flips the softmax winner for some text rows
(row cosine 0.78 after one layer). Scores are computed as an fp32-output matmul instead, softmax in fp32, then
the value matmul; GQA by folding the 7 query heads of each kv head into the row dimension."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import ttnn

from . import host
from .config import TE, TextEncoderConfig

MEM = ttnn.DRAM_MEMORY_CONFIG
PFX = "model."
SEQ_MULTIPLE = 128


@dataclass
class TEPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    packer_l1_acc: bool = False  # True accumulates K blocks in bf16 (Qwen2.5 down_proj has K = 18944)
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4  # q/k biases make large logits


class TELayer:
    def __init__(self, dev, ckpt, idx: int, prec: TEPrecision, perm, cfg: TextEncoderConfig):
        g = lambda k: ckpt.get(f"{PFX}layers.{idx}.{k}", torch.bfloat16)
        dt = lambda t, dtype=prec.weight_dtype: ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                                                device=dev, memory_config=MEM)
        row = lambda t: dt(t.reshape(1, 1, 1, -1), ttnn.bfloat16)
        wq = host.permute_heads_rows(g("self_attn.q_proj.weight"), cfg.heads, cfg.head_dim, perm)
        wk = host.permute_heads_rows(g("self_attn.k_proj.weight"), cfg.kv_heads, cfg.head_dim, perm)
        bq = host.permute_heads_rows(g("self_attn.q_proj.bias"), cfg.heads, cfg.head_dim, perm)
        bk = host.permute_heads_rows(g("self_attn.k_proj.bias"), cfg.kv_heads, cfg.head_dim, perm)
        self.wqkv = dt(host.fuse_qkv(wq, wk, g("self_attn.v_proj.weight")))
        self.bqkv = row(torch.cat([bq, bk, g("self_attn.v_proj.bias")]))
        self.wo = dt(host.linear_to_mm(g("self_attn.o_proj.weight")))
        self.w_gateup = dt(host.swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")))
        self.w_down = dt(host.linear_to_mm(g("mlp.down_proj.weight")))
        self.ln1 = row(g("input_layernorm.weight"))
        self.ln2 = row(g("post_attention_layernorm.weight"))


class QwenVLTextEncoder:
    def __init__(self, dev, ckpt, prec: Optional[TEPrecision] = None, cfg: TextEncoderConfig = TE,
                 n_layers: Optional[int] = None):
        self.dev, self.cfg, self.ckpt = dev, cfg, ckpt
        self.prec = prec or TEPrecision()
        self.perm = host.interleave_pairs_permutation(cfg.head_dim)
        n = cfg.num_layers if n_layers is None else n_layers
        self.layers: List[TELayer] = [TELayer(dev, ckpt, i, self.prec, self.perm, cfg) for i in range(n)]
        self.norm = ttnn.from_torch(ckpt.get(PFX + "norm.weight", torch.bfloat16).reshape(1, 1, 1, -1),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)
        self.final_norm = n == cfg.num_layers
        self.grid = dev.compute_with_storage_grid_size()
        arch = dev.arch()
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=self.prec.mm_fidelity, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=self.prec.packer_l1_acc)
        self.ck_norm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=self.prec.sdpa_fidelity, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        self.trans_mat = ttnn.from_torch(host.rot_transformation_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=MEM)

    def _linear(self, x, w, b=None):
        return ttnn.linear(x, w, bias=b, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=ttnn.bfloat16)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.cfg.rms_eps, weight=w, compute_kernel_config=self.ck_norm,
                             memory_config=MEM)

    def embeddings(self, layout, vision: List[torch.Tensor]) -> torch.Tensor:
        """Expanded layout from host.expand -> [L, 3584] bf16 input embeddings."""
        tok_pos = [i for i, e in enumerate(layout) if e[0] == "tok"]
        ids = torch.tensor([layout[i][1] for i in tok_pos])
        rows = self.ckpt.get_rows(PFX + "embed_tokens.weight", ids).to(torch.bfloat16)
        out = torch.empty(len(layout), self.cfg.hidden, dtype=torch.bfloat16)
        out[tok_pos] = rows
        for i, e in enumerate(layout):
            if e[0] == "img":
                out[i] = vision[e[1]][e[2]].to(torch.bfloat16)
        return out

    def _attention(self, q, k, v, mask, Lp):
        """q [1, 28, Lp, 128], k / v [1, 4, Lp, 128] (bf16) -> [1, 28, Lp, 128] bf16; fp32 scores + softmax."""
        c = self.cfg
        g = c.heads // c.kv_heads
        qg = ttnn.reshape(q, [c.kv_heads, 1, g * Lp, c.head_dim])
        kg = ttnn.reshape(k, [c.kv_heads, 1, Lp, c.head_dim])
        vg = ttnn.reshape(v, [c.kv_heads, 1, Lp, c.head_dim])
        scores = ttnn.matmul(qg, kg, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.ck_sdpa,
                             memory_config=MEM)
        scores = ttnn.reshape(scores, [c.kv_heads, g, Lp, Lp])
        scores = ttnn.multiply(scores, c.head_dim ** -0.5, memory_config=MEM)
        masked = ttnn.add(scores, mask, memory_config=MEM)
        ttnn.deallocate(scores)
        probs = ttnn.softmax(masked, dim=-1, numeric_stable=True, compute_kernel_config=self.ck_sdpa, memory_config=MEM)
        ttnn.deallocate(masked)
        probs = ttnn.reshape(probs, [c.kv_heads, 1, g * Lp, Lp])
        out = ttnn.matmul(probs, vg, dtype=ttnn.bfloat16, compute_kernel_config=self.ck_sdpa, memory_config=MEM)
        ttnn.deallocate(probs)
        return ttnn.reshape(out, [1, c.heads, Lp, c.head_dim])

    def encode(self, embeds: torch.Tensor, positions: torch.Tensor, keep_from: int = 0, taps=None) -> torch.Tensor:
        """embeds [L, 3584] bf16, positions [3, L] -> final-norm hidden states [L - keep_from, 3584] bf16 (host)."""
        c = self.cfg
        L = embeds.shape[0]
        Lp = host.round_up(L, SEQ_MULTIPLE)
        cos, sin = host.te_cos_sin(positions, Lp)
        tab = lambda t: ttnn.from_torch(t.reshape(1, 1, Lp, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=self.dev, memory_config=MEM)
        cos_t, sin_t = tab(cos), tab(sin)
        h = ttnn.from_torch(host.pad_rows(embeds.to(torch.bfloat16), Lp).reshape(1, 1, Lp, -1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=MEM)
        causal = torch.full((Lp, Lp), host.MASK_NEG).triu(1).reshape(1, 1, Lp, Lp)
        mask = ttnn.from_torch(causal, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=MEM)
        for layer in self.layers:
            x = self._rms(h, layer.ln1)
            qkv = self._linear(x, layer.wqkv, layer.bqkv)
            ttnn.deallocate(x)
            if len(qkv.shape) != 4:
                qkv = ttnn.reshape(qkv, [1, 1, Lp, -1])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.kv_heads,
                                                             transpose_k_heads=False, memory_config=MEM)
            ttnn.deallocate(qkv)
            qr = ttnn.experimental.rotary_embedding_llama(q, cos_t, sin_t, self.trans_mat, is_decode_mode=False,
                                                          compute_kernel_config=self.ck_norm)
            kr = ttnn.experimental.rotary_embedding_llama(k, cos_t, sin_t, self.trans_mat, is_decode_mode=False,
                                                          compute_kernel_config=self.ck_norm)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            attn = self._attention(qr, kr, v, mask, Lp)
            for t in (qr, kr, v):
                ttnn.deallocate(t)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=MEM)
            ttnn.deallocate(attn)
            o = self._linear(a, layer.wo)
            ttnn.deallocate(a)
            h2 = ttnn.add(h, o, memory_config=MEM)
            ttnn.deallocate(o)
            ttnn.deallocate(h)
            x2 = self._rms(h2, layer.ln2)
            m = ttnn.experimental.minimal_matmul(x2, layer.w_gateup, compute_kernel_config=self.ck_mm,
                                                 dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=True)
            ttnn.deallocate(x2)
            d = self._linear(m, layer.w_down)
            ttnn.deallocate(m)
            h = ttnn.add(h2, d, memory_config=MEM)
            ttnn.deallocate(d)
            ttnn.deallocate(h2)
            if taps is not None:
                taps.append(ttnn.to_torch(h)[0, 0, :L].float())
        if self.final_norm:
            n = self._rms(h, self.norm)
            ttnn.deallocate(h)
            h = n
        out = ttnn.to_torch(h)[0, 0, keep_from:L].to(torch.bfloat16)
        for t in (h, cos_t, sin_t, mask):
            ttnn.deallocate(t)
        return out
