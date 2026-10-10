# SPDX-License-Identifier: Apache-2.0
"""Plain-torch (CPU, float32) reference of the edit DiT and the Qwen2.5-VL language model, written from the
ComfyUI semantics described in host.py. Used only by the parity checks (experiments/qwen-edit); blocks are read
from the checkpoint one at a time so a full forward fits in a few GB of host memory."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from . import host
from .config import DIT, TE, DiTConfig, TextEncoderConfig


def _rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _ln(x, eps):
    return F.layer_norm(x, (x.shape[-1],), eps=eps)


def rope_pairs(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """ComfyUI apply_rope1: adjacent (2i, 2i+1) pairs rotated by angles [S, 64]; x [heads, S, 128]."""
    cos, sin = angles.cos(), angles.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([cos * x0 - sin * x1, sin * x0 + cos * x1], dim=-1).flatten(-2)


class RefBlock:
    def __init__(self, ckpt, i: int, cfg: DiTConfig = DIT):
        p = f"transformer_blocks.{i}."
        self.w = {k[len(p):]: ckpt.get(k, torch.float32) for k in ckpt.keys()
                  if k.startswith(p) and k.endswith((".weight", ".bias")) and "_mod." not in k}
        self.cfg = cfg

    def lin(self, x, name):
        return F.linear(x, self.w[name + ".weight"], self.w.get(name + ".bias"))

    def __call__(self, img, ref, txt, rows: Dict[str, Dict[str, torch.Tensor]], angles):
        """img [N, D], ref [R, D], txt [T, D] float32; rows from host.StepRows.blocks[i]; angles [T + N + R, 64]."""
        c = self.cfg
        f = lambda d, k: d[k].float()
        mod = lambda x, r, n: _ln(x, c.eps) * f(r, f"ln{n}_w") + f(r, f"ln{n}_b")
        hi, hr, ht = mod(img, rows["img"], 1), mod(ref, rows["ref"], 1), mod(txt, rows["txt"], 1)
        hs = torch.cat([hi, hr])

        def heads(x):
            return x.view(x.shape[0], c.heads, c.head_dim).transpose(0, 1)

        q_s = _rms(heads(self.lin(hs, "attn.to_q")), self.w["attn.norm_q.weight"], c.eps)
        k_s = _rms(heads(self.lin(hs, "attn.to_k")), self.w["attn.norm_k.weight"], c.eps)
        v_s = heads(self.lin(hs, "attn.to_v"))
        q_t = _rms(heads(self.lin(ht, "attn.add_q_proj")), self.w["attn.norm_added_q.weight"], c.eps)
        k_t = _rms(heads(self.lin(ht, "attn.add_k_proj")), self.w["attn.norm_added_k.weight"], c.eps)
        v_t = heads(self.lin(ht, "attn.add_v_proj"))
        q = rope_pairs(torch.cat([q_t, q_s], 1), angles)
        k = rope_pairs(torch.cat([k_t, k_s], 1), angles)
        v = torch.cat([v_t, v_s], 1)
        o = F.scaled_dot_product_attention(q[None], k[None], v[None])[0].transpose(0, 1).reshape(q.shape[1], -1)
        T, N = txt.shape[0], img.shape[0]
        o_t, o_s = o[:T], o[T:]
        a_s = self.lin(o_s, "attn.to_out.0")
        img = img + f(rows["img"], "gate1") * a_s[:N]
        ref = ref + f(rows["ref"], "gate1") * a_s[N:]
        txt = txt + f(rows["txt"], "gate1") * self.lin(o_t, "attn.to_add_out")

        def mlp(x, name):
            return self.lin(F.gelu(self.lin(x, f"{name}.net.0.proj"), approximate="tanh"), f"{name}.net.2")

        img = img + f(rows["img"], "gate2") * mlp(mod(img, rows["img"], 2), "img_mlp")
        ref = ref + f(rows["ref"], "gate2") * mlp(mod(ref, rows["ref"], 2), "img_mlp")
        txt = txt + f(rows["txt"], "gate2") * mlp(mod(txt, rows["txt"], 2), "txt_mlp")
        return img, ref, txt


class RefDiT:
    def __init__(self, ckpt, cfg: DiTConfig = DIT):
        self.ckpt, self.cfg = ckpt, cfg
        g = lambda k: ckpt.get(k, torch.float32)
        self.img_in = (g("img_in.weight"), g("img_in.bias"))
        self.txt_norm = g("txt_norm.weight")
        self.txt_in = (g("txt_in.weight"), g("txt_in.bias"))
        self.proj_out = (g("proj_out.weight"), g("proj_out.bias"))

    @torch.no_grad()
    def forward(self, x_tokens, ref_tokens, cond, step_rows: host.StepRows, geo: host.Geometry,
                n_blocks: Optional[int] = None, taps: Optional[List] = None):
        """x_tokens [N, 64], ref_tokens [R, 64], cond [T, 3584] -> velocity [N, 64] float32."""
        c = self.cfg
        img = F.linear(x_tokens.float(), *self.img_in)
        ref = F.linear(ref_tokens.float(), *self.img_in)
        txt = F.linear(_rms(cond.float(), self.txt_norm, 1e-6), *self.txt_in)
        ids = torch.cat([host.text_ids(geo), host.image_ids(geo), host.ref_ids(geo)])
        angles = host.rope_angles(ids).float()
        nb = c.n_layers if n_blocks is None else n_blocks
        for i in range(nb):
            blk = RefBlock(self.ckpt, i, c)
            img, ref, txt = blk(img, ref, txt, step_rows.blocks[i], angles)
            del blk
            if taps is not None:
                taps.append((img.clone(), ref.clone(), txt.clone()))
        out = _ln(img, c.eps) * step_rows.out_w.float() + step_rows.out_b.float()
        return F.linear(out, *self.proj_out)


class RefLM:
    """Qwen2.5-7B language model (ComfyUI Llama2_ with qkv bias, GQA 28/4, mrope), final norm applied."""

    def __init__(self, ckpt, cfg: TextEncoderConfig = TE):
        self.ckpt, self.cfg = ckpt, cfg
        self.norm = ckpt.get("model.norm.weight", torch.float32)

    @torch.no_grad()
    def forward(self, embeds: torch.Tensor, positions: torch.Tensor, n_layers: Optional[int] = None,
                final_norm: bool = True, dtype=torch.float32, taps=None) -> torch.Tensor:
        """dtype=torch.bfloat16 runs every op in bf16 (ComfyUI's GPU precision) instead of float32."""
        c = self.cfg
        L = embeds.shape[0]
        angles = host.te_angles(positions)
        h = embeds.to(dtype)
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        for i in range(c.num_layers if n_layers is None else n_layers):
            g = lambda k: self.ckpt.get(f"model.layers.{i}.{k}", dtype)
            x = _rms(h, g("input_layernorm.weight"), c.rms_eps)
            q = F.linear(x, g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")).view(L, c.heads, c.head_dim).transpose(0, 1)
            k = F.linear(x, g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")).view(L, c.kv_heads, c.head_dim).transpose(0, 1)
            v = F.linear(x, g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")).view(L, c.kv_heads, c.head_dim).transpose(0, 1)
            q = host.apply_rope_rotate_half(q.float(), angles).to(dtype)
            k = host.apply_rope_rotate_half(k.float(), angles).to(dtype)
            rep = c.heads // c.kv_heads
            k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
            a = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=causal)[0]
            h = h + F.linear(a.transpose(0, 1).reshape(L, -1), g("self_attn.o_proj.weight"))
            x = _rms(h, g("post_attention_layernorm.weight"), c.rms_eps)
            m = F.silu(F.linear(x, g("mlp.gate_proj.weight"))) * F.linear(x, g("mlp.up_proj.weight"))
            h = h + F.linear(m, g("mlp.down_proj.weight"))
            if taps is not None:
                taps.append(h.float().clone())
        return _rms(h, self.norm.to(dtype), c.rms_eps) if final_norm else h
