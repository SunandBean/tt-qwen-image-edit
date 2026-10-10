# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-Edit-2511 MMDiT (60 dual-stream blocks) on ONE Blackhole (P100a), TTNN, batch 1.
Mirrors ref_algo.RefDiT op for op; built from the same Apache-2.0 patterns as the sibling FLUX.2 klein port (github.com/SunandBean/tt-flux2-klein).

Streams: text, image (the latent being denoised) and references (VAE latents of the input images). Image and
reference tokens share every weight but not the modulation: index_timestep_zero gives the references the t = 0
rows. Each stream is its own tile-padded tensor; attention runs on [text ; image ; references] with pad keys
masked. Linear layers carry biases (minimal_matmul bias), the MLP is GELU(tanh) fused into the first matmul, and
gated residuals use dit_minimal_matmul_addcmul_fused. The modulation linears never touch the device: their rows
for every sigma are computed on the host (host.TimeConditioning) and uploaded once.

Weights: bfloat8_b for the 12 block linears (the checkpoint is fp8 e4m3 + per-tensor scale, plus the merged
Lightning LoRA), bf16 for norms, biases and the small in/out projections. ~14.5 GB of DRAM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from . import host
from .config import DIT, DiTConfig

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover
    get_matmul_config = None

MEM = ttnn.DRAM_MEMORY_CONFIG
STREAMS = ("img", "ref", "txt")


def _as_4d(t):
    if len(t.shape) == 4:
        return t
    shape = list(t.shape)
    return ttnn.reshape(t, [1] * (4 - len(shape)) + shape)


@dataclass
class DiTPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    sdpa_fp32_acc: bool = True
    fused_addcmul: bool = False  # needs residual and weight in the same tile format (bf16 weights)


def _dev(dev, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor, dtype=ttnn.bfloat16):
    return _dev(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1), dtype)


class StreamWeights:
    """One stream's linears in one block: fused qkv, attention out, MLP in / out (all with biases)."""

    def __init__(self, dev, g, names, wd):
        q, k, v, o, f1, f2 = names
        self.wqkv = _dev(dev, host.fuse_qkv(g(q + ".weight"), g(k + ".weight"), g(v + ".weight")), wd)
        self.bqkv = _row(dev, torch.cat([g(q + ".bias"), g(k + ".bias"), g(v + ".bias")]))
        self.wo, self.bo = _dev(dev, host.linear_to_mm(g(o + ".weight")), wd), _row(dev, g(o + ".bias"))
        self.w1, self.b1 = _dev(dev, host.linear_to_mm(g(f1 + ".weight")), wd), _row(dev, g(f1 + ".bias"))
        self.w2, self.b2 = _dev(dev, host.linear_to_mm(g(f2 + ".weight")), wd), _row(dev, g(f2 + ".bias"))


class Block:
    def __init__(self, dev, ckpt, i: int, prec: DiTPrecision):
        g = lambda k: ckpt.get(f"transformer_blocks.{i}.{k}", torch.bfloat16)
        wd = prec.weight_dtype
        self.img = StreamWeights(dev, g, ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                                          "img_mlp.net.0.proj", "img_mlp.net.2"), wd)
        self.txt = StreamWeights(dev, g, ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
                                          "txt_mlp.net.0.proj", "txt_mlp.net.2"), wd)
        self.norm_q, self.norm_k = _row(dev, g("attn.norm_q.weight")), _row(dev, g("attn.norm_k.weight"))
        self.norm_added_q, self.norm_added_k = _row(dev, g("attn.norm_added_q.weight")), _row(dev, g("attn.norm_added_k.weight"))


@dataclass
class Prepared:
    geo: host.Geometry
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    mask: Optional[ttnn.Tensor]


class QwenEditDiT:
    def __init__(self, dev, ckpt, prec: Optional[DiTPrecision] = None, cfg: DiTConfig = DIT,
                 n_blocks: Optional[int] = None):
        self.dev, self.cfg = dev, cfg
        self.prec = prec or DiTPrecision()
        self.grid = dev.compute_with_storage_grid_size()
        g = lambda k: ckpt.get(k, torch.bfloat16)
        self.n_blocks = cfg.n_layers if n_blocks is None else n_blocks
        self.blocks: List[Block] = [Block(dev, ckpt, i, self.prec) for i in range(self.n_blocks)]
        self.img_in = (_dev(dev, host.linear_to_mm(g("img_in.weight"))), _row(dev, g("img_in.bias")))
        self.txt_norm = _row(dev, g("txt_norm.weight"))
        self.txt_in = (_dev(dev, host.linear_to_mm(g("txt_in.weight"))), _row(dev, g("txt_in.bias")))
        self.proj_out = (_dev(dev, host.linear_to_mm(g("proj_out.weight"))), _row(dev, g("proj_out.bias")))
        self.trans_mat = _dev(dev, host.rot_transformation_mat())
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self.ck_rope = ck(ttnn.MathFidelity.HiFi4, True)
        self._mm_cfg_cache: Dict[tuple, object] = {}
        self.sdpa_chunks = (256, 128, 64, 32)
        self.gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)
        self.rows: Dict[float, dict] = {}

    # ------------------------------------------------------------------ conditioning rows
    def load_rows(self, step_rows: Dict[float, host.StepRows]):
        """Upload the modulation rows of each sigma (kept for the life of the pipeline)."""
        for sigma, sr in step_rows.items():
            if sigma in self.rows:
                continue
            blocks = [{s: {k: _row(self.dev, v) for k, v in b[s].items()} for s in STREAMS} for b in sr.blocks[: self.n_blocks]]
            self.rows[sigma] = {"blocks": blocks, "out_w": _row(self.dev, sr.out_w), "out_b": _row(self.dev, sr.out_b)}

    # ------------------------------------------------------------------ geometry
    def prepare(self, geo: host.Geometry) -> Prepared:
        cos, sin = host.cos_sin(host.joint_ids(geo))
        mask = host.key_mask(geo)
        f = lambda t: _dev(self.dev, t.reshape(1, 1, t.shape[0], -1))
        return Prepared(geo, f(cos), f(sin), None if mask is None else _dev(self.dev, mask))

    @staticmethod
    def release(prep: Prepared):
        for t in (prep.cos, prep.sin, prep.mask):
            if t is not None:
                ttnn.deallocate(t)

    # ------------------------------------------------------------------ ops
    def _cfg(self, M, K, N):
        key = (M, K, N)
        if key not in self._mm_cfg_cache:
            self._mm_cfg_cache[key] = get_matmul_config(M, K, N, self.grid)
        return self._mm_cfg_cache[key]

    def _mm(self, x, w, b, act=None):
        M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
        out = ttnn.experimental.minimal_matmul(x, w, bias_tensor=b, fused_activation=act, config=self._cfg(M, K, N),
                                               compute_kernel_config=self.ck_mm, dtype=ttnn.bfloat16, memory_config=MEM)
        return _as_4d(out)

    def _gated(self, resid, x, w, b, gate):
        """resid + gate * (x @ w + b); frees resid."""
        if self.prec.fused_addcmul:
            M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
            out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, w, 1.0, resid, gate, bias_tensor=b, config=self._cfg(M, K, N), compute_kernel_config=self.ck_mm,
                dtype=ttnn.bfloat16, memory_config=MEM)
            out = _as_4d(out)
        else:
            y = self._mm(x, w, b)
            gy = ttnn.multiply(y, gate, memory_config=MEM)
            ttnn.deallocate(y)
            out = ttnn.add(resid, gy, memory_config=MEM)
            ttnn.deallocate(gy)
        ttnn.deallocate(resid)
        return out

    def _ln(self, x, w, b):
        return ttnn.layer_norm(x, epsilon=self.cfg.eps, weight=w, bias=b, compute_kernel_config=self.ck_norm,
                               memory_config=MEM)

    def _rms(self, x, w, eps=None):
        return ttnn.rms_norm(x, epsilon=self.cfg.eps if eps is None else eps, weight=w,
                             compute_kernel_config=self.ck_norm, memory_config=MEM)

    def _heads(self, qkv):
        c = self.cfg
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        return q, k, v

    def _rows(self, x, start, stop):
        return ttnn.slice(x, [0, 0, start, 0], [1, 1, stop, x.shape[-1]], memory_config=MEM)

    def _block(self, h: Dict[str, ttnn.Tensor], blk: Block, rows, prep: Prepared):
        geo = prep.geo
        weights = {"img": blk.img, "ref": blk.img, "txt": blk.txt}
        qkn = {"img": (blk.norm_q, blk.norm_k), "ref": (blk.norm_q, blk.norm_k), "txt": (blk.norm_added_q, blk.norm_added_k)}
        order = [s for s in ("txt", "img", "ref") if s in h]
        qs, ks, vs = [], [], []
        for s in order:
            x = self._ln(h[s], rows[s]["ln1_w"], rows[s]["ln1_b"])
            q, k, v = self._heads(self._mm(x, weights[s].wqkv, weights[s].bqkv))
            ttnn.deallocate(x)
            qs.append(self._rms(q, qkn[s][0]))
            ks.append(self._rms(k, qkn[s][1]))
            vs.append(v)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
        q = ttnn.concat(qs, dim=2, memory_config=MEM)
        k = ttnn.concat(ks, dim=2, memory_config=MEM)
        v = ttnn.concat(vs, dim=2, memory_config=MEM)
        for t in (*qs, *ks, *vs):
            ttnn.deallocate(t)
        o = self._sdpa(q, k, v, prep)
        bounds = {"txt": (0, geo.n_txt_pad), "img": (geo.img_start, geo.img_start + geo.n_img_pad),
                  "ref": (geo.ref_start, geo.ref_start + geo.n_ref_pad)}
        for s in order:
            o_s = self._rows(o, *bounds[s])
            h[s] = self._gated(h[s], o_s, weights[s].wo, weights[s].bo, rows[s]["gate1"])
            ttnn.deallocate(o_s)
        ttnn.deallocate(o)
        for s in order:
            x = self._ln(h[s], rows[s]["ln2_w"], rows[s]["ln2_b"])
            m = self._mm(x, weights[s].w1, weights[s].b1, act=self.gelu)
            ttnn.deallocate(x)
            h[s] = self._gated(h[s], m, weights[s].w2, weights[s].b2, rows[s]["gate2"])
            ttnn.deallocate(m)
        return h

    def _sdpa(self, q, k, v, prep: Prepared):
        S = prep.geo.n_joint
        qr = ttnn.experimental.rotary_embedding_llama(q, prep.cos, prep.sin, self.trans_mat, is_decode_mode=False,
                                                      compute_kernel_config=self.ck_rope)
        kr = ttnn.experimental.rotary_embedding_llama(k, prep.cos, prep.sin, self.trans_mat, is_decode_mode=False,
                                                      compute_kernel_config=self.ck_rope)
        for t in (q, k):
            ttnn.deallocate(t)
        chunk = next(ch for ch in self.sdpa_chunks if S % ch == 0)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk,
                                     exp_approx_mode=False)
        attn = ttnn.transformer.scaled_dot_product_attention(
            qr, kr, v, attn_mask=prep.mask, is_causal=False, scale=self.cfg.head_dim ** -0.5, program_config=cfg,
            compute_kernel_config=self.ck_sdpa, memory_config=MEM)
        for t in (qr, kr, v):
            ttnn.deallocate(t)
        out = _as_4d(ttnn.transformer.concatenate_heads(attn, memory_config=MEM))
        ttnn.deallocate(attn)
        return out

    # ------------------------------------------------------------------ public
    def context(self, cond: torch.Tensor, geo: host.Geometry) -> ttnn.Tensor:
        """[T, 3584] conditioning -> txt_in(txt_norm(cond)) [1, 1, n_txt_pad, dim] on device."""
        p = _dev(self.dev, host.pad_rows(cond.to(torch.bfloat16), geo.n_txt_pad).reshape(1, 1, geo.n_txt_pad, -1))
        n = self._rms(p, self.txt_norm, eps=1e-6)
        ttnn.deallocate(p)
        out = self._mm(n, *self.txt_in)
        ttnn.deallocate(n)
        return out

    def embed(self, tokens: torch.Tensor, rows: int) -> ttnn.Tensor:
        p = _dev(self.dev, host.pad_rows(tokens.to(torch.bfloat16), rows).reshape(1, 1, rows, -1))
        out = self._mm(p, *self.img_in)
        ttnn.deallocate(p)
        return out

    def step(self, x_tokens: torch.Tensor, ref_embedded: Optional[ttnn.Tensor], ctx: ttnn.Tensor, sigma: float,
             prep: Prepared, taps: Optional[list] = None) -> torch.Tensor:
        """x_tokens [N, 64] host latent tokens -> velocity [N, 64] float32 on host. ref_embedded is img_in of the
        reference tokens (constant over the steps), ctx the text context; neither is consumed."""
        geo = prep.geo
        r = self.rows[sigma]
        h = {"txt": ttnn.clone(ctx, memory_config=MEM), "img": self.embed(x_tokens, geo.n_img_pad)}
        if ref_embedded is not None:
            h["ref"] = ttnn.clone(ref_embedded, memory_config=MEM)
        for i, blk in enumerate(self.blocks):
            h = self._block(h, blk, r["blocks"][i], prep)
            if taps is not None:
                taps.append({s: ttnn.to_torch(t)[0, 0].float() for s, t in h.items()})
        for s in ("txt", "ref"):
            if s in h:
                ttnn.deallocate(h[s])
        xn = self._ln(h["img"], r["out_w"], r["out_b"])
        ttnn.deallocate(h["img"])
        out = self._mm(xn, *self.proj_out)
        ttnn.deallocate(xn)
        v = ttnn.to_torch(out)[0, 0, : geo.n_img].float()
        ttnn.deallocate(out)
        return v
