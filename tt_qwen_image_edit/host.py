# SPDX-License-Identifier: Apache-2.0
"""Host-side (CPU) parts of the single-P100a Qwen-Image-Edit-2511 port, following the reference ComfyUI `qwen_edit`
graph for batch 1:

  * images: image 1 is center-cropped and Lanczos-resized to the output size (ImageScale); every image is seen by
    the text encoder at ~384^2 (area resize, then bilinear to a multiple of 28) and VAE-encoded at ~1024^2
    (area resize to a multiple of 8) as a reference latent
  * prompt: "Picture k: <|vision_start|><|image_pad|><|vision_end|>" per image + prompt inside the edit chat
    template; the hidden states from the user turn on (template prefix dropped) condition the DiT
  * DiT tokens: [text ; image (output latent) ; references], 2x2 patches (odd latent sizes padded circularly),
    rope ids (index, h, w) centred on the grid, references at index 1..k, text on the diagonal after the image
  * index_timestep_zero: reference tokens are modulated with the t = 0 embedding, image tokens with t
  * schedule: ModelSamplingAuraFlow(shift 3.1) "simple" sigmas, Euler in float32, noise from torch.manual_seed
On the device every segment is padded to whole tiles; pad keys are masked."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import (DIT, IM_START, IMAGE_PAD, IMAGE_PROMPT, LATENTS_MEAN, LATENTS_STD, NEWLINE, SAMPLER, TE,
                     TEMPLATE, TILE, USER, VISION, DiTConfig)

MASK_NEG = -1e9  # additive bias for masked keys (bf16-representable, as in the other P100a ports)
SDPA_MULTIPLE = 256  # the joint sequence is padded (text segment) to this so SDPA can use 256-token chunks


def round_up(n: int, m: int = TILE) -> int:
    return (n + m - 1) // m * m


# ----------------------------------------------------------------------------- weight layout helpers
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def fuse_qkv(wq, wk, wv) -> torch.Tensor:
    return torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(wv)], dim=1).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """[K, 2N] weight for minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p+1 = up tile p."""
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    k, n = g.shape
    assert n % tile == 0
    return torch.stack([g.view(k, n // tile, tile), u.view(k, n // tile, tile)], dim=2).reshape(k, 2 * n).contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: rotate_half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1)."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """Permute the per-head output rows of a [out, in] weight (or an [out] bias)."""
    if w.dim() == 1:
        return w.view(n_heads, head_dim)[:, perm].reshape(-1).contiguous()
    out, inp = w.shape
    assert out == n_heads * head_dim
    return w.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    """[1, 1, 32, 32] T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent-pair rotation)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[-2] == rows:
        return t
    out = torch.zeros(*t.shape[:-2], rows, t.shape[-1], dtype=t.dtype)
    out[..., : t.shape[-2], :] = t
    return out


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """Token layout of one generation on the device: [text (padded) ; image (padded) ; references (padded)]."""

    width: int
    height: int
    refs: Tuple[Tuple[int, int], ...]  # (latent h, latent w) of each reference (unpadded)
    n_txt: int  # valid text tokens

    @property
    def lat_h(self) -> int:
        return self.height // 8

    @property
    def lat_w(self) -> int:
        return self.width // 8

    @property
    def grid_h(self) -> int:
        return (self.lat_h + 1) // 2

    @property
    def grid_w(self) -> int:
        return (self.lat_w + 1) // 2

    @property
    def n_img(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n_img_pad(self) -> int:
        return round_up(self.n_img)

    @property
    def ref_grids(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(((h + 1) // 2, (w + 1) // 2) for h, w in self.refs)

    @property
    def n_ref(self) -> int:
        return sum(h * w for h, w in self.ref_grids)

    @property
    def n_ref_pad(self) -> int:
        return round_up(self.n_ref)

    @property
    def n_txt_pad(self) -> int:
        base = round_up(self.n_txt)
        spatial = self.n_img_pad + self.n_ref_pad
        return round_up(base + spatial, SDPA_MULTIPLE) - spatial

    @property
    def n_joint(self) -> int:
        return self.n_txt_pad + self.n_img_pad + self.n_ref_pad

    @property
    def img_start(self) -> int:
        return self.n_txt_pad

    @property
    def ref_start(self) -> int:
        return self.n_txt_pad + self.n_img_pad


def _grid_ids(lat_h: int, lat_w: int, index: int) -> torch.Tensor:
    """ComfyUI QwenImageTransformer2DModel.process_img ids (t_len 1, offsets 0): (index, h - h_len//2, w - w_len//2)."""
    h_len, w_len = (lat_h + 1) // 2, (lat_w + 1) // 2
    hh = torch.arange(h_len, dtype=torch.float32) - (h_len // 2)
    ww = torch.arange(w_len, dtype=torch.float32) - (w_len // 2)
    ids = torch.zeros(h_len, w_len, 3)
    ids[..., 0] = index
    ids[..., 1] = hh[:, None]
    ids[..., 2] = ww[None, :]
    return ids.reshape(-1, 3)


def image_ids(geo: Geometry) -> torch.Tensor:
    return _grid_ids(geo.lat_h, geo.lat_w, 0)


def ref_ids(geo: Geometry) -> torch.Tensor:
    if not geo.refs:
        return torch.zeros(0, 3)
    return torch.cat([_grid_ids(h, w, i + 1) for i, (h, w) in enumerate(geo.refs)])


def text_ids(geo: Geometry) -> torch.Tensor:
    start = max(geo.grid_w // 2, geo.grid_h // 2)
    return torch.arange(start, start + geo.n_txt, dtype=torch.float32)[:, None].repeat(1, 3)


def joint_ids(geo: Geometry) -> torch.Tensor:
    """[n_joint, 3] ids in device order; pad rows get id 0 (their keys are masked, their outputs dropped)."""
    out = torch.zeros(geo.n_joint, 3)
    out[: geo.n_txt] = text_ids(geo)
    out[geo.img_start: geo.img_start + geo.n_img] = image_ids(geo)
    out[geo.ref_start: geo.ref_start + geo.n_ref] = ref_ids(geo)
    return out


def rope_angles(ids: torch.Tensor, cfg: DiTConfig = DIT) -> torch.Tensor:
    """ComfyUI flux rope: per axis omega = theta ** -linspace(0, (d-2)/d, d/2) (float64), angle = pos * omega."""
    out = []
    for i, d in enumerate(cfg.axes_dims):
        scale = torch.linspace(0, (d - 2) / d, steps=d // 2, dtype=torch.float64)
        omega = 1.0 / (cfg.rope_theta ** scale)
        out.append(torch.outer(ids[:, i].to(torch.float32).to(torch.float64), omega))
    return torch.cat(out, dim=-1)  # [S, 64]


def cos_sin(ids: torch.Tensor, dtype=torch.bfloat16):
    """[S, 128] tables in the adjacent-pair layout of rotary_embedding_llama (ComfyUI apply_rope1 pairs)."""
    a = rope_angles(ids).float()
    return a.cos().repeat_interleave(2, -1).to(dtype), a.sin().repeat_interleave(2, -1).to(dtype)


def key_mask(geo: Geometry, dtype=torch.bfloat16) -> Optional[torch.Tensor]:
    """Additive [1, 1, S, S] bias masking every pad key, or None when nothing is padded."""
    valid = torch.zeros(geo.n_joint, dtype=torch.bool)
    valid[: geo.n_txt] = True
    valid[geo.img_start: geo.img_start + geo.n_img] = True
    valid[geo.ref_start: geo.ref_start + geo.n_ref] = True
    if bool(valid.all()):
        return None
    m = torch.where(valid, 0.0, MASK_NEG).to(torch.float32)
    return m.expand(geo.n_joint, geo.n_joint).reshape(1, 1, geo.n_joint, geo.n_joint).to(dtype)


# ----------------------------------------------------------------------------- latents
def latent_in(z: torch.Tensor) -> torch.Tensor:
    """Wan21 process_in: (z - mean) / std on [1, 16, 1, h, w]."""
    mean = torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD).view(1, -1, 1, 1, 1)
    return (z - mean.to(z.dtype)) / std.to(z.dtype)


def latent_out(z: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD).view(1, -1, 1, 1, 1)
    return z * std.to(z.dtype) + mean.to(z.dtype)


def pack(z: torch.Tensor) -> torch.Tensor:
    """[1, 16, 1, h, w] -> [(h/2)(w/2), 64] tokens (feature = c*4 + ph*2 + pw); odd sizes padded circularly."""
    ph, pw = z.shape[-2] % 2, z.shape[-1] % 2
    if ph or pw:
        z = F.pad(z, (0, pw, 0, ph, 0, 0), mode="circular")
    b, c, t, h, w = z.shape
    x = z.view(b, c, t, h // 2, 2, w // 2, 2).permute(0, 2, 3, 5, 1, 4, 6)
    return x.reshape(t * (h // 2) * (w // 2), c * 4)


def unpack(x: torch.Tensor, lat_h: int, lat_w: int) -> torch.Tensor:
    """[N, 64] tokens -> [1, 16, 1, lat_h, lat_w] (inverse of pack, crop of the circular pad)."""
    gh, gw = (lat_h + 1) // 2, (lat_w + 1) // 2
    z = x[: gh * gw].reshape(1, 1, gh, gw, 16, 2, 2).permute(0, 4, 1, 2, 5, 3, 6)
    return z.reshape(1, 16, 1, gh * 2, gw * 2)[..., :lat_h, :lat_w]


def init_noise(geo: Geometry, seed: int) -> torch.Tensor:
    """ComfyUI prepare_noise: torch.manual_seed(seed); randn(latent shape) float32 on CPU."""
    g = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, 16, 1, geo.lat_h, geo.lat_w), generator=g, dtype=torch.float32)


# ----------------------------------------------------------------------------- schedule
def sigmas(steps: int = SAMPLER.steps, shift: float = SAMPLER.shift) -> List[float]:
    """ModelSamplingAuraFlow (multiplier 1) table, "simple" scheduler picks, trailing 0."""
    t = torch.arange(1, 1001, 1) / 1000
    table = shift * t / (1 + (shift - 1) * t) if shift != 1.0 else t
    ss = len(table) / steps
    out = [float(table[-(1 + int(x * ss))]) for x in range(steps)]
    return [float(v) for v in torch.FloatTensor(out + [0.0])]


def euler_step(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    """CONST denoised = x - v * sigma; Euler d = (x - denoised) / sigma (float32 like k-diffusion)."""
    s = torch.tensor(sigma, dtype=torch.float32)
    denoised = x - v.float() * s
    d = (x - denoised) / s
    return x + d * (sigma_next - sigma)


# ----------------------------------------------------------------------------- timestep conditioning
def timestep_embedding(t: float, dim: int = DIT.timestep_channels) -> torch.Tensor:
    """get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000) in float32."""
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    emb = torch.tensor([t], dtype=torch.float32)[:, None] * torch.exp(exponent)[None]
    emb = 1000 * emb
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


@dataclass
class StepRows:
    """bf16 [dim] modulation rows of one sigma: per block, LayerNorm (weight = 1 + scale, bias = shift) and
    gates for the image (t), reference (t = 0) and text (t) streams, plus the final norm_out pair."""

    blocks: List[Dict[str, Dict[str, torch.Tensor]]]  # [block]["img" | "ref" | "txt"][ln1_w, ln1_b, gate1, ...]
    out_w: torch.Tensor
    out_b: torch.Tensor


class TimeConditioning:
    """t -> time_text_embed -> per-block img_mod / txt_mod rows. The modulation linears (about 6.8B of the 20B
    parameters) only ever see the timestep embedding, so they stay on the host and are evaluated once per sigma."""

    def __init__(self, ckpt, cfg: DiTConfig = DIT):
        self.ckpt, self.cfg = ckpt, cfg
        g = lambda k: ckpt.get(k, torch.float32)
        self.l1 = (g("time_text_embed.timestep_embedder.linear_1.weight"), g("time_text_embed.timestep_embedder.linear_1.bias"))
        self.l2 = (g("time_text_embed.timestep_embedder.linear_2.weight"), g("time_text_embed.timestep_embedder.linear_2.bias"))
        self.norm_out = (g("norm_out.linear.weight"), g("norm_out.linear.bias"))

    def temb(self, t: float) -> torch.Tensor:
        """[1, dim] bf16, the model runs the embedder in bf16 after the float32 sinusoid."""
        e = timestep_embedding(t).to(torch.bfloat16).float()
        h = F.linear(e, *self.l1).to(torch.bfloat16).float()
        h = F.silu(h).to(torch.bfloat16).float()
        return F.linear(h, *self.l2).to(torch.bfloat16)

    @staticmethod
    def _rows(params: torch.Tensor, d: int) -> Dict[str, torch.Tensor]:
        """[6 * dim] mod params -> LayerNorm folds and gates (mod1 = shift, scale, gate; mod2 likewise)."""
        sh1, sc1, g1, sh2, sc2, g2 = params.split(d)
        one = torch.ones((), dtype=torch.float32)
        b = lambda t: t.to(torch.bfloat16)
        return {"ln1_w": b(one + sc1), "ln1_b": b(sh1), "gate1": b(g1),
                "ln2_w": b(one + sc2), "ln2_b": b(sh2), "gate2": b(g2)}

    def rows(self, sigma_values: Sequence[float]) -> Dict[float, StepRows]:
        """Rows of every sigma in one pass over the modulation weights (each block's weights read once)."""
        d = self.cfg.dim
        temb = {s: self.temb(s) for s in sigma_values}
        zero = self.temb(0.0)
        act = {s: F.silu(t.float()).to(torch.bfloat16).float() for s, t in temb.items()}
        act0 = F.silu(zero.float()).to(torch.bfloat16).float()
        per_sigma = {s: [] for s in sigma_values}
        for i in range(self.cfg.n_layers):
            wi = self.ckpt.get(f"transformer_blocks.{i}.img_mod.1.weight", torch.float32)
            bi = self.ckpt.get(f"transformer_blocks.{i}.img_mod.1.bias", torch.float32)
            wt = self.ckpt.get(f"transformer_blocks.{i}.txt_mod.1.weight", torch.float32)
            bt = self.ckpt.get(f"transformer_blocks.{i}.txt_mod.1.bias", torch.float32)
            ref = self._rows(F.linear(act0, wi, bi)[0].to(torch.bfloat16).float(), d)
            for s in sigma_values:
                img = self._rows(F.linear(act[s], wi, bi)[0].to(torch.bfloat16).float(), d)
                txt = self._rows(F.linear(act[s], wt, bt)[0].to(torch.bfloat16).float(), d)
                per_sigma[s].append({"img": img, "ref": ref, "txt": txt})
            del wi, wt
        out = {}
        for s in sigma_values:
            emb = F.linear(act[s], *self.norm_out)[0].to(torch.bfloat16).float()
            scale, shift = emb.split(d)  # LastLayer: scale first
            out[s] = StepRows(per_sigma[s], (1.0 + scale).to(torch.bfloat16), shift.to(torch.bfloat16))
        return out


# ----------------------------------------------------------------------------- images
def to_float_image(img) -> torch.Tensor:
    """PIL image -> [1, H, W, 3] float32 in [0, 1] (ComfyUI IMAGE)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy())[None]


def lanczos_center(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """ImageScale(lanczos, crop=center): crop to the target aspect, then PIL Lanczos through uint8."""
    from PIL import Image

    s = image.movedim(-1, 1)
    old_w, old_h = s.shape[-1], s.shape[-2]
    old_aspect, new_aspect = old_w / old_h, width / height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
    s = s.narrow(-2, y, old_h - y * 2).narrow(-1, x, old_w - x * 2)
    arr = np.clip(255.0 * s[0].movedim(0, -1).numpy(), 0, 255).astype(np.uint8)
    out = Image.fromarray(arr).resize((width, height), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(out).astype(np.float32) / 255.0)[None]


def area_resize(image: torch.Tensor, total: int, multiple: int = 1) -> torch.Tensor:
    """TextEncodeQwenImageEditPlus resizes: scale to ~total pixels ("area"), optionally to a multiple of 8."""
    s = image.movedim(-1, 1)
    scale_by = math.sqrt(total / (s.shape[3] * s.shape[2]))
    if multiple == 1:
        w, h = round(s.shape[3] * scale_by), round(s.shape[2] * scale_by)
    else:
        w = round(s.shape[3] * scale_by / float(multiple)) * multiple
        h = round(s.shape[2] * scale_by / float(multiple)) * multiple
    return F.interpolate(s, size=(h, w), mode="area").movedim(1, -1)


def vision_patches(image: torch.Tensor):
    """process_qwen2vl_images: bilinear to a multiple of 28, CLIP-normalize, [grid_h * grid_w, 1176] patches."""
    c = VISION
    _, height, width, _ = image.shape
    factor = c.patch * c.merge
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    min_pixels, max_pixels = 3136, 12845056
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    img = F.interpolate(image.permute(0, 3, 1, 2)[:1], size=(h_bar, w_bar), mode="bilinear", align_corners=False)[0]
    norm = img.clone()
    for ch in range(3):
        norm[ch] = (img[ch] - c.mean[ch]) / c.std[ch]
    gh, gw = h_bar // c.patch, w_bar // c.patch
    pv = norm.unsqueeze(0).repeat(2, 1, 1, 1)
    patches = pv.reshape(1, c.temporal_patch, 3, gh // c.merge, c.merge, c.patch, gw // c.merge, c.merge, c.patch)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(gh * gw, 3 * c.temporal_patch * c.patch * c.patch), (1, gh, gw)


# ----------------------------------------------------------------------------- prompt
@dataclass
class Prompt:
    ids: List[int]  # template token ids; each IMAGE_PAD expands to the image's vision tokens
    template_end: int  # rows before this index are dropped from the conditioning


def build_prompt(tokenizer, prompt: str, n_images: int) -> Prompt:
    image_prompt = "".join(IMAGE_PROMPT.format(i + 1) for i in range(n_images))
    ids = tokenizer(TEMPLATE.format(image_prompt + prompt))["input_ids"]
    template_end, count = -1, 0
    for i, tok in enumerate(ids):
        if tok == IM_START and count < 2:
            template_end = i
            count += 1
    # ComfyUI checks the length of the expanded output, which is never shorter than the token list
    if len(ids) > template_end + 3 and ids[template_end + 1] == USER and ids[template_end + 2] == NEWLINE:
        template_end += 3
    assert sum(1 for t in ids if t == IMAGE_PAD) == n_images, "image placeholders"
    return Prompt(list(ids), template_end)


def expand(prompt: Prompt, vision_lens: Sequence[int]):
    """Expanded layout: for each token either ("tok", id) or ("img", k, row). Returns the list and the image spans
    (start, size) in the expanded sequence."""
    layout, spans, k = [], [], 0
    for tok in prompt.ids:
        if tok == IMAGE_PAD and k < len(vision_lens):
            spans.append((len(layout), vision_lens[k]))
            layout.extend(("img", k, r) for r in range(vision_lens[k]))
            k += 1
        else:
            layout.append(("tok", tok))
    return layout, spans


def mrope_positions(seq_len: int, spans: Sequence[Tuple[int, int]], grids: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
    """ComfyUI Qwen25_7BVLI.forward position ids [3, L] (no attention mask: every token is attended)."""
    if not spans:
        return torch.arange(seq_len)[None].repeat(3, 1)
    pos = torch.ones((3, seq_len), dtype=torch.long)
    offset = 0
    for i, ((start, size), grid) in enumerate(zip(spans, grids)):
        if i == 0:
            pos[:, :start] = torch.arange(0, start)
        end = start + size
        len_max = int(max(grid)) // 2
        start_next = len_max + start
        pos[:, end:] = torch.arange(start_next + offset, start_next + (seq_len - end) + offset)
        pos[0, start:end] = start + offset
        max_d = int(grid[1]) // 2
        pos[1, start:end] = torch.arange(start + offset, start + max_d + offset).unsqueeze(1).repeat(
            1, math.ceil((end - start) / max_d)).flatten(0)[: end - start]
        max_d = int(grid[2]) // 2
        pos[2, start:end] = torch.arange(start + offset, start + max_d + offset).unsqueeze(0).repeat(
            math.ceil((end - start) / max_d), 1).flatten(0)[: end - start]
        offset += len_max - (end - start)
    return pos


def te_angles(positions: torch.Tensor) -> torch.Tensor:
    """[L, 64] rope angles of the LM (frequency j uses axis t / h / w per mrope_section), float32 like ComfyUI."""
    c = TE
    inv_freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2).float() / c.head_dim))
    freqs = positions.float()[:, :, None] * inv_freq[None, None, :]  # [3, L, 64]
    axis = torch.cat([torch.full((n,), i) for i, n in enumerate(c.mrope_section)])
    return freqs[axis, :, torch.arange(c.head_dim // 2)].t().contiguous()  # [L, 64]


def te_cos_sin(positions: torch.Tensor, rows: int, dtype=torch.bfloat16):
    """[rows, 128] tables in the permuted adjacent-pair layout (pad rows: angle 0)."""
    a = te_angles(positions)
    a = pad_rows(a, rows)
    return a.cos().repeat_interleave(2, -1).to(dtype), a.sin().repeat_interleave(2, -1).to(dtype)


def apply_rope_rotate_half(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Reference LM rope: rotate_half with emb = cat(freqs, freqs)."""
    emb = torch.cat([angles, angles], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin
