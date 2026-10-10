# SPDX-License-Identifier: Apache-2.0
"""Lazy reader for the ComfyUI single-file checkpoints: dequantizes fp8 weights and merges a LoRA.

Two fp8 layouts occur:
  * "comfy_quant" (DiT): <m>.weight float8_e4m3fn with a scalar <m>.weight_scale; the matmul runs in full
    precision on the dequantized weight (quantization metadata "full_precision_matrix_mult": true)
  * "scaled_fp8" (text encoder): <m>.weight float8_e4m3fn with a scalar <m>.scale_weight
Everything else (biases, norms, a few bf16 layers) is read as stored.

The LoRA (lora_up [out, r], lora_down [r, in], alpha) is merged like ComfyUI's LoRA patch:
W += strength * alpha / r * up @ down, on the dequantized weight in float32."""
from __future__ import annotations

from typing import Dict, Optional

import torch
from safetensors import safe_open


class ComfyCheckpoint:
    def __init__(self, path: str, lora_path: Optional[str] = None, lora_strength: float = 1.0, lora_prefix: str = ""):
        self.path, self.lora_path = path, lora_path
        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self._f.keys())
        self._lora = safe_open(lora_path, framework="pt", device="cpu") if lora_path else None
        self._lora_keys = set(self._lora.keys()) if self._lora is not None else set()
        self.lora_strength = float(lora_strength)
        self.lora_prefix = lora_prefix  # e.g. "diffusion_model." (Wan lightx2v LoRAs)

    def keys(self):
        return self._keys

    def has(self, key: str) -> bool:
        return key in self._keys

    def raw(self, key: str) -> torch.Tensor:
        return self._f.get_tensor(key)

    def _scale(self, module: str) -> Optional[torch.Tensor]:
        for suffix in ("weight_scale", "scale_weight"):
            k = f"{module}.{suffix}"
            if k in self._keys:
                return self._f.get_tensor(k).float()
        return None

    def get(self, key: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> torch.Tensor:
        """Tensor `key` dequantized (and LoRA-merged for a module weight), cast to dtype."""
        t = self._f.get_tensor(key)
        merged = False
        if key.endswith(".weight"):
            module = key[: -len(".weight")]
            scale = self._scale(module) if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else None
            if scale is not None:
                t = t.float() * scale
            delta = self.lora_delta(module)
            if delta is not None:
                t = t.float() + delta
                merged = True
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            t = t.float()
        if dtype is not None and t.dtype != dtype:
            return t.to(dtype)
        return t if merged else t.clone()

    def lora_delta(self, module: str) -> Optional[torch.Tensor]:
        if self._lora is None or self.lora_strength == 0.0:
            return None
        module = self.lora_prefix + module
        up_k, down_k = f"{module}.lora_up.weight", f"{module}.lora_down.weight"
        if up_k not in self._lora_keys:
            return None
        up = self._lora.get_tensor(up_k).float()
        down = self._lora.get_tensor(down_k).float()
        rank = down.shape[0]
        alpha_k = f"{module}.alpha"
        alpha = float(self._lora.get_tensor(alpha_k).float()) if alpha_k in self._lora_keys else float(rank)
        return (self.lora_strength * alpha / rank) * (up @ down)

    def get_rows(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        """Selected rows of a stored (unquantized) tensor, e.g. token embeddings."""
        sl = self._f.get_slice(key)
        return torch.stack([sl[i: i + 1][0] for i in rows.reshape(-1).tolist()], dim=0)


def state_dict(ckpt: ComfyCheckpoint, prefix: str, dtype=torch.float32, strip: bool = True) -> Dict[str, torch.Tensor]:
    """Dequantized tensors under `prefix` (scale / quant bookkeeping entries dropped)."""
    out = {}
    skip = (".weight_scale", ".scale_weight", ".input_scale", ".scale_input", ".comfy_quant")
    for k in sorted(ckpt.keys()):
        if not k.startswith(prefix) or k.endswith(skip):
            continue
        name = k[len(prefix):] if strip else k
        out[name] = ckpt.get(k, dtype)
    return out
