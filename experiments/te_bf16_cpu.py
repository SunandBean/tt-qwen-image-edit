"""CPU: RefLM in bf16 vs the float32 reference saved by device_check te_diag (inherent bf16 gap)."""
import os, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("DEPLOY") or str(ROOT.parents[1] / "deploy"))
from qwenedit import host, ref_algo
from qwenedit.config import SAMPLER, paths
from qwenedit.pipeline import _tokenizer
from qwenedit.vision import VisionTower
from qwenedit.weights import ComfyCheckpoint
from PIL import Image
torch.set_num_threads(16)
def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double(); a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))
p = paths(); ck = ComfyCheckpoint(p["te"]); tok = _tokenizer(p["tokenizer"])
img = host.to_float_image(Image.open(ROOT / "fixtures/input.png").convert("RGB"))
patches, grid = host.vision_patches(host.area_resize(img, SAMPLER.vl_area))
emb_v = VisionTower(ck)(patches, grid)
pr = host.build_prompt(tok, "Change the season to winter with snow on the ground", 1)
layout, spans = host.expand(pr, [emb_v.shape[0]])
pos = host.mrope_positions(len(layout), spans, [grid])
from qwenedit.tt_text_encoder import QwenVLTextEncoder
emb = QwenVLTextEncoder.embeddings(type("X", (), {"ckpt": ck, "cfg": __import__("qwenedit.config", fromlist=["TE"]).TE})(), layout, [emb_v])
ref32 = torch.load(ROOT / "device-check/te_ref_f32.pt")
bf = ref_algo.RefLM(ck).forward(emb, pos, dtype=torch.bfloat16)[pr.template_end:].float()
cos = torch.nn.functional.cosine_similarity(bf, ref32, dim=-1)
print({"bf16_cpu_vs_f32_pcc": pcc(bf, ref32), "row_cos_mean": float(cos.mean()), "row_cos_min": float(cos.min()),
       "worst_rows": cos.topk(5, largest=False).indices.tolist()})
torch.save(bf, ROOT / "device-check/te_ref_bf16.pt")
