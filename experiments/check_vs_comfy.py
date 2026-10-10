"""CPU check of deploy/qwenedit (host.py, ref_algo.py, vision.py) against ComfyUI's own modules.

Run with ComfyUI's venv (ComfyUI is only imported here as the behavioural oracle, nothing of it
is shipped):
    COMFY=~/ComfyUI MODELS=~/ComfyUI/models ~/ComfyUI/venv/bin/python check_vs_comfy.py
Checks (float32, CPU):
  1. prompt token ids / template_end vs QwenImageTokenizer + QwenImageTEModel's prefix logic
  2. LM mrope position ids and a tiny random Qwen2.5 LM (2 layers) vs Qwen25_7BVLI.forward
  3. two real LM layers vs ComfyUI Llama2_ (real weights, mrope with an image span)
  4. vision tower (real weights) vs Qwen2VLVisionTransformer
  5. DiT: real embeddings + block 0 (1-layer QwenImageTransformer2DModel, index_timestep_zero, odd latent sizes,
     two references) vs RefDiT(n_blocks=1)
Writes check_vs_comfy.json next to this file."""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
sys.path.insert(0, os.path.expanduser(os.environ.get("COMFY", "~/ComfyUI")))
MODELS = os.path.expanduser(os.environ.get("MODELS", "~/ComfyUI/models"))
os.environ.setdefault("QWEN_EDIT_MODELS", MODELS)
torch.set_num_threads(int(os.environ.get("THREADS", "12")))

from qwenedit import host, ref_algo, vision  # noqa: E402
from qwenedit.config import DIT, IMAGE_PROMPT, TEMPLATE, paths  # noqa: E402
from qwenedit.weights import ComfyCheckpoint, state_dict  # noqa: E402

report = {}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def maxdiff(a, b):
    return float((a.double() - b.double()).abs().max())


def save():
    (ROOT / "check_vs_comfy.json").write_text(json.dumps(report, indent=2))


def check_tokens():
    from comfy.text_encoders.qwen_image import QwenImageTokenizer
    from transformers import AutoTokenizer

    tok_dir = os.environ.get("TOKENIZER") or os.path.join(os.environ["COMFY_DIR"], "comfy/text_encoders/qwen25_tokenizer")
    ours = AutoTokenizer.from_pretrained(tok_dir)
    comfy_tok = QwenImageTokenizer()
    out = {}
    llama_template = TEMPLATE
    for prompt, n in (("Make the sky purple and add a small red kite.", 1), ("하늘을 보라색으로 바꿔 줘", 2),
                      ("Put the cat from Picture 2 on the sofa in Picture 1, keep lighting", 3), ("", 1)):
        p = host.build_prompt(ours, prompt, n)
        image_prompt = "".join(IMAGE_PROMPT.format(i + 1) for i in range(n))
        imgs = [torch.zeros(1, 8, 8, 3) for _ in range(n)]
        toks = comfy_tok.tokenize_with_weights(image_prompt + prompt, images=imgs, llama_template=llama_template)
        pairs = toks["qwen25_7b"][0]
        ids = [151655 if isinstance(t[0], dict) else int(t[0]) for t in pairs]
        # QwenImageTEModel.encode_token_weights prefix logic
        te, count = -1, 0
        for i, v in enumerate(pairs):
            e = v[0]
            if not torch.is_tensor(e) and isinstance(e, int) and e == 151644 and count < 2:
                te = i
                count += 1
        if len(pairs) > te + 3 and pairs[te + 1][0] == 872 and pairs[te + 2][0] == 198:
            te += 3
        out[f"{n}:{prompt[:20]}"] = {"ids_equal": ids == p.ids, "template_end": [te, p.template_end], "len": len(ids)}
    report["tokens"] = out
    save()


def check_lm_tiny():
    """Random tiny LM with the real rope layout: positions and outputs vs ComfyUI's forward."""
    import comfy.ops
    from comfy.text_encoders import llama

    torch.manual_seed(0)
    cfg = dict(vocab_size=64, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2)
    m = llama.Qwen25_7BVLI(cfg, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    for p in m.model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    captured = {}
    orig = llama.Llama2_.forward

    def spy(self, *a, **kw):
        captured["pos"] = kw.get("position_ids")
        return orig(self, *a, **kw)

    llama.Llama2_.forward = spy
    L = 120
    spans = [(20, 30), (60, 12)]
    grids = [(1, 10, 12), (1, 6, 8)]
    info = [{"type": "image", "index": s, "size": n, "extra": torch.tensor([g])} for (s, n), g in zip(spans, grids)]
    emb = torch.randn(1, L, 256)
    with torch.no_grad():
        out, _ = m.forward(None, embeds=emb.clone(), embeds_info=info)  # ComfyUI adds residuals in place
    llama.Llama2_.forward = orig
    pos = host.mrope_positions(L, spans, grids)
    ours_pos_equal = bool(torch.equal(pos, captured["pos"].long()))

    class Ck:
        def __init__(self, sd):
            self.sd = sd

        def get(self, k, dtype=None):
            return self.sd[k].to(dtype or torch.float32)

    sd = {"model." + k: v for k, v in m.model.state_dict().items()}
    from qwenedit.config import TextEncoderConfig

    tcfg = TextEncoderConfig(num_layers=2, hidden=256, heads=4, kv_heads=2, intermediate=512)
    ref = ref_algo.RefLM(Ck(sd), tcfg)
    mine = ref.forward(emb[0], pos)
    report["lm_tiny"] = {"positions_equal": ours_pos_equal, "pcc": pcc(mine, out[0]), "maxdiff": maxdiff(mine, out[0])}
    save()


def check_lm_real():
    import comfy.ops
    from comfy.text_encoders import llama

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    n = 2
    cfg = dict(num_hidden_layers=n)
    m = llama.Llama2_(llama.Qwen25_7BVLI_Config(**cfg), device="cpu", dtype=torch.float32,
                      ops=comfy.ops.disable_weight_init)
    sd = {}
    for k in list(m.state_dict().keys()):
        if k.startswith("embed_tokens"):
            continue
        sd[k] = ck.get("model." + k, torch.float32)
    sd["embed_tokens.weight"] = torch.zeros(m.embed_tokens.weight.shape)
    print(m.load_state_dict(sd, strict=False))
    torch.manual_seed(1)
    L = 96
    spans, grids = [(30, 42)], [(1, 12, 14)]
    pos = host.mrope_positions(L, spans, grids)
    emb = torch.randn(1, L, 3584) * 0.02
    with torch.no_grad():
        out, _ = m(None, embeds=emb.clone(), position_ids=pos.float())
    ref = ref_algo.RefLM(ck)
    mine = ref.forward(emb[0], pos, n_layers=n, final_norm=True)
    report["lm_real_2layers"] = {"pcc": pcc(mine, out[0]), "maxdiff": maxdiff(mine, out[0])}
    save()


def check_vision():
    import comfy.ops
    from comfy.text_encoders import qwen_vl

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    t0 = time.time()
    ours = vision.VisionTower(ck, dtype=torch.float32)
    ours_bf16 = vision.VisionTower(ck)
    load_s = time.time() - t0
    m = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=3584, device="cpu", dtype=torch.float32,
                                         ops=comfy.ops.disable_weight_init)
    sd = {k: ck.get("visual." + k, torch.float32) for k in m.state_dict().keys()}
    print(m.load_state_dict(sd, strict=True))
    torch.manual_seed(2)
    img = torch.rand(1, 300, 500, 3)
    vl = host.area_resize(img, 384 * 384)
    patches, grid = host.vision_patches(vl)
    cpatches, cgrid = qwen_vl.process_qwen2vl_images(vl)
    with torch.no_grad():
        ref_out = m(cpatches.float(), cgrid)
    t0 = time.time()
    mine = ours(patches, grid)
    bf = ours_bf16(patches, grid)
    report["vision"] = {"bf16_pcc": pcc(bf, ref_out), "patches_equal": bool(torch.equal(patches, cpatches)),
                        "grid": [list(grid), cgrid.tolist()],
                        "pcc": pcc(mine, ref_out), "maxdiff": maxdiff(mine, ref_out), "tokens": int(mine.shape[0]),
                        "load_s": load_s, "run_s": time.time() - t0}
    save()


def check_dit():
    import comfy.ops
    from comfy.ldm.qwen_image.model import QwenImageTransformer2DModel

    p = paths()
    ck = ComfyCheckpoint(p["dit"], p["lora"])
    m = QwenImageTransformer2DModel(num_layers=1, default_ref_method="index_timestep_zero", dtype=torch.float32,
                                    device="cpu", operations=comfy.ops.disable_weight_init)
    sd = {}
    for k in m.state_dict().keys():
        if k == "__index_timestep_zero__":
            sd[k] = torch.tensor([])
            continue
        sd[k] = ck.get(k, torch.float32)
    print(m.load_state_dict(sd, strict=True))
    torch.manual_seed(3)
    width, height = 280, 216  # latent 35 x 27: both odd -> circular pad
    refs = [(29, 41), (24, 24)]
    geo = host.Geometry(width, height, tuple(refs), n_txt=77)
    x = torch.randn(1, 16, 1, geo.lat_h, geo.lat_w)
    ref_lat = [torch.randn(1, 16, 1, h, w) for h, w in refs]
    cond = torch.randn(1, geo.n_txt, 3584)
    sigma = 0.756098
    with torch.no_grad():
        out = m(x, torch.tensor([sigma]), cond, ref_latents=ref_lat, ref_latents_method="index_timestep_zero")
    tc = host.TimeConditioning(ck)
    tc.cfg = DIT
    # restrict the per-block rows to block 0
    import qwenedit.host as H

    class OneLayer:
        n_layers = 1
        dim = DIT.dim

    tc.cfg = OneLayer
    rows = tc.rows([sigma])[sigma]
    tc.cfg = DIT
    ref = ref_algo.RefDiT(ck)
    v = ref.forward(H.pack(x), torch.cat([H.pack(r) for r in ref_lat]), cond[0], rows, geo, n_blocks=1)
    mine = H.unpack(v, geo.lat_h, geo.lat_w)
    report["dit_1block"] = {"pcc": pcc(mine, out), "maxdiff": maxdiff(mine, out), "shape": list(out.shape),
                            "out_absmean": float(out.abs().mean())}
    save()


if __name__ == "__main__":
    os.environ.setdefault("COMFY_DIR", os.path.expanduser(os.environ.get("COMFY", "~/ComfyUI")))
    which = sys.argv[1:] or ["tokens", "lm_tiny", "lm_real", "vision", "dit"]
    for name in which:
        t0 = time.time()
        try:
            globals()["check_" + name]()
        except Exception as exc:  # keep going, record the failure
            import traceback

            report[name + "_error"] = traceback.format_exc()
            save()
        print(name, round(time.time() - t0, 1), "s", flush=True)
    print(json.dumps(report, indent=2))
