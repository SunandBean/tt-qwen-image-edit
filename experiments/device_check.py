"""P100a parity + timing check of the single-chip Qwen-Image-Edit-2511 port against the CPU reference (ref_algo,
itself checked against ComfyUI by check_vs_comfy.py). Run inside the model image (run_device.sh).
STAGES (comma-separated, default all):
  dit2   2 DiT blocks on device vs RefDiT(n_blocks=2), 512^2 output + one 512^2 reference, random text context
  te2    2 LM layers vs RefLM, random embeddings with an image span
  dit    full 60-block first step vs RefDiT (CPU, streams one block at a time)
  te     full LM on a real prompt + image vs RefLM
  e2e    full edit of fixtures/input.png at a few sizes, timings, PNGs in device-check/
Writes device-check/report.json as stages finish."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("DEPLOY") or str(ROOT.parents[1] / "deploy"))
from qwenedit import host, ref_algo  # noqa: E402
from qwenedit.config import SAMPLER, paths  # noqa: E402
from qwenedit.pipeline import Timing, close_device, dram_stats, open_device  # noqa: E402
from qwenedit.weights import ComfyCheckpoint  # noqa: E402

OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
report = {"status": "starting", "stages": {}}
torch.set_num_threads(int(os.environ.get("THREADS", "16")))


def peak_rss_gb():
    for f in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return round(int(Path(f).read_text()) / 2**30, 2)
        except OSError:
            continue
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def proc_mem():
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        k = line.split(":")[0]
        if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
            out[k] = round(int(line.split()[1]) / 2**20, 2)
    return out


def save():
    report["peak_mem_gb"] = peak_rss_gb()
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def stage(name):
    def wrap(fn):
        def run(*a, **kw):
            t0 = time.time()
            report["stages"][name] = {"status": "running"}
            save()
            try:
                res = fn(*a, **kw)
                report["stages"][name] = {"status": "ok", "s": round(time.time() - t0, 1), "peak_mem_gb": peak_rss_gb(),
                                          **(res or {})}
            except Exception:
                report["stages"][name] = {"status": "error", "s": round(time.time() - t0, 1), "error": traceback.format_exc()}
            save()
            print(name, json.dumps(report["stages"][name])[:2000], flush=True)
        return run
    return wrap


def small_geo(n_txt=77, size=512, ref=(64, 64)):
    return host.Geometry(size, size, (ref,), n_txt=n_txt)


@stage("dit2")
def check_dit2(dev):
    from qwenedit.tt_dit import QwenEditDiT

    p = paths()
    ck = ComfyCheckpoint(p["dit"], p["lora"])
    t0 = time.time()
    dit = QwenEditDiT(dev, ck, n_blocks=2)
    load_s = time.time() - t0
    sig = host.sigmas()
    tc = host.TimeConditioning(ck)

    class Two:
        n_layers, dim = 2, 3072

    tc.cfg = Two
    rows = tc.rows([sig[0], sig[2]])
    dit.load_rows(rows)
    torch.manual_seed(0)
    geo = small_geo()
    x = torch.randn(1, 16, 1, geo.lat_h, geo.lat_w)
    refl = torch.randn(1, 16, 1, *geo.refs[0])
    cond = torch.randn(geo.n_txt, 3584)
    prep = dit.prepare(geo)
    ctx = dit.context(cond, geo)
    ref_emb = dit.embed(host.pack(refl), geo.n_ref_pad)
    out = {"load_s": load_s, "dram": dram_stats(dev)}
    ref = ref_algo.RefDiT(ck)
    for s in (sig[0], sig[2]):
        t0 = time.time()
        v = dit.step(host.pack(x.to(torch.bfloat16)), ref_emb, ctx, s, prep)
        dt = time.time() - t0
        r = ref.forward(host.pack(x.to(torch.bfloat16)), host.pack(refl), cond, rows[s], geo, n_blocks=2)
        out[f"sigma_{s:.3f}"] = {"pcc": pcc(v, r), "step_s": dt, "absmean": float(r.abs().mean())}
    # taps per block for diagnosis
    taps = []
    dit.step(host.pack(x.to(torch.bfloat16)), ref_emb, ctx, sig[0], prep, taps=taps)
    rtaps = []
    ref.forward(host.pack(x.to(torch.bfloat16)), host.pack(refl), cond, rows[sig[0]], geo, n_blocks=2, taps=rtaps)
    for i, (t, (ri, rr, rt)) in enumerate(zip(taps, rtaps)):
        out[f"block{i}"] = {"img": pcc(t["img"][: geo.n_img], ri), "ref": pcc(t["ref"][: geo.n_ref], rr),
                            "txt": pcc(t["txt"][: geo.n_txt], rt)}
    return out


@stage("te2")
def check_te2(dev):
    from qwenedit.tt_text_encoder import QwenVLTextEncoder

    ck = ComfyCheckpoint(paths()["te"])
    te = QwenVLTextEncoder(dev, ck, n_layers=2)
    torch.manual_seed(1)
    L = 300
    spans, grids = [(40, 196)], [(1, 28, 28)]
    pos = host.mrope_positions(L, spans, grids)
    emb = (torch.randn(L, 3584) * 0.02).to(torch.bfloat16)
    t0 = time.time()
    mine = te.encode(emb, pos)
    dt = time.time() - t0
    ref = ref_algo.RefLM(ck).forward(emb.float(), pos, n_layers=2, final_norm=False)
    return {"pcc": pcc(mine, ref), "s": dt}


def fixture_images():
    from PIL import Image

    fx = ROOT / "fixtures"
    names = sorted(p for p in fx.glob("input*.png"))
    return [Image.open(p).convert("RGB") for p in names]


@stage("te")
def check_te(dev):
    from qwenedit.pipeline import _tokenizer
    from qwenedit.tt_text_encoder import QwenVLTextEncoder
    from qwenedit.vision import VisionTower

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    te = QwenVLTextEncoder(dev, ck)
    vt = VisionTower(ck)
    tok = _tokenizer(p["tokenizer"])
    img = host.to_float_image(fixture_images()[0])
    patches, grid = host.vision_patches(host.area_resize(img, SAMPLER.vl_area))
    emb_v = vt(patches, grid)
    pr = host.build_prompt(tok, os.environ.get("PROMPT", "Change the season to winter with snow on the ground"), 1)
    layout, spans = host.expand(pr, [emb_v.shape[0]])
    pos = host.mrope_positions(len(layout), spans, [grid])
    emb = te.embeddings(layout, [emb_v])
    t0 = time.time()
    mine = te.encode(emb, pos, keep_from=pr.template_end)
    dt = time.time() - t0
    ref = ref_algo.RefLM(ck).forward(emb.float(), pos)[pr.template_end:]
    torch.save({"cond_tt": mine, "cond_ref": ref.to(torch.bfloat16)}, OUT / "te_cond.pt")
    return {"pcc": pcc(mine, ref), "rows": int(mine.shape[0]), "tokens": len(layout), "s": dt, "dram": dram_stats(dev)}


@stage("dit")
def check_dit(dev):
    from qwenedit.tt_dit import QwenEditDiT

    p = paths()
    ck = ComfyCheckpoint(p["dit"], p["lora"])
    t0 = time.time()
    dit = QwenEditDiT(dev, ck)
    load_s = time.time() - t0
    sig = host.sigmas()
    t0 = time.time()
    rows = host.TimeConditioning(ck).rows(sig[:-1])
    rows_s = time.time() - t0
    dit.load_rows(rows)
    out = {"load_s": load_s, "rows_s": rows_s, "dram": dram_stats(dev)}
    torch.manual_seed(0)
    geo = small_geo(n_txt=160)
    x = torch.randn(1, 16, 1, geo.lat_h, geo.lat_w)
    refl = torch.randn(1, 16, 1, *geo.refs[0])
    cond = torch.randn(geo.n_txt, 3584) * 3
    prep = dit.prepare(geo)
    ctx = dit.context(cond, geo)
    ref_emb = dit.embed(host.pack(refl), geo.n_ref_pad)
    times = []
    for _ in range(3):
        t0 = time.time()
        v = dit.step(host.pack(x.to(torch.bfloat16)), ref_emb, ctx, sig[0], prep)
        times.append(time.time() - t0)
    out["step_s"] = times
    t0 = time.time()
    r = ref_algo.RefDiT(ck).forward(host.pack(x.to(torch.bfloat16)), host.pack(refl), cond, rows[sig[0]], geo)
    out["ref_s"] = time.time() - t0
    out["pcc_step0"] = pcc(v, r)
    return out


@stage("e2e")
def check_e2e(dev):
    from qwenedit.pipeline import QwenEditTT

    t0 = time.time()
    pipe = QwenEditTT(dev, host_threads=int(os.environ.get("THREADS", "16")))
    out = {"load_s": time.time() - t0, "dram": dram_stats(dev), "cases": {}, "mem_after_load": proc_mem()}
    imgs = fixture_images()
    prompt = os.environ.get("PROMPT", "Change the season to winter with snow on the ground")
    sizes = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("SIZES", "1024x1024,1024x1024,832x1216").split(",")]
    for i, (w, h) in enumerate(sizes):
        t0 = time.time()
        image, timing = pipe.generate(prompt, w, h, 42, imgs[:1])
        name = f"e2e_{i}_{w}x{h}.png"
        image.save(OUT / name)
        out["cases"][name] = {"total_s": time.time() - t0, **{k: round(v, 2) for k, v in timing.values.items()},
                              "mem": proc_mem()}
        report["stages"]["e2e"] = {"status": "running", **out}
        save()
    if len(imgs) > 1:
        t0 = time.time()
        image, timing = pipe.generate(os.environ.get("PROMPT2", "Put the object from Picture 2 into Picture 1"),
                                      1024, 1024, 7, imgs[:3])
        image.save(OUT / "e2e_multi.png")
        out["cases"]["e2e_multi.png"] = {"total_s": time.time() - t0, **{k: round(v, 2) for k, v in timing.values.items()}}
    out["dram_end"] = dram_stats(dev)
    return out


@stage("te_diag")
def check_te_diag(dev):
    """TE precision variants on the real prompt vs float32 RefLM; also CPU bf16 RefLM (inherent bf16 gap)."""
    import ttnn
    from qwenedit.pipeline import _tokenizer
    from qwenedit.tt_text_encoder import QwenVLTextEncoder, TEPrecision
    from qwenedit.vision import VisionTower

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    vt = VisionTower(ck)
    tok = _tokenizer(p["tokenizer"])
    img = host.to_float_image(fixture_images()[0])
    patches, grid = host.vision_patches(host.area_resize(img, SAMPLER.vl_area))
    emb_v = vt(patches, grid)
    pr = host.build_prompt(tok, os.environ.get("PROMPT", "Change the season to winter with snow on the ground"), 1)
    layout, spans = host.expand(pr, [emb_v.shape[0]])
    pos = host.mrope_positions(len(layout), spans, [grid])
    out = {}
    ref = None
    ref_bf16 = torch.load(OUT / "te_ref_bf16.pt") if (OUT / "te_ref_bf16.pt").exists() else None
    variants = (("bfp8_hifi2", TEPrecision()), ("bfp8_hifi4", TEPrecision(mm_fidelity=ttnn.MathFidelity.HiFi4)),
                ("bf16_hifi4", TEPrecision(weight_dtype=ttnn.bfloat16, mm_fidelity=ttnn.MathFidelity.HiFi4)))
    for name, prec in variants:
        te = QwenVLTextEncoder(dev, ck, prec)
        emb = te.embeddings(layout, [emb_v])
        if ref is None:
            ref = ref_algo.RefLM(ck).forward(emb.float(), pos)[pr.template_end:]
            torch.save(ref, OUT / "te_ref_f32.pt")
        taps = []
        mine = te.encode(emb, pos, keep_from=pr.template_end, taps=taps).float()
        if name == variants[0][0]:
            rtaps = []
            ref_algo.RefLM(ck).forward(emb.float(), pos, taps=rtaps)
            out["per_layer_pcc"] = [round(pcc(a, b), 5) for a, b in zip(taps, rtaps)]
            out["per_layer_rowcos_min"] = [round(float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min()), 4)
                                           for a, b in zip(taps, rtaps)]
        cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
        big = ref.abs().amax(0).topk(8).indices
        keep = torch.ones(ref.shape[1], dtype=torch.bool); keep[big] = False
        out[name] = {"pcc": pcc(mine, ref), "row_cos_mean": float(cos.mean()), "row_cos_min": float(cos.min()),
                     "pcc_wo_top8_channels": pcc(mine[:, keep], ref[:, keep]), "top_channels": big.tolist(),
                     "ref_absmax": float(ref.abs().max()), "worst_rows": cos.topk(5, largest=False).indices.tolist(),
                     "pcc_vs_cpu_bf16": pcc(mine, ref_bf16) if ref_bf16 is not None else None}
        report["stages"]["te_diag"] = {"status": "running", **out}
        save()
        del te
        ttnn.synchronize_device(dev)
    return out


@stage("te_rows")
def check_te_rows(dev):
    """Layer-0 rows: which tokens are wrong, with mrope positions and with plain 1-D positions."""
    from qwenedit.pipeline import _tokenizer
    from qwenedit.tt_text_encoder import QwenVLTextEncoder
    from qwenedit.vision import VisionTower

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    vt = VisionTower(ck)
    tok = _tokenizer(p["tokenizer"])
    img = host.to_float_image(fixture_images()[0])
    patches, grid = host.vision_patches(host.area_resize(img, SAMPLER.vl_area))
    emb_v = vt(patches, grid)
    pr = host.build_prompt(tok, "Change the season to winter with snow on the ground", 1)
    layout, spans = host.expand(pr, [emb_v.shape[0]])
    te = QwenVLTextEncoder(dev, ck, n_layers=1)
    emb = te.embeddings(layout, [emb_v])
    out = {"spans": spans, "L": len(layout)}
    for name, pos in (("mrope", host.mrope_positions(len(layout), spans, [grid])),
                      ("plain", torch.arange(len(layout))[None].repeat(3, 1))):
        mine = te.encode(emb, pos).float()
        ref = ref_algo.RefLM(ck).forward(emb.float(), pos, n_layers=1, final_norm=False)
        cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
        bad = (cos < 0.99).nonzero().flatten().tolist()
        out[name] = {"pcc": pcc(mine, ref), "bad_rows": bad[:40], "n_bad": len(bad),
                     "worst": [(int(i), round(float(cos[i]), 4)) for i in cos.topk(8, largest=False).indices],
                     "ref_norm_bad": [round(float(ref[i].norm()), 2) for i in bad[:8]],
                     "mine_norm_bad": [round(float(mine[i].norm()), 2) for i in bad[:8]]}
    return out


@stage("te_ops")
def check_te_ops(dev):
    """Layer 0 op by op: each device op is compared with torch applied to the device op's own inputs."""
    import ttnn
    import torch.nn.functional as F
    from qwenedit.pipeline import _tokenizer
    from qwenedit.tt_text_encoder import QwenVLTextEncoder, SEQ_MULTIPLE
    from qwenedit.vision import VisionTower
    from qwenedit.config import TE as c

    p = paths()
    ck = ComfyCheckpoint(p["te"])
    tok = _tokenizer(p["tokenizer"])
    pr = host.build_prompt(tok, "Change the season to winter with snow on the ground", 1)
    vt = VisionTower(ck)
    img = host.to_float_image(fixture_images()[0])
    patches, grid = host.vision_patches(host.area_resize(img, SAMPLER.vl_area))
    emb_v = vt(patches, grid)
    layout, spans = host.expand(pr, [emb_v.shape[0]])
    te = QwenVLTextEncoder(dev, ck, n_layers=1)
    emb = te.embeddings(layout, [emb_v])
    L = emb.shape[0]
    Lp = host.round_up(L, SEQ_MULTIPLE)
    pos = host.mrope_positions(L, spans, [grid])
    T = lambda t: ttnn.to_torch(t).float()
    lay = te.layers[0]
    out = {}
    h = ttnn.from_torch(host.pad_rows(emb, Lp).reshape(1, 1, Lp, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    x = te._rms(h, lay.ln1)
    g = lambda k: ck.get(f"model.layers.0.{k}", torch.float32)
    xr = ref_algo._rms(T(h)[0, 0], g("input_layernorm.weight"), c.rms_eps)
    out["rms"] = pcc(T(x)[0, 0, :L], xr[:L])
    qkv = te._linear(x, lay.wqkv, lay.bqkv)
    perm = te.perm
    wq = host.permute_heads_rows(g("self_attn.q_proj.weight"), c.heads, c.head_dim, perm)
    bq = host.permute_heads_rows(g("self_attn.q_proj.bias"), c.heads, c.head_dim, perm)
    wk = host.permute_heads_rows(g("self_attn.k_proj.weight"), c.kv_heads, c.head_dim, perm)
    bk = host.permute_heads_rows(g("self_attn.k_proj.bias"), c.kv_heads, c.head_dim, perm)
    xin = T(x)[0, 0]
    qkv_ref = torch.cat([F.linear(xin, wq, bq), F.linear(xin, wk, bk), F.linear(xin, g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias"))], -1)
    qd = T(qkv).reshape(Lp, -1)
    out["qkv"] = pcc(qd[:L], qkv_ref[:L])
    out["qkv_q_absmax"] = float(qkv_ref[:L, :3584].abs().max())
    out["qkv_k_absmax"] = float(qkv_ref[:L, 3584:4096].abs().max())
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.kv_heads, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cos, sin = host.te_cos_sin(pos, Lp)
    tab = lambda t: ttnn.from_torch(t.reshape(1, 1, Lp, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    cos_t, sin_t = tab(cos), tab(sin)
    qr = ttnn.experimental.rotary_embedding_llama(q, cos_t, sin_t, te.trans_mat, is_decode_mode=False, compute_kernel_config=te.ck_norm)
    kr = ttnn.experimental.rotary_embedding_llama(k, cos_t, sin_t, te.trans_mat, is_decode_mode=False, compute_kernel_config=te.ck_norm)
    qh, kh, vh = T(q)[0], T(k)[0], T(v)[0]
    qr_ref = host.apply_rope_adjacent(qh, cos.float(), sin.float())
    kr_ref = host.apply_rope_adjacent(kh, cos.float(), sin.float())
    out["rope_q"] = pcc(T(qr)[0][:, :L], qr_ref[:, :L])
    out["rope_k"] = pcc(T(kr)[0][:, :L], kr_ref[:, :L])
    for name, fid in (("sdpa_hifi4", ttnn.MathFidelity.HiFi4), ("sdpa_hifi2", ttnn.MathFidelity.HiFi2)):
        for chunk in (128, 64, 32):
            ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
            cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=te.grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False)
            a = ttnn.transformer.scaled_dot_product_attention(qr, kr, vh_t if False else v, is_causal=True, scale=c.head_dim ** -0.5, program_config=cfg, compute_kernel_config=ckc)
            qq, kk, vv = T(qr)[0], T(kr)[0], vh
            rep = c.heads // c.kv_heads
            kk, vv = kk.repeat_interleave(rep, 0), vv.repeat_interleave(rep, 0)
            mask = torch.ones(Lp, Lp, dtype=torch.bool).tril()
            aref = F.scaled_dot_product_attention(qq[None], kk[None], vv[None], attn_mask=mask)[0]
            ad = T(a)[0]
            rc = F.cosine_similarity(ad[:, :L].transpose(0, 1).reshape(L, -1), aref[:, :L].transpose(0, 1).reshape(L, -1), dim=-1)
            out[f"{name}_c{chunk}"] = {"pcc": pcc(ad[:, :L], aref[:, :L]), "row_cos_min": float(rc.min()),
                                       "worst": rc.topk(4, largest=False).indices.tolist()}
            logits = (qq[:, :L] @ kk[:, :L].transpose(1, 2)) * c.head_dim ** -0.5
            out["logit_absmax"] = float(logits.abs().max())
            ttnn.deallocate(a)
    return out


def main():
    stages = os.environ.get("STAGES", "dit2,te2,te,dit,e2e").split(",")
    dev = open_device()
    report["status"] = "running"
    save()
    try:
        for s in stages:
            globals()["check_" + s](dev)
            import ttnn

            ttnn.synchronize_device(dev)
            dev.clear_program_cache()
        report["status"] = "done"
    finally:
        save()
        close_device(dev)


if __name__ == "__main__":
    main()
