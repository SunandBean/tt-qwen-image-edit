"""Run the same edit through the reference ComfyUI setup (GPU) for an output-level comparison with the P100a port.

Builds the exact reference `qwen_edit` graph (plus SaveLatent), submits it to the
ComfyUI API, waits, and copies the PNG + latent next to this file (comfy-parity/). Host python, stdlib only:
    COMFY_URL=http://127.0.0.1:8190 COMFY_IN=<ComfyUI input dir> COMFY_OUT=<ComfyUI output dir> \
      python3 comfy_parity.py <prompt> <width> <height> <seed> fixtures/input.png [fixtures/input2.png ...]"""
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8190")
IN = Path(os.path.expanduser(os.environ["COMFY_IN"]))
OUTDIR = Path(os.path.expanduser(os.environ["COMFY_OUT"]))
DEST = ROOT / "comfy-parity"


def graph(prompt, names, width, height, seed, prefix):
    nodes = {}

    def add(cls, **inputs):
        key = str(len(nodes) + 1)
        nodes[key] = {"class_type": cls, "inputs": inputs}
        return key

    o = lambda k, s=0: [k, s]
    unet = add("UNETLoader", unet_name="qwen_image_edit_2511_fp8mixed.safetensors", weight_dtype="default")
    model = add("ModelSamplingAuraFlow", model=o(unet), shift=3.1)
    model = add("CFGNorm", model=o(model), strength=1.0)
    model = add("LoraLoaderModelOnly", model=o(model), lora_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
                strength_model=1.0)
    clip = add("CLIPLoader", clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")
    vae = add("VAELoader", vae_name="qwen_image_vae.safetensors")
    refs = {}
    base = None
    for i, name in enumerate(names[:3], start=1):
        loaded = add("LoadImage", image=name)
        if i == 1:
            loaded = add("ImageScale", image=o(loaded), upscale_method="lanczos", width=width, height=height, crop="center")
            base = loaded
        refs[f"image{i}"] = o(loaded)
    pos = add("TextEncodeQwenImageEditPlus", clip=o(clip), prompt=prompt, vae=o(vae), **refs)
    neg = add("TextEncodeQwenImageEditPlus", clip=o(clip), prompt="", vae=o(vae), **refs)
    pos = add("FluxKontextMultiReferenceLatentMethod", conditioning=o(pos), reference_latents_method="index_timestep_zero")
    neg = add("FluxKontextMultiReferenceLatentMethod", conditioning=o(neg), reference_latents_method="index_timestep_zero")
    latent = add("VAEEncode", pixels=o(base), vae=o(vae))
    sampled = add("KSampler", model=o(model), seed=seed, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                  positive=o(pos), negative=o(neg), latent_image=o(latent), denoise=1.0)
    add("SaveLatent", samples=o(sampled), filename_prefix=f"latents/{prefix}")
    image = add("VAEDecode", samples=o(sampled), vae=o(vae))
    add("SaveImage", images=o(image), filename_prefix=prefix)
    return nodes


def main():
    prompt, width, height, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    names = []
    for p in sys.argv[5:]:
        name = f"qwenedit_parity_{Path(p).name}"
        shutil.copy(ROOT / p, IN / name)
        names.append(name)
    prefix = f"qwenedit_parity_{uuid.uuid4().hex[:8]}"
    body = json.dumps({"prompt": graph(prompt, names, width, height, seed, prefix), "client_id": "qwenedit-parity"}).encode()
    t0 = time.time()
    req = urllib.request.Request(URL + "/prompt", data=body, headers={"Content-Type": "application/json"})
    pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    while True:
        hist = json.loads(urllib.request.urlopen(f"{URL}/history/{pid}").read())
        if pid in hist and hist[pid].get("status", {}).get("completed"):
            break
        if pid in hist and hist[pid].get("status", {}).get("status_str") == "error":
            raise SystemExit(json.dumps(hist[pid]["status"])[:2000])
        time.sleep(1)
    elapsed = time.time() - t0
    DEST.mkdir(exist_ok=True)
    tag = f"{width}x{height}_s{seed}_n{len(names)}"
    for f in OUTDIR.rglob(prefix + "*"):
        shutil.copy(f, DEST / (tag + f.suffix))
    for n in names:
        (IN / n).unlink(missing_ok=True)
    meta = {"prompt": prompt, "width": width, "height": height, "seed": seed, "images": sys.argv[5:], "elapsed_s": elapsed,
            "prompt_id": pid}
    (DEST / (tag + ".json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
