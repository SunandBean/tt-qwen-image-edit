#!/usr/bin/env python3
"""Staging run of the release model service with the qwenedit backend on the P100a, beside the running one.

The host holds the shared card lease for the whole run (`p100a_lease.py run`, from $LEASE_TOOLS), so the
running model service yields the card when idle; inside the container tt_service takes a private lease
directory. The staged
service listens on 127.0.0.1:20015 and gets real /predict requests (1 and 3 images, other sizes, contract
errors). Results in stage/report.json and stage/*.png. Host python, stdlib only:
    IMAGE=<tt-metal image> LEASE_TOOLS=<dir holding p100a_lease.py> python3 stage_service.py"""
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPLOY = ROOT.parents[1] / "deploy"
HOME = Path.home()
# The directory holding p100a_lease.py. No default: it lived outside this tree, so a clone has to say
# where it is rather than silently mount a path that may not exist.
LEASE_TOOLS = Path(os.environ["LEASE_TOOLS"])
OUT = ROOT / "stage"
PORT = 20015
NAME = "qwenedit-stage"
IMAGE = os.environ["IMAGE"]  # the tt-metal / ttnn image the port was built against; no portable default
TOKENIZER = "/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
report = {"cases": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def call(path, body=None, timeout=1200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def wait_ready(bound=1200):
    deadline = time.time() + bound
    while time.time() < deadline:
        try:
            status, body = call("/health", timeout=5)
            if body.get("status") == "ok":
                return body
            if body.get("status") == "error":
                raise RuntimeError(body)
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise TimeoutError("staged service did not become ready")


def main():
    OUT.mkdir(exist_ok=True)
    lease = OUT / "lease"
    lease.mkdir(exist_ok=True)
    os.chmod(lease, 0o777)
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    docker = ["docker", "run", "-d", "--name", NAME, "--ipc", "host", "--device", "/dev/tenstorrent",
              "--memory", "16g", "--memory-swap", "16g", "-p", f"127.0.0.1:{PORT}:20000",
              "-v", "/dev/hugepages-1G:/dev/hugepages-1G", "-v", f"{HOME}/.cache/huggingface:/hf:ro",
              "-v", f"{HOME}/ComfyUI/models:/comfy:ro", "-v", f"{DEPLOY}:/service:ro",
              "-v", f"{LEASE_TOOLS}:/p100a:ro", "-v", f"{lease}:/run/lock/p100a",
              "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
              "-e", "TT_METAL_VISIBLE_DEVICES=0", "-e", "MESH_DEVICE=P100", "-e", "PHOTO_TT_BACKENDS=qwenedit",
              "-e", "QWEN_EDIT_MODELS=/comfy", "-e", f"QWEN_EDIT_TOKENIZER={TOKENIZER}",
              "-e", "PHOTO_MAX_DIMENSION=1920", "-e", "PHOTO_DIMENSION_STEP=8", "-e", "PHOTO_MAX_PIXELS=2088960",
              "-e", "PHOTO_MODEL_DEFAULT_SIZE=1024", IMAGE,
              "python", "-m", "uvicorn", "tt_service:app", "--app-dir", "/service", "--host", "0.0.0.0", "--port",
              "20000", "--workers", "1", "--lifespan", "on"]
    t0 = time.time()
    subprocess.run(docker, check=True, capture_output=True)
    try:
        report["health"] = wait_ready()
        report["startup_s"] = round(time.time() - t0, 1)
        save()
        fx = ROOT / "fixtures"
        cases = [
            ("one_1024", {"prompt": "Change the season to winter with snow on the ground", "model": "qwenedit",
                          "width": 1024, "height": 1024, "seed": 42, "num_steps": 4, "images": [b64(fx / "input.png")]}),
            ("one_1360x768", {"prompt": "Make it a rainy night with wet reflections", "model": "qwenedit",
                              "width": 1360, "height": 768, "seed": 3, "images": [b64(fx / "input.png")]}),
            ("three_1024", {"prompt": "Put the cat from Picture 2 on the rocks of Picture 1 under the sky of Picture 3",
                            "model": "qwenedit", "width": 1024, "height": 1024, "seed": 9,
                            "images": [b64(fx / "input.png"), b64(fx / "input2.png"), b64(fx / "input3.png")]}),
            ("no_image_422", {"prompt": "x", "model": "qwenedit", "width": 1024, "height": 1024}),
            ("wrong_steps_422", {"prompt": "x", "model": "qwenedit", "num_steps": 8, "images": [b64(fx / "input.png")]}),
        ]
        for name, body in cases:
            t1 = time.time()
            status, out = call("/predict", body)
            entry = {"status": status, "s": round(time.time() - t1, 1)}
            if status == 200:
                (OUT / f"{name}.png").write_bytes(base64.b64decode(out.pop("image")))
                entry.update({k: out[k] for k in ("width", "height", "model", "weights_revision", "license",
                                                  "reference_count", "timing_ms") if k in out})
            else:
                entry["detail"] = out.get("detail")
            report["cases"][name] = entry
            save()
            print(name, json.dumps(entry)[:600], flush=True)
        report["health_after"] = call("/health")[1]
    finally:
        logs = subprocess.run(["docker", "logs", "--tail", "200", NAME], capture_output=True, text=True)
        (OUT / "service.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "stop", "-t", "60", NAME], capture_output=True)
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
        save()


if __name__ == "__main__":
    sys.exit(main())
