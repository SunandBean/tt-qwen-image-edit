# SPDX-License-Identifier: Apache-2.0
"""HTTP server for Qwen-Image-Edit-2511 on one Blackhole p100a.

    uvicorn tt_qwen_image_edit.server:app --host 0.0.0.0 --port 20000

One model, one card, one process. Requests are serialised: the card runs one image at a time.
"""
from __future__ import annotations

import base64
import io
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import HF_REPO, LICENSE, SAMPLER
from .device import close_device, dram_stats, open_device
from .pipeline import QwenEditTT

MIN_DIM, MAX_DIM, DIM_STEP, MAX_PIXELS = 512, 1920, 8, 1920 * 1088
MAX_IMAGE_B64 = 14_000_000
TURN_WAIT_S = float(os.environ.get("QWEN_EDIT_TURN_WAIT_S", "900"))

STATE: dict = {"status": "loading", "error": None, "generating": False, "load_s": None, "dram": None}
MODEL: Optional[QwenEditTT] = None
DEVICE = None
LOCK = Lock()


def _validate_dimension(value: int, axis: str) -> None:
    if not MIN_DIM <= value <= MAX_DIM:
        raise ValueError(f"{axis} must be between {MIN_DIM} and {MAX_DIM}")
    if value % DIM_STEP:
        raise ValueError(f"{axis} must be a multiple of {DIM_STEP}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global MODEL, DEVICE
    try:
        DEVICE = open_device()
        MODEL = QwenEditTT(DEVICE)
        STATE.update(status="ok", load_s=round(MODEL.load_s, 2), dram=dram_stats(DEVICE))
    except Exception as exc:  # the card never came up; /health says why
        STATE.update(status="error", error=str(exc))
    try:
        yield
    finally:
        MODEL = None
        if DEVICE is not None:
            close_device(DEVICE)


app = FastAPI(title="Qwen-Image-Edit-2511 on p100a", lifespan=lifespan, docs_url=None, redoc_url=None)


class Request(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt: str = Field(min_length=1, max_length=4000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    width: int = 1024
    height: int = 1024
    num_steps: Optional[int] = None
    images: Optional[List[str]] = None

    @field_validator("images")
    @classmethod
    def check_images(cls, values):
        if not values:
            return None
        from PIL import Image

        for value in values:
            if len(value) > MAX_IMAGE_B64:
                raise ValueError("Image is too large")
            try:
                with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as img:
                    if img.format not in ("PNG", "JPEG", "WEBP") or img.width * img.height > 16_000_000 \
                            or getattr(img, "n_frames", 1) != 1:
                        raise ValueError("Unsupported image format or dimensions")
                    img.verify()
            except Exception as exc:
                raise ValueError("Image cannot be decoded") from exc
        return values


def _check(request: Request) -> None:
    """Contract errors raise ValueError -> 422, without touching the card."""
    if request.num_steps is not None and request.num_steps != SAMPLER.steps:
        raise ValueError(f"{HF_REPO} runs {SAMPLER.steps} steps")
    if not request.images:
        raise ValueError(f"{HF_REPO} edits images: send 1 to {SAMPLER.max_images} image(s)")
    if len(request.images) > SAMPLER.max_images:
        raise ValueError(f"At most {SAMPLER.max_images} image(s) are supported")
    _validate_dimension(request.width, "width")
    _validate_dimension(request.height, "height")
    if request.width * request.height > MAX_PIXELS:
        raise ValueError(f"At most {MAX_PIXELS} pixels")


@app.get("/health")
def health():
    return dict(STATE)


@app.get("/info")
def info():
    return dict(STATE) | {
        "model": HF_REPO,
        "lora": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "license": LICENSE,
        "device": "Tenstorrent Blackhole p100a",
        "num_steps": SAMPLER.steps,
        "max_images": SAMPLER.max_images,
        "resolution_limits": {"min": MIN_DIM, "max": MAX_DIM, "step": DIM_STEP, "max_pixels": MAX_PIXELS},
    }


@app.post("/predict")
def predict(request: Request):
    from PIL import Image

    if not request.prompt.strip():
        raise HTTPException(400, "Empty prompt")
    try:
        _check(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if STATE["status"] != "ok" or MODEL is None:
        raise HTTPException(503, STATE["error"] or "Model is not ready")
    if not LOCK.acquire(timeout=TURN_WAIT_S):
        raise HTTPException(409, "Another image is in progress")
    try:
        STATE["generating"] = True
        images = [Image.open(io.BytesIO(base64.b64decode(v))) for v in request.images]
        t0 = time.perf_counter()
        out, timing = MODEL.generate(request.prompt, request.width, request.height, request.seed, images)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return {
            "image": base64.b64encode(buf.getvalue()).decode(),
            "width": out.width,
            "height": out.height,
            "model": HF_REPO,
            "license": LICENSE,
            "seed": request.seed,
            "num_steps": SAMPLER.steps,
            "reference_count": len(images),
            "timing_ms": {k: round(v * 1000, 2) for k, v in timing.values.items()}
                         | {"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
        }
    except Exception as exc:
        STATE.update(status="error", error=str(exc))
        raise HTTPException(503, str(exc)) from exc
    finally:
        STATE["generating"] = False
        LOCK.release()
