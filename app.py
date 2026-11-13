"""Turf Preview: house/yard photo -> artificial grass mockup (Hugging Face Space)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from gradio_client import Client, handle_file
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("turf-preview")

MAX_UPLOAD_BYTES = 3 * 1024 * 1024
SEC_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(self)",
    "Cache-Control": "no-store",
}

STYLES = {
    "lush-green": (
        "lush, vibrant medium-green residential artificial turf with dense, uniform "
        "blades about 1.5 inches tall and a natural slight sheen, like a premium landscape turf"
    ),
    "olive-natural": (
        "natural-looking artificial turf with mixed olive-green, sage-green, and light-green "
        "blades plus a subtle tan-brown thatch layer near the base, for a realistic, lived-in lawn look"
    ),
    "striped": (
        "two-tone green artificial turf with crisp, professional alternating light-green and "
        "dark-green mowing stripes running in straight, parallel bands across the whole lawn"
    ),
    "dark-emerald": (
        "deep, rich dark emerald-green artificial turf with dense, full blades and a formal, manicured look"
    ),
    "putting-green": (
        "short, tight, very dense golf putting-green style artificial turf in a bright, "
        "consistent green with a smooth, even surface"
    ),
}

# Fast ZeroGPU Space (duration=60s, 4-step lightning). Best uses a few more steps.
LIGHTNING_SPACE = "https://akhaliq-qwen-image-edit-2511-lightning.hf.space"
# Fallback edit Space (duration=120s).
LAMPYY_SPACE = "https://lampyy-qwen-image-edit.hf.space"

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"


def build_prompt(style_key: str) -> str:
    style = STYLES[style_key]
    return (
        "Edit this daytime photo of a real residential property. Keep the exact same house and scene: "
        "identical architecture, roof shape and pitch, roofing, siding material, the exact original exterior "
        "paint and trim colors (do not repaint or recolor anything), windows, doors, garage, chimney, porch, "
        "trees, shrubs, hedges, flower beds, mulch, rocks, walkways, patios, hardscape, curbs, driveway, "
        "street, sky, neighboring objects, framing, and camera angle, all unchanged and in the same positions. "
        "Only replace the existing lawn, grass, and bare-dirt yard areas with professionally installed "
        f"artificial grass: {style}. "
        "The turf follows the exact same lawn shape and boundaries, with clean, neat edges where it meets "
        "the walkways, driveway, curb, flower beds, and planters. Leave every non-lawn surface exactly as it is. "
        "Keep the original time of day, natural daylight, sun direction, shadows, sky, and weather exactly "
        "the same as in the input photo. "
        "Photorealistic real-estate photography, sharp detail, natural colors. Not a rendering or illustration, "
        "not a different house, no extra buildings, no added decorations, no text, no watermark, no people, "
        "no cars added."
    )


def pin_ok(provided: str | None) -> bool:
    expected = os.environ.get("ACCESS_PIN") or ""
    if not expected or provided is None:
        return False
    a = hashlib.sha256(provided.encode("utf-8")).digest()
    b = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def json_err(message: str, status: int, code: str = "") -> JSONResponse:
    body = {"error": message}
    if code:
        body["code"] = code
    return JSONResponse(body, status_code=status, headers=SEC_HEADERS)


def json_ok(data: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status, headers=SEC_HEADERS)


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or None


def client_for(src: str) -> Client:
    kwargs = {"verbose": False}
    tok = hf_token()
    if tok:
        kwargs["token"] = tok
    return Client(src, **kwargs)


def result_to_jpeg_b64(result) -> tuple[str, int, int]:
    """Accept a Gradio image result (path, dict, or nested) and return JPEG base64 + size."""
    path = None
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, dict):
        path = result.get("path") or result.get("url")
    elif isinstance(result, (str, Path)):
        path = str(result)
    if not path or not Path(path).exists():
        raise RuntimeError(f"no image path in result: {type(result)} {repr(result)[:300]}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii"), width, height


def write_upload(data: bytes, suffix: str = ".jpg") -> str:
    fd, name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(name).write_bytes(data)
    return name


def call_lightning(image_path: str, prompt: str, seed: int | None, steps: int) -> tuple:
    c = client_for(LIGHTNING_SPACE)
    return c.predict(
        image=handle_file(image_path),
        prompt=prompt,
        seed=float(seed if seed is not None else 0),
        randomize_seed=seed is None,
        true_guidance_scale=1.0,
        num_inference_steps=float(steps),
        api_name="/infer",
    ), "qwen-image-edit-2511-lightning"


def call_lampyy(image_path: str, prompt: str, seed: int | None, steps: int) -> tuple:
    c = client_for(LAMPYY_SPACE)
    return c.predict(
        prompt=prompt,
        input_image=handle_file(image_path),
        extra_images=[],
        steps=float(steps),
        seed=int(seed if seed is not None else 0),
        randomize_seed=seed is None,
        lora_scale=1.0,
        api_name="/edit",
    ), "qwen-image-edit-lampyy"


def friendly_error(err: Exception) -> JSONResponse:
    msg = str(err)
    log.error("ai_error %s", msg[:800])
    low = msg.lower()
    if "zero gpu runs limit" in low or "exceeded your zerogpu" in low:
        return json_err(
            "Today's free Hugging Face GPU allowance is used up. Try again later, or after signing in on Hugging Face.",
            429,
            "zerogpu_limit",
        )
    if "gpu duration" in low and "maximum allowed" in low:
        return json_err(
            "The free GPU time limit blocked this run. Try Saver mode, or try again in a few minutes.",
            429,
            "gpu_duration",
        )
    if "quota" in low or "429" in low or "rate" in low or "limit" in low:
        return json_err(
            "The free AI queue is busy or the daily allowance is used up. Please try again in a few minutes.",
            429,
            "quota",
        )
    if "nsfw" in low or "safety" in low or "flagged" in low:
        return json_err("The AI couldn't use this photo. Try a different photo of the yard.", 422, "safety")
    return json_err("The AI service had a hiccup. Please tap Generate again in a minute.", 502)


app = FastAPI(title="Turf Preview", docs_url=None, redoc_url=None)


@app.middleware("http")
async def add_sec_headers(request, call_next):
    resp = await call_next(request)
    for k, v in SEC_HEADERS.items():
        if k == "Cache-Control" and request.url.path in ("/", "/index.html"):
            resp.headers.setdefault("Cache-Control", "no-cache")
            continue
        resp.headers.setdefault(k, v)
    return resp


@app.get("/")
@app.get("/index.html")
def home():
    return FileResponse(INDEX, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache", **SEC_HEADERS})


@app.post("/api/check")
async def api_check(x_access_pin: str | None = Header(default="", alias="x-access-pin")):
    ok = pin_ok(x_access_pin or "")
    return json_ok({"ok": ok}, 200 if ok else 401)


@app.post("/api/generate")
async def api_generate(
    image: UploadFile = File(...),
    style: str = Form(""),
    width: str = Form("1024"),
    height: str = Form("1024"),
    quality: str = Form("best"),
    seed: str = Form(""),
    x_access_pin: str | None = Header(default="", alias="x-access-pin"),
):
    if not pin_ok(x_access_pin or ""):
        return json_err("Wrong access PIN. Tap Settings and enter the PIN again.", 401)
    if style not in STYLES:
        return json_err("Please pick a turf style.", 400)

    data = await image.read()
    if not data:
        return json_err("Please add a yard photo first.", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        return json_err("That photo is too large. Try a smaller photo.", 413)

    parsed_seed = int(seed) if seed and seed.isdigit() and len(seed) <= 9 else None
    # Saver: 4 lightning steps. Best: 8 lightning steps (still within 60s GPU).
    steps = 4 if quality == "saver" else 8
    prompt = build_prompt(style)
    tmp = write_upload(data, suffix=".jpg" if "jpeg" in (image.content_type or "") or "jpg" in (image.filename or "").lower() else ".png")
    started = time.time()
    last_err: Exception | None = None
    result = None
    model = ""
    try:
        attempts = [
            lambda: call_lightning(tmp, prompt, parsed_seed, steps),
            lambda: call_lampyy(tmp, prompt, parsed_seed, 16 if quality == "saver" else 28),
        ]
        for i, fn in enumerate(attempts):
            try:
                result, model = fn()
                last_err = None
                break
            except Exception as e:
                last_err = e
                log.warning("ai_retry attempt=%s model_err=%s", i + 1, str(e)[:400])
                low = str(e).lower()
                # Don't burn the fallback on a hard quota stop from this account.
                if "zerogpu runs limit" in low or "exceeded your zerogpu" in low:
                    break
        if last_err:
            return friendly_error(last_err)
        b64, out_w, out_h = result_to_jpeg_b64(result)
        ms = int((time.time() - started) * 1000)
        log.info("generated model=%s style=%s quality=%s ms=%s", model, style, quality, ms)
        return json_ok({"image": b64, "width": out_w, "height": out_h, "model": model, "quality": quality, "ms": ms})
    except Exception as e:
        return friendly_error(e)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@app.get("/health")
def health():
    return json_ok({"ok": True, "pin_set": bool(os.environ.get("ACCESS_PIN")), "token_set": bool(hf_token())})
