"""Vision: screenshots, image files, camera, and questions about them.

Every image carries where it came from and when it was captured, plus a basic
quality check (blank, very dark, tiny), so answers can say what was actually
seen and how fresh it is. Analysis uses the configured "vision" model role
(local by default).
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from pathlib import Path
from typing import Any

from jarvis.capabilities.base import Tool, ToolContext, ToolResult
from jarvis.core.permissions import Risk
from jarvis.models.base import ImageInput, Message

MAX_EDGE = 1568  # keep images within what vision models handle well


def _quality(img: Any) -> list[str]:
    from PIL import ImageStat  # noqa: PLC0415

    notes = []
    w, h = img.size
    if w < 64 or h < 64:
        notes.append("very small image")
    stat = ImageStat.Stat(img.convert("L"))
    mean, std = stat.mean[0], stat.stddev[0]
    if std < 4:
        notes.append("nearly uniform (blank or covered)")
    if mean < 18:
        notes.append("very dark")
    if mean > 245 and std < 10:
        notes.append("washed out")
    return notes


def _encode(img: Any, source: str, captured_at: float, save_dir: Path | None) -> tuple[ImageInput, dict[str, Any]]:
    img = img.convert("RGB")
    orig = img.size
    img.thumbnail((MAX_EDGE, MAX_EDGE))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    saved = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        saved = save_dir / f"{source.replace(':', '-')}-{time.strftime('%Y%m%d-%H%M%S')}.png"
        saved.write_bytes(data)
        # retention: keep the newest 50 captures
        for old in sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)[:-50]:
            old.unlink(missing_ok=True)
    meta = {"source": source, "captured_at": captured_at, "size": list(orig), "sent_size": list(img.size),
            "quality": _quality(img), "saved": str(saved) if saved else None}
    return ImageInput(data_b64=base64.b64encode(data).decode(), media_type="image/png", source=source, captured_at=captured_at,
                      width=img.size[0], height=img.size[1]), meta


def capture_screen(monitor: int = 1) -> tuple[Any, float]:
    import mss  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with mss.mss() as sct:
        mons = sct.monitors
        if monitor < 0 or monitor >= len(mons):
            raise ValueError(f"monitor {monitor} does not exist (0 = all screens, 1..{len(mons) - 1})")
        shot = sct.grab(mons[monitor])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"), time.time()


def _available_screen() -> tuple[bool, str]:
    try:
        import mss  # noqa: F401, PLC0415
        from PIL import Image  # noqa: F401, PLC0415
        return True, ""
    except ImportError as e:
        return False, f"{e.name} is not installed (pip install 'jarvis-core[vision]')"


def _available_camera() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401, PLC0415
        return True, ""
    except ImportError:
        return False, "camera capture needs OpenCV (pip install 'jarvis-core[camera]')"


async def analyze(ctx: ToolContext, image: ImageInput, meta: dict[str, Any], question: str) -> ToolResult:
    age = max(0.0, time.time() - meta["captured_at"])
    quality = f" Image quality notes: {', '.join(meta['quality'])}." if meta["quality"] else ""
    prompt = [
        Message(role="system", content="You describe images accurately. Only report what is visible. If something is unreadable or "
                                       "uncertain, say so. Never guess at text you cannot read."),
        Message(role="user", content=f"{question}\n(Image source: {meta['source']}, captured {age:.0f} seconds ago.{quality})",
                images=[image]),
    ]
    try:
        resp, route = await ctx.runtime.router.chat(prompt, role="vision", max_tokens=1200)
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"no vision model could analyse the image: {e}", data=meta)
    answer = resp.message.content.strip()
    return ToolResult(ok=True, content=f"[{meta['source']}, captured {time.strftime('%H:%M:%S', time.localtime(meta['captured_at']))}, "
                                       f"analysed by {route.provider}/{route.model}]{quality}\n{answer}",
                      data={**meta, "model": f"{route.provider}/{route.model}"})


async def _screenshot(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    try:
        img, at = await asyncio.to_thread(capture_screen, args.get("monitor", 1))
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"screenshot failed: {e}")
    image, meta = await asyncio.to_thread(_encode, img, "screenshot", at, ctx.runtime.paths.captures)
    question = args.get("question") or "Describe what is on this screen: the applications, windows and any important content."
    return await analyze(ctx, image, meta, question)


async def _read_text(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    args = {**args, "question": "Transcribe all readable text in this image exactly, preserving line breaks. Mark unreadable parts as [unreadable]."}
    if args.get("path"):
        return await _image_file(args, ctx)
    return await _screenshot(args, ctx)


async def _image_file(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    from jarvis.capabilities.files import PathDenied, _policy  # noqa: PLC0415

    try:
        p = _policy(ctx).resolve(args["path"], ctx)
    except PathDenied as e:
        return ToolResult.fail(str(e))
    if not p.is_file():
        return ToolResult.fail(f"{p} is not a file")
    try:
        img = await asyncio.to_thread(Image.open, p)
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        return ToolResult.fail(f"{p} is not a readable image: {e}")
    image, meta = await asyncio.to_thread(_encode, img, f"file:{p.name}", p.stat().st_mtime, None)
    return await analyze(ctx, image, meta, args.get("question") or "Describe this image.")


async def _camera(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    import cv2  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    def grab() -> tuple[Any, float]:
        cap = cv2.VideoCapture(int(args.get("camera", 0)))
        try:
            if not cap.isOpened():
                raise RuntimeError("no camera could be opened")
            for _ in range(5):  # let exposure settle
                cap.read()
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("the camera returned no frame")
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), time.time()
        finally:
            cap.release()

    try:
        img, at = await asyncio.to_thread(grab)
    except Exception as e:  # noqa: BLE001
        return ToolResult.fail(f"camera capture failed: {e}")
    image, meta = await asyncio.to_thread(_encode, img, f"camera:{args.get('camera', 0)}", at, ctx.runtime.paths.captures)
    return await analyze(ctx, image, meta, args.get("question") or "Describe what the camera sees.")


def vision_tools() -> list[Tool]:
    kw = ("screen", "screenshot", "looking at", "see", "image", "photo", "picture", "camera", "read the text", "what's on", "ocr")
    return [
        Tool("look_at_screen", "Take a screenshot now and answer a question about it (or describe it). The capture is fresh at call time.",
             {"type": "object", "properties": {"question": {"type": "string"}, "monitor": {"type": "integer", "minimum": 0, "default": 1}}},
             _screenshot, Risk.LOW_RISK, "vision.screen", "vision", available=_available_screen,
             describe=lambda a: "capture the screen and analyse it with the vision model",
             keywords=kw),
        Tool("read_text_on_screen", "Read the exact text from the screen (fresh screenshot) or from an image file (OCR).",
             {"type": "object", "properties": {"path": {"type": "string"}, "monitor": {"type": "integer", "minimum": 0, "default": 1}}},
             _read_text, Risk.LOW_RISK, "vision.ocr", "vision", available=_available_screen, keywords=kw),
        Tool("look_at_image", "Answer a question about an image file on this computer.",
             {"type": "object", "properties": {"path": {"type": "string"}, "question": {"type": "string"}}, "required": ["path"]},
             _image_file, Risk.SAFE, "vision.image", "vision", available=_available_screen, keywords=kw),
        Tool("look_through_camera", "Capture a frame from a camera attached to this computer and describe it.",
             {"type": "object", "properties": {"camera": {"type": "integer", "minimum": 0, "default": 0}, "question": {"type": "string"}}},
             _camera, Risk.SENSITIVE, "vision.camera", "vision", available=_available_camera,
             describe=lambda a: f"take a picture with camera {a.get('camera', 0)}", keywords=kw),
    ]


def register(runtime: Any) -> None:
    runtime.tools.register_all(vision_tools())
