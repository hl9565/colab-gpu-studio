"""Reference images: a parameter of type "image" carries an asset id; the file ships inside the job snapshot."""

from __future__ import annotations

import base64
import io

from PIL import Image

from ..common import APIError

# All files travel base64-twice inside the one execute_code string; a 0.9 MB PNG made Colab answer
# "service unavailable", a 100 KB JPEG worked. So every image is re-encoded as JPEG and the total is capped.
MAX_TOTAL_BYTES = 400 * 1024
DEFAULT_MAX_EDGE = 1088
DEFAULT_JPEG_QUALITY = 85

# Progressive (max_edge_cap, jpeg_quality) ladder used only when the default pass exceeds MAX_TOTAL_BYTES.
COMPRESSION_LADDER: tuple[tuple[int | None, int], ...] = (
    (None, DEFAULT_JPEG_QUALITY),
    (768, 72),
    (640, 65),
    (512, 58),
    (448, 50),
    (384, 42),
    (320, 35),
)


def _jpeg(data: bytes, max_edge: int, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue()


def resolve_input_images(own_asset, project_id: str, manifest: dict, params: dict) -> tuple[dict, list[dict]]:
    """Swap asset ids for the file names ComfyUI will see; return (params, files to ship)."""
    from ..storage import asset_path

    resolved, sources = dict(params), []
    for name, spec in (manifest.get("parameters") or {}).items():
        if spec.get("type") != "image" or params.get(name) in (None, ""):
            continue
        asset = own_asset(params[name])
        if asset.get("project_id") != project_id:
            raise APIError("ASSET_PROJECT_MISMATCH", "参考素材不属于当前项目")
        if asset.get("mime") not in ("image/png", "image/jpeg", "image/webp"):
            raise APIError("VALIDATION_ERROR", f"参数 {name} 只接受 PNG/JPEG/WebP 图片")
        filename = f"input_{name}.jpg"
        resolved[name] = filename
        sources.append(
            (
                filename,
                asset_path(asset).read_bytes(),
                spec.get("max_edge", DEFAULT_MAX_EDGE),
            )
        )

    files: list[dict] = []
    if sources:
        for edge_cap, quality in COMPRESSION_LADDER:
            files = [
                {
                    "filename": filename,
                    "content": _jpeg(
                        raw_bytes,
                        base_max_edge if edge_cap is None else min(base_max_edge, edge_cap),
                        quality=quality,
                    ),
                }
                for filename, raw_bytes, base_max_edge in sources
            ]
            if sum(len(f["content"]) for f in files) <= MAX_TOTAL_BYTES:
                break
        else:
            raise APIError("VALIDATION_ERROR", "参考图压缩后仍超过 400KB（随启动代码发送的上限），请少选几张或换小图")

    for f in files:
        f["content_b64"] = base64.b64encode(f.pop("content")).decode("ascii")
    return resolved, files
