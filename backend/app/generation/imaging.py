"""全景后处理与物件抠图（纯 numpy/PIL，不依赖额外库）。"""
from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image, ImageFilter

log = logging.getLogger("imaging")

PANO_W, PANO_H = 4096, 2048


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def make_seamless(img: Image.Image, blend_frac: float = 0.05) -> Image.Image:
    """左右接缝：把最右侧一段与最左侧一段交叉淡化后裁掉重复部分，保证 u=0 与 u=1 处连续。

    结果序列为 [B, W-B) + 混合区（从 a[W-B+i] 渐变到 a[i]），混合区末尾 ≈ a[B-1]，
    紧接着循环回 a[B]，两端都连续。"""
    a = _arr(img)
    h, w, _ = a.shape
    b = max(8, int(w * blend_frac))
    t = np.linspace(0, 1, b, dtype=np.float32)[None, :, None]
    t = t * t * (3 - 2 * t)  # smoothstep
    blend = a[:, w - b:] * (1 - t) + a[:, :b] * t
    out = np.concatenate([a[:, b:w - b], blend], axis=1)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def soften_poles(img: Image.Image, start_deg: float = 55.0) -> Image.Image:
    """两极柔化：越接近天顶/天底，水平方向模糊越强并逐渐趋向该行平均色，消除收缩成一点时的拉丝。"""
    a = _arr(img)
    h, w, _ = a.shape
    pitch = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    out = a.copy()
    for y in range(h):
        p = abs(pitch[y])
        if p < start_deg:
            continue
        k = (p - start_deg) / (90.0 - start_deg)  # 0..1
        radius = int(2 + k * k * w * 0.08)
        row = a[y]
        cs = np.cumsum(np.concatenate([row[-radius:], row, row[:radius]], axis=0), axis=0)
        blurred = (cs[2 * radius:] - cs[:-2 * radius]) / (2 * radius)
        blurred = blurred[:w]
        mean = row.mean(axis=0, keepdims=True)
        mix = k ** 3
        out[y] = blurred * (1 - mix) + mean * mix
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def finalize_panorama(data: bytes, native_equirect: bool) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if not native_equirect:
        img = make_seamless(img)
    img = img.resize((PANO_W, PANO_H), Image.LANCZOS)
    img = soften_poles(img)
    return img


def to_jpeg(img: Image.Image, quality: int = 88, max_w: int | None = None) -> bytes:
    if max_w and img.width > max_w:
        img = img.resize((max_w, max_w * img.height // img.width), Image.LANCZOS)
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=quality, optimize=True, progressive=True)
    return b.getvalue()


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _border_palette(rgb: np.ndarray, ring: int, n: int = 24) -> np.ndarray:
    """边框一圈像素的代表色（背景可能是纯白、灰色影棚布、带渐变的纸，甚至白色地面）。"""
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[:ring].reshape(-1, 3), rgb[-ring:].reshape(-1, 3),
                             rgb[:, :ring].reshape(-1, 3), rgb[:, -ring:].reshape(-1, 3)])
    q = (border // 16).astype(np.int32)
    keys, inv, counts = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_inverse=True, return_counts=True)
    order = np.argsort(-counts)[:n]
    # 只保留占边框 1% 以上的颜色（排除物件碰到边缘的那一点）
    order = [k for k in order if counts[k] >= 0.01 * len(border)] or list(order[:1])
    return np.array([border[inv.ravel() == k].mean(axis=0) for k in order], np.float32)


def background_mask(rgb: np.ndarray, tol: float = 26.0) -> np.ndarray:
    """与边框连通、颜色接近边框颜色的区域 = 背景。"""
    h, w, _ = rgb.shape
    ring = max(2, int(min(h, w) * 0.02))
    palette = _border_palette(rgb, ring)
    d = np.min(np.linalg.norm(rgb[:, :, None, :] - palette[None, None], axis=-1), axis=-1)
    # 模型常在物件下面画一块白色“地面”：与背景连通的近白色也算背景
    whiteish = ((255.0 - rgb).max(axis=2) < 22) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 14)
    cand = (d < tol) | whiteish
    region = np.zeros_like(cand)
    region[:ring] = cand[:ring]
    region[-ring:] = cand[-ring:]
    region[:, :ring] |= cand[:, :ring]
    region[:, -ring:] |= cand[:, -ring:]
    for _ in range(max(h, w) * 2):
        nxt = _dilate(region) & cand
        if (nxt == region).all():
            break
        region = nxt
    return region


def matte_object(data: bytes, work_size: int = 512) -> Image.Image:
    """白底（或任何纯色/渐变影棚背景）物件图 → 透明背景 RGBA，并裁切到物件边界（底边即物件着地处）。"""
    try:  # 装了 rembg 就用更好的抠图
        from rembg import remove  # type: ignore

        out = Image.open(io.BytesIO(remove(data))).convert("RGBA")
        return _crop_alpha(out)
    except Exception:
        pass
    img = Image.open(io.BytesIO(data)).convert("RGB")
    small = img.copy()
    small.thumbnail((work_size, work_size))
    bg = background_mask(np.asarray(small, np.float32))
    alpha_small = Image.fromarray(((~bg) * 255).astype(np.uint8), "L")
    alpha = alpha_small.resize(img.size, Image.BILINEAR)
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.6))
    a = np.asarray(alpha, np.float32) / 255.0
    rgb = np.asarray(img, np.float32)
    # 去除边缘的背景色描边：按“与背景色混合”的模型反解前景色
    bg_col = np.asarray(small, np.float32)[bg].mean(axis=0) if bg.any() else np.array([255.0, 255, 255])
    safe = np.maximum(a, 0.15)[..., None]
    fg = (rgb - (1 - safe) * bg_col) / safe
    fg = np.where(a[..., None] > 0.98, rgb, fg)
    out = np.dstack([fg.clip(0, 255), a * 255]).astype(np.uint8)
    return _crop_alpha(Image.fromarray(out, "RGBA"))


def _crop_alpha(img: Image.Image, pad: int = 6) -> Image.Image:
    a = np.asarray(img)[..., 3]
    ys, xs = np.where(a > 12)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + 1)
    return img.crop((x0, y0, x1, y1))


def to_png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG", optimize=True)
    return b.getvalue()


def blurred_warm_background(width: int = 2048, height: int = 1024) -> Image.Image:
    """audio_only 条件用：只有一片模糊的暖色。"""
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    top = np.array([236, 214, 184], np.float32)
    bottom = np.array([176, 140, 108], np.float32)
    arr = top * (1 - y) + bottom * y
    arr = np.repeat(arr, width, axis=1)
    rng = np.random.default_rng(7)
    blobs = rng.normal(0, 10, (8, 16, 3)).astype(np.float32)
    blob_img = Image.fromarray(np.clip(blobs + 128, 0, 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)
    arr = arr + (np.asarray(blob_img, np.float32) - 128)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(24))
