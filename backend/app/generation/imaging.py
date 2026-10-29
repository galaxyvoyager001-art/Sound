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


def background_mask(rgb: np.ndarray, tol: float = 30.0) -> np.ndarray:
    """与边框连通的近白色区域 = 背景。"""
    dist = (255.0 - rgb).max(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    near_white = (dist < tol) & (sat < tol * 0.8)
    region = np.zeros_like(near_white)
    region[0, :] = near_white[0, :]
    region[-1, :] = near_white[-1, :]
    region[:, 0] = near_white[:, 0]
    region[:, -1] = near_white[:, -1]
    for _ in range(max(rgb.shape[:2]) * 2):
        nxt = _dilate(region) & near_white
        if (nxt == region).all():
            break
        region = nxt
    return region


def matte_object(data: bytes, work_size: int = 512) -> Image.Image:
    """白底物件插画 → 透明背景 RGBA，并裁切到物件边界（底边即物件着地处）。"""
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
    # 去除边缘白边：按“与白色混合”的模型反解前景色
    safe = np.maximum(a, 0.15)[..., None]
    fg = (rgb - (1 - safe) * 255.0) / safe
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
