"""图像生成：通义万相（默认）、Blockade Labs Skybox（全景）、本地程序化绘制（演示/降级）。

官方文档要点（2026-10 查阅）：
- wan2.6-t2i：同步接口 POST /api/v1/services/aigc/multimodal-generation/generation，
  body = {model, input:{messages:[{role:user, content:[{text}]}]}, parameters:{size:"W*H", n, negative_prompt,
  prompt_extend, watermark}}；总像素 1280²~1440²，宽高比 1:4~4:1；结果在
  output.choices[].message.content[].image（PNG URL，24 小时有效）。
- wan2.5 及更早：异步 POST /api/v1/services/aigc/text2image/image-synthesis（X-DashScope-Async: enable），
  轮询 GET /api/v1/tasks/{task_id}，结果在 output.results[].url。
- 两者都不输出透明背景，也不保证等距柱状投影，所以抠图和全景后处理在本地完成。
- Skybox：POST https://backend.blockadelabs.com/api/v1/skybox（x-api-key），
  轮询 GET /api/v1/imagine/requests/{id}，完成后有 file_url 与 depth_map_url。
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
from typing import Optional

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .base import ProviderError, usage, with_retry

log = logging.getLogger("image")


class ImageProvider:
    name = "base"

    async def generate(self, prompt: str, negative: str, width: int, height: int, seed: Optional[int] = None) -> bytes:
        raise NotImplementedError


class PanoramaProvider:
    name = "base"
    native_equirect = False

    async def generate_panorama(self, prompt: str, negative: str, seed: Optional[int] = None
                                ) -> tuple[bytes, Optional[bytes]]:
        """返回 (全景图字节, 服务自带的深度图字节或 None)。"""
        raise NotImplementedError


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.content


class WanxImage(ImageProvider):
    name = "wanx"

    def __init__(self, api_key: str, api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model

    @property
    def _is_messages_api(self) -> bool:
        m = self.model.lower()
        return not any(m.startswith(p) for p in ("wan2.5", "wan2.2", "wan2.1", "wanx"))

    @staticmethod
    def fit_size(width: int, height: int, lo: int = 1280 * 1280, hi: int = 1440 * 1440) -> tuple[int, int]:
        """按服务限制调整尺寸：保持宽高比，总像素落在 [lo, hi]，边长为 16 的倍数。"""
        ratio = width / height
        ratio = min(4.0, max(0.25, ratio))
        area = min(hi, max(lo, width * height))
        h = math.sqrt(area / ratio)
        w = h * ratio
        w16, h16 = int(w // 16 * 16), int(h // 16 * 16)
        while w16 * h16 < lo:
            w16 += 16
        while w16 * h16 > hi:
            w16 -= 16
        return w16, h16

    async def generate(self, prompt: str, negative: str, width: int, height: int, seed: Optional[int] = None) -> bytes:
        w, h = self.fit_size(width, height)
        headers = {"Authorization": f"Bearer {self.api_key}", "X-DashScope-DataInspection": "disable"}
        params: dict = {"size": f"{w}*{h}", "n": 1, "negative_prompt": negative[:500],
                        "prompt_extend": False, "watermark": False}
        if seed is not None:
            params["seed"] = seed

        async def call() -> bytes:
            async with httpx.AsyncClient(timeout=180) as c:
                if self._is_messages_api:
                    body = {"model": self.model,
                            "input": {"messages": [{"role": "user", "content": [{"text": prompt[:2000]}]}]},
                            "parameters": params}
                    r = await c.post(f"{self.api_base}/services/aigc/multimodal-generation/generation",
                                     json=body, headers=headers)
                    self._raise(r)
                    data = r.json()
                    url = None
                    for ch in data.get("output", {}).get("choices", []):
                        for item in ch.get("message", {}).get("content", []):
                            if item.get("image"):
                                url = item["image"]
                                break
                    if not url:
                        raise ProviderError(f"万相没有返回图片：{str(data)[:300]}")
                else:
                    body = {"model": self.model, "input": {"prompt": prompt[:500], "negative_prompt": negative[:500]},
                            "parameters": params}
                    r = await c.post(f"{self.api_base}/services/aigc/text2image/image-synthesis", json=body,
                                     headers={**headers, "X-DashScope-Async": "enable"})
                    self._raise(r)
                    task_id = r.json()["output"]["task_id"]
                    url = await self._poll(c, task_id, headers)
            usage.add(f"image:{self.model}:count", 1)
            return await _download(url)

        return await with_retry(call, attempts=3, base_delay=2.0, what=f"万相生成 {self.model}")

    async def _poll(self, c: httpx.AsyncClient, task_id: str, headers: dict) -> str:
        for _ in range(60):
            await asyncio.sleep(4)
            r = await c.get(f"{self.api_base}/tasks/{task_id}", headers=headers)
            self._raise(r)
            out = r.json().get("output", {})
            st = out.get("task_status")
            if st == "SUCCEEDED":
                for res in out.get("results", []):
                    if res.get("url"):
                        return res["url"]
                raise ProviderError("万相任务成功但没有图片")
            if st in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ProviderError(f"万相任务失败：{out.get('message')}", retryable=st != "FAILED")
        raise ProviderError("万相任务超时")

    @staticmethod
    def _raise(r: httpx.Response) -> None:
        if r.status_code >= 400:
            raise ProviderError(f"万相 HTTP {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code in (429, 500, 502, 503, 504), status=r.status_code)


class WanxPanorama(PanoramaProvider):
    """用万相按 2:1 生成，再由 generation.panorama 做无缝和两极处理。"""
    name = "wanx"
    native_equirect = False

    def __init__(self, image: WanxImage):
        self.image = image

    async def generate_panorama(self, prompt: str, negative: str, seed: Optional[int] = None):
        return await self.image.generate(prompt, negative, 2048, 1024, seed), None


class SkyboxPanorama(PanoramaProvider):
    name = "skybox"
    native_equirect = True
    BASE = "https://backend.blockadelabs.com/api/v1"

    def __init__(self, api_key: str, style_id: str):
        self.api_key = api_key
        self.style_id = int(style_id) if style_id else None

    async def _style(self, c: httpx.AsyncClient) -> int:
        if self.style_id:
            return self.style_id
        r = await c.get(f"{self.BASE}/skybox/styles", headers={"x-api-key": self.api_key})
        r.raise_for_status()
        styles = r.json()
        # 优先选名字里带 watercolor / painting 的风格
        for key in ("watercolor", "painting", "illustration", "anime"):
            for s in styles:
                if key in str(s.get("name", "")).lower():
                    self.style_id = int(s["id"])
                    return self.style_id
        self.style_id = int(styles[0]["id"])
        return self.style_id

    async def generate_panorama(self, prompt: str, negative: str, seed: Optional[int] = None):
        async def call():
            async with httpx.AsyncClient(timeout=60) as c:
                body = {"skybox_style_id": await self._style(c), "prompt": prompt[:1900],
                        "negative_text": negative[:550]}
                if seed is not None:
                    body["seed"] = seed
                r = await c.post(f"{self.BASE}/skybox", json=body, headers={"x-api-key": self.api_key})
                if r.status_code == 403 and "negative" in r.text.lower():
                    body.pop("negative_text")
                    r = await c.post(f"{self.BASE}/skybox", json=body, headers={"x-api-key": self.api_key})
                if r.status_code >= 400:
                    raise ProviderError(f"Skybox HTTP {r.status_code}: {r.text[:200]}",
                                        retryable=r.status_code in (429, 500, 502, 503), status=r.status_code)
                rid = r.json()["id"]
                # 官方建议用 webhook；本地部署没有公网地址，所以低频轮询（每 8 秒）
                for _ in range(60):
                    await asyncio.sleep(8)
                    q = await c.get(f"{self.BASE}/imagine/requests/{rid}", headers={"x-api-key": self.api_key})
                    if q.status_code >= 400:
                        continue
                    req = q.json().get("request", q.json())
                    st = req.get("status")
                    if st == "complete":
                        img = await _download(req["file_url"])
                        depth = await _download(req["depth_map_url"]) if req.get("depth_map_url") else None
                        usage.add("image:skybox:count", 1)
                        return img, depth
                    if st in ("abort", "error"):
                        raise ProviderError(f"Skybox 生成失败：{req.get('error_message')}")
                raise ProviderError("Skybox 生成超时")
        return await with_retry(call, attempts=2, base_delay=5.0, what="Skybox 全景")


# ---------------- 本地程序化绘制（演示模式 / 降级兜底） ----------------
def _font(size: int):
    for p in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "C:/Windows/Fonts/msyh.ttc", "/System/Library/Fonts/PingFang.ttc"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _png(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def paint_placeholder_panorama(seed: int = 0, width: int = 2048, height: int = 1024, warm: bool = True) -> bytes:
    """画一张暖色调的“水彩”弄堂占位全景：天空、两侧房屋剪影、地面。仅用于演示与降级。"""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, height)[:, None]
    sky_top, sky_hor = np.array([196, 214, 222]), np.array([244, 228, 200])
    ground_hor, ground_bot = np.array([186, 168, 140]), np.array([120, 104, 86])
    img = np.zeros((height, width, 3), np.float32)
    up = y < 0.5
    t = (y / 0.5)
    img[:] = np.where(up, sky_top * (1 - t) + sky_hor * t, 0)
    t2 = (y - 0.5) / 0.5
    img[:] = np.where(~up, ground_hor * (1 - t2) + ground_bot * t2, img)
    pil = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil, "RGBA")
    # 房屋剪影：沿方位角排列，水平方向可循环
    x = 0
    while x < width:
        w = int(rng.integers(120, 280))
        top = int(height * (0.5 - rng.uniform(0.12, 0.30)))
        col = tuple(int(c) for c in rng.integers([150, 120, 100], [190, 160, 135])) + (235,)
        if rng.random() < 0.82:
            d.rectangle([x, top, x + w, int(height * 0.53)], fill=col)
            d.polygon([(x - 10, top), (x + w // 2, top - int(rng.integers(15, 45))), (x + w + 10, top)],
                      fill=(90, 84, 82, 240))
            for wx in range(x + 18, x + w - 30, 46):
                d.rectangle([wx, top + 30, wx + 22, top + 62], fill=(92, 74, 60, 200))
        x += w + int(rng.integers(0, 40))
    # 晾衣竿
    for _ in range(4):
        x0 = int(rng.integers(0, width - 300))
        yy = int(height * rng.uniform(0.33, 0.42))
        d.line([(x0, yy), (x0 + 260, yy + 6)], fill=(70, 60, 50, 200), width=3)
        for k in range(4):
            cx = x0 + 30 + k * 55
            d.rectangle([cx, yy + 4, cx + 30, yy + 44],
                        fill=tuple(int(c) for c in rng.integers([150, 120, 110], [230, 210, 190])) + (210,))
    pil = pil.filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(pil).astype(np.float32)
    paper = rng.normal(0, 6, (height // 4, width // 4, 1)).astype(np.float32)
    paper = np.asarray(Image.fromarray(paper[..., 0]).resize((width, height), Image.BICUBIC))[..., None]
    arr = arr + paper
    if warm:
        arr = arr * np.array([1.04, 1.0, 0.92])
    return _png(Image.fromarray(arr.clip(0, 255).astype(np.uint8)))


def paint_placeholder_object(name: str, seed: int = 0, size: int = 768) -> bytes:
    """演示用物件插画：白底上的暖色圆角形状 + 名称（后续会被抠图成透明背景）。"""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    col = tuple(int(c) for c in rng.integers([140, 90, 60], [200, 150, 110]))
    m = size // 6
    d.rounded_rectangle([m, m * 2, size - m, size - m // 2], radius=size // 10, fill=col)
    d.ellipse([size // 2 - m, m // 2, size // 2 + m, m * 2 + m // 2], fill=tuple(max(0, c - 30) for c in col))
    f = _font(size // 9)
    tw = d.textlength(name, font=f)
    d.text(((size - tw) / 2, size * 0.55), name, fill=(255, 248, 235), font=f)
    return _png(img.filter(ImageFilter.GaussianBlur(0.8)))


class ProceduralPanorama(PanoramaProvider):
    name = "procedural"
    native_equirect = True

    async def generate_panorama(self, prompt: str, negative: str, seed: Optional[int] = None):
        return paint_placeholder_panorama(seed or abs(hash(prompt)) % 10000), None


class ProceduralImage(ImageProvider):
    name = "procedural"

    async def generate(self, prompt: str, negative: str, width: int, height: int, seed: Optional[int] = None) -> bytes:
        name = prompt.split("|", 1)[0][:8] if "|" in prompt else prompt[:6]
        return paint_placeholder_object(name, seed or abs(hash(prompt)) % 10000)
