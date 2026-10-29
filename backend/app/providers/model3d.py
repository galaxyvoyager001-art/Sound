"""可选：图生 3D（默认关闭）。开启后，物件插画完成后再提交图生 3D，得到 GLB 替代 billboard。

Meshy（官方文档）：POST https://api.meshy.ai/openapi/v1/image-to-3d（Authorization: Bearer），
image_url 可用 data URI；GET /openapi/v1/image-to-3d/{id}，status=SUCCEEDED 时 model_urls.glb。
"""
from __future__ import annotations

import asyncio
import base64

import httpx

from .base import ProviderError, usage, with_retry


class Model3DGenerator:
    name = "off"
    available = False

    async def image_to_glb(self, png: bytes) -> bytes:
        raise ProviderError("3D 生成未开启", retryable=False)


class MeshyImageTo3D(Model3DGenerator):
    name = "meshy"
    available = True
    BASE = "https://api.meshy.ai/openapi/v1/image-to-3d"

    def __init__(self, api_key: str, polycount: int = 15000):
        self.api_key = api_key
        self.polycount = polycount

    async def image_to_glb(self, png: bytes) -> bytes:
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async def call() -> bytes:
            async with httpx.AsyncClient(timeout=60) as c:
                body = {"image_url": "data:image/png;base64," + base64.b64encode(png).decode(),
                        "should_texture": True, "enable_pbr": False,
                        "should_remesh": True, "target_polycount": self.polycount}
                r = await c.post(self.BASE, json=body, headers=headers)
                if r.status_code >= 400:
                    raise ProviderError(f"Meshy HTTP {r.status_code}: {r.text[:200]}",
                                        retryable=r.status_code >= 429, status=r.status_code)
                tid = r.json()["result"]
                for _ in range(120):
                    await asyncio.sleep(10)
                    q = await c.get(f"{self.BASE}/{tid}", headers=headers)
                    if q.status_code >= 400:
                        continue
                    d = q.json()
                    if d.get("status") == "SUCCEEDED" and d.get("model_urls", {}).get("glb"):
                        g = await c.get(d["model_urls"]["glb"], follow_redirects=True, timeout=120)
                        g.raise_for_status()
                        usage.add("model3d:meshy:count", 1)
                        return g.content
                    if d.get("status") in ("FAILED", "CANCELED"):
                        raise ProviderError(f"Meshy 任务失败：{d.get('task_error')}", retryable=False)
                raise ProviderError("Meshy 任务超时")

        return await with_retry(call, attempts=2, what="图生 3D")
