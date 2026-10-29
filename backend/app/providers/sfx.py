"""文生音效（可选）：ElevenLabs POST /v1/sound-generation（xi-api-key），返回音频二进制。"""
from __future__ import annotations

from typing import Optional

import httpx

from .base import ProviderError, usage, with_retry


class SoundEffectGenerator:
    name = "off"
    available = False

    async def generate(self, prompt_en: str, duration_s: Optional[float] = None, loop: bool = False) -> bytes:
        raise ProviderError("没有配置文生音效服务", retryable=False)


class ElevenLabsSFX(SoundEffectGenerator):
    name = "elevenlabs"
    available = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def generate(self, prompt_en: str, duration_s: Optional[float] = None, loop: bool = False) -> bytes:
        body: dict = {"text": prompt_en, "loop": loop, "prompt_influence": 0.45}
        if duration_s:
            body["duration_seconds"] = max(0.5, min(30.0, duration_s))

        async def call() -> bytes:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post("https://api.elevenlabs.io/v1/sound-generation",
                                 params={"output_format": "mp3_44100_128"}, json=body,
                                 headers={"xi-api-key": self.api_key})
            if r.status_code >= 400:
                raise ProviderError(f"ElevenLabs HTTP {r.status_code}: {r.text[:200]}",
                                    retryable=r.status_code >= 429, status=r.status_code)
            usage.add("sfx:elevenlabs:count", 1)
            return r.content

        return await with_retry(call, attempts=2, what="文生音效")
