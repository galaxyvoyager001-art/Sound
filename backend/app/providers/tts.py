"""语音合成：百炼 CosyVoice WebSocket（run-task → continue-task → finish-task，二进制帧返回音频）。

合成结果按文本缓存（陪聊者常用的问候、收尾语重复率高）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from .base import ProviderError, usage, with_retry

log = logging.getLogger("tts")


class TTS:
    async def synthesize(self, text: str) -> Optional[bytes]:
        """返回 mp3 字节；返回 None 表示由前端使用浏览器自带语音兜底。"""
        raise NotImplementedError


class NullTTS(TTS):
    async def synthesize(self, text: str) -> Optional[bytes]:
        return None


class CosyVoiceTTS(TTS):
    def __init__(self, api_key: str, ws_url: str, model: str, voice: str, rate: float,
                 cache_dir: Path, sample_rate: int = 22050):
        self.api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self.voice = voice
        self.rate = rate
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.model}|{self.voice}|{self.rate}|{text}".encode()).hexdigest()[:20]
        return self.cache_dir / f"{h}.mp3"

    async def synthesize(self, text: str) -> Optional[bytes]:
        text = text.strip()
        if not text:
            return None
        cp = self._cache_path(text)
        if cp.exists():
            return cp.read_bytes()
        audio = await with_retry(lambda: self._once(text), attempts=3, base_delay=0.5, what="语音合成")
        cp.write_bytes(audio)
        usage.add(f"tts:{self.model}:chars", len(text))
        return audio

    async def _once(self, text: str) -> bytes:
        import websockets

        task_id = uuid.uuid4().hex
        headers = {"Authorization": f"Bearer {self.api_key}", "X-DashScope-DataInspection": "disable"}
        chunks: list[bytes] = []
        async with websockets.connect(self.ws_url, additional_headers=headers, max_size=2 ** 24,
                                      open_timeout=10) as ws:
            await ws.send(json.dumps({
                "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"task_group": "audio", "task": "tts", "function": "SpeechSynthesizer",
                            "model": self.model,
                            "parameters": {"text_type": "PlainText", "voice": self.voice, "format": "mp3",
                                           "sample_rate": self.sample_rate, "volume": 50,
                                           "rate": self.rate, "pitch": 1.0},
                            "input": {}}}))
            started = False
            async def recv_until(pred, timeout=20.0):
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if isinstance(msg, bytes):
                        chunks.append(msg)
                        continue
                    data = json.loads(msg)
                    ev = data.get("header", {}).get("event")
                    if ev == "task-failed":
                        raise ProviderError(f"TTS 失败：{data.get('header', {}).get('error_message')}")
                    if pred(ev):
                        return ev
            await recv_until(lambda ev: ev == "task-started")
            started = True
            await ws.send(json.dumps({"header": {"action": "continue-task", "task_id": task_id, "streaming": "duplex"},
                                      "payload": {"input": {"text": text}}}))
            await ws.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                                      "payload": {"input": {}}}))
            await recv_until(lambda ev: ev == "task-finished", timeout=30.0)
        if not started or not chunks:
            raise ProviderError("TTS 没有返回音频")
        return b"".join(chunks)
