"""实时语音识别：百炼 Fun-ASR / Paraformer 实时 WebSocket 协议。

协议（官方文档）：连接 wss://…/api-ws/v1/inference → 发送 run-task → 收到 task-started 后
以二进制帧发送单声道音频 → 持续收到 result-generated → 发送 finish-task → task-finished。
连接异常时自动重连，重连期间的音频暂存（最多 10 秒），时间戳统一换算成“会谈录音时间”。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from .base import usage

log = logging.getLogger("asr")

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2


@dataclass
class ASRResult:
    text: str
    start_s: float  # 相对会谈录音开始
    end_s: float
    final: bool


ResultCallback = Callable[[ASRResult], Awaitable[None]]


class ASRSession:
    async def start(self, on_result: ResultCallback) -> None: ...

    async def send(self, pcm16: bytes) -> None: ...

    async def feed(self, pcm16: bytes, active: bool) -> None:
        await self.send(pcm16)

    async def close(self) -> None: ...


class NullASRSession(ASRSession):
    """演示模式：不识别（由工作人员面板输入文字模拟老人说话）。"""

    async def start(self, on_result: ResultCallback) -> None:
        self.on_result = on_result

    async def send(self, pcm16: bytes) -> None:
        return None

    async def close(self) -> None:
        return None


class DashScopeASRSession(ASRSession):
    def __init__(self, api_key: str, ws_url: str, model: str, vocabulary_id: str = "",
                 max_sentence_silence_ms: int = 800):
        self.api_key = api_key
        self.ws_url = ws_url
        self.model = model
        self.vocabulary_id = vocabulary_id
        self.max_sentence_silence_ms = max_sentence_silence_ms
        self._ws = None
        self._task_id = ""
        self._ready = asyncio.Event()
        self._closed = False
        self._bytes_total = 0  # 会谈开始以来收到的音频字节
        self._task_offset_s = 0.0  # 当前识别任务开始时对应的会谈时间
        self._pending: list[bytes] = []
        self._reader: Optional[asyncio.Task] = None
        self._on_result: Optional[ResultCallback] = None
        self._reconnect_lock = asyncio.Lock()
        # 省钱：只在可能有人说话时把音频送去识别（按音频时长计费）；安静超过 IDLE_CLOSE_S 秒就结束识别任务
        self._idle = False
        self._preroll: deque[bytes] = deque()
        self._last_active = time.monotonic()
        self._waking: Optional[asyncio.Task] = None
        self._finishing: Optional[asyncio.Task] = None

    IDLE_CLOSE_S = 12.0
    PREROLL_BYTES = BYTES_PER_SECOND  # 开口前 1 秒的音频一起送去识别，避免吞掉第一个字

    def _run_task(self) -> dict:
        params: dict = {"format": "pcm", "sample_rate": SAMPLE_RATE,
                        "semantic_punctuation_enabled": False,
                        "max_sentence_silence": self.max_sentence_silence_ms,
                        "punctuation_prediction_enabled": True,
                        "inverse_text_normalization_enabled": True}
        if self.model.startswith("paraformer"):
            params["heartbeat"] = True
            params["disfluency_removal_enabled"] = False
        if self.vocabulary_id:
            params["vocabulary_id"] = self.vocabulary_id
        return {"header": {"action": "run-task", "task_id": self._task_id, "streaming": "duplex"},
                "payload": {"task_group": "audio", "task": "asr", "function": "recognition",
                            "model": self.model, "parameters": params, "input": {}}}

    async def start(self, on_result: ResultCallback) -> None:
        self._on_result = on_result
        await self._connect()

    async def _connect(self) -> None:
        import websockets

        self._ready.clear()
        self._task_id = uuid.uuid4().hex
        # 暂存的音频（预录 + 断线期间）开始于当前时间之前
        self._task_offset_s = (self._bytes_total - sum(len(c) for c in self._pending)) / BYTES_PER_SECOND
        headers = {"Authorization": f"Bearer {self.api_key}", "X-DashScope-DataInspection": "disable"}
        self._ws = await websockets.connect(self.ws_url, additional_headers=headers, max_size=2 ** 22,
                                            open_timeout=10, ping_interval=20)
        await self._ws.send(json.dumps(self._run_task()))
        self._reader = asyncio.create_task(self._read_loop(self._ws))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=10)
        except asyncio.TimeoutError:
            log.warning("ASR task-started 超时")
            raise
        # 发送重连期间暂存的音频
        pending, self._pending = self._pending, []
        for chunk in pending:
            await self._ws.send(chunk)

    async def _read_loop(self, ws) -> None:
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                data = json.loads(msg)
                event = data.get("header", {}).get("event")
                if event == "task-started":
                    self._ready.set()
                elif event == "result-generated":
                    sent = data.get("payload", {}).get("output", {}).get("sentence", {})
                    if sent.get("heartbeat"):
                        continue
                    text = (sent.get("text") or "").strip()
                    if not text:
                        continue
                    begin = (sent.get("begin_time") or 0) / 1000.0
                    end = sent.get("end_time")
                    end = end / 1000.0 if end is not None else begin
                    final = bool(sent.get("sentence_end"))
                    if final:
                        dur = (data.get("payload", {}).get("usage") or {}).get("duration")
                        if dur:
                            usage.add(f"asr:{self.model}:seconds", float(dur))
                    if self._on_result:
                        await self._on_result(ASRResult(text, self._task_offset_s + begin,
                                                        self._task_offset_s + end, final))
                elif event == "task-failed":
                    log.warning("ASR 任务失败：%s", data.get("header", {}).get("error_message"))
                    break
                elif event == "task-finished":
                    break
        except Exception as e:  # 连接断开
            log.warning("ASR 连接中断：%s", e)
        finally:
            self._ready.clear()
            if not self._closed and not self._idle:
                asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        async with self._reconnect_lock:
            if self._closed or self._ready.is_set() or self._idle:
                return
            for attempt in range(6):
                try:
                    await asyncio.sleep(min(8, 0.5 * 2 ** attempt))
                    await self._connect()
                    log.info("ASR 已重连")
                    return
                except Exception as e:
                    log.warning("ASR 重连失败：%s", e)

    async def send(self, pcm16: bytes) -> None:
        await self.feed(pcm16, True)

    async def feed(self, pcm16: bytes, active: bool) -> None:
        """active=False：这一段大概没人说话，只计时、放进预录缓冲，不送去识别。"""
        now = time.monotonic()
        if not active:
            self._bytes_total += len(pcm16)
            self._preroll.append(pcm16)
            while sum(len(c) for c in self._preroll) > self.PREROLL_BYTES:
                self._preroll.popleft()
            if self._ready.is_set() and not self._idle and now - self._last_active > self.IDLE_CLOSE_S:
                self._idle = True
                self._finishing = asyncio.create_task(self._finish_task())
            return
        self._last_active = now
        if self._idle or (self._ws is None and not self._ready.is_set()):
            # 从安静中醒来：带上预录音频，重新开始识别任务
            self._pending.extend(self._preroll)
            self._preroll.clear()
            self._bytes_total += len(pcm16)
            self._pending.append(pcm16)
            if self._waking is None or self._waking.done():
                self._waking = asyncio.create_task(self._wake())
            return
        self._bytes_total += len(pcm16)
        if self._ready.is_set() and self._ws is not None:
            try:
                await self._ws.send(pcm16)
                return
            except Exception:
                self._ready.clear()
        self._pending.append(pcm16)
        # 最多暂存 10 秒
        while sum(len(c) for c in self._pending) > BYTES_PER_SECOND * 10:
            dropped = self._pending.pop(0)
            log.debug("丢弃暂存音频 %d 字节", len(dropped))

    async def _wake(self) -> None:
        if self._finishing is not None and not self._finishing.done():
            await self._finishing  # 上一个识别任务还在收尾
        async with self._reconnect_lock:
            if self._closed or self._ready.is_set():
                return
            self._idle = False
            try:
                await self._connect()
            except Exception as e:
                log.warning("ASR 连接失败：%s", e)
                asyncio.create_task(self._reconnect())

    async def _finish_task(self) -> None:
        ws = self._ws
        try:
            if ws is not None:
                await ws.send(json.dumps({"header": {"action": "finish-task", "task_id": self._task_id,
                                                     "streaming": "duplex"}, "payload": {"input": {}}}))
                if self._reader:
                    await asyncio.wait_for(self._reader, timeout=5)
        except Exception:
            pass
        finally:
            self._ready.clear()
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            if self._ws is ws:
                self._ws = None

    async def close(self) -> None:
        self._closed = True
        try:
            if self._ws is not None and self._ready.is_set():
                await self._ws.send(json.dumps({"header": {"action": "finish-task", "task_id": self._task_id,
                                                           "streaming": "duplex"}, "payload": {"input": {}}}))
                if self._reader:
                    await asyncio.wait_for(self._reader, timeout=5)
        except Exception:
            pass
        finally:
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:
                    pass
