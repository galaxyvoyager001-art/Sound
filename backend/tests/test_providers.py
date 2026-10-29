"""云端 Provider 的离线测试：用模拟 HTTP 与本地 WebSocket 服务器复现百炼的协议。"""
import asyncio
import io
import json

import httpx
import pytest
from PIL import Image

from app.conversation.schemas import UtteranceAnalysis
from app.providers import asr as asr_mod
from app.providers import image as image_mod
from app.providers import llm as llm_mod
from app.providers.asr import ASRResult, DashScopeASRSession
from app.providers.image import WanxImage
from app.providers.llm import DashScopeLLM
from app.providers.tts import CosyVoiceTTS


def patch_client(monkeypatch, module, handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(module.httpx, "AsyncClient", factory)


async def test_llm_schema_fallback_and_retry(monkeypatch):
    calls = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        calls.append(body)
        assert req.headers["Authorization"] == "Bearer k"
        assert body.get("enable_thinking") is False
        rf = body["response_format"]
        if rf["type"] == "json_schema":
            assert rf["json_schema"]["strict"] is True
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        if len(calls) == 2:  # 第一次 json_object：输出不合规
            content = '{"labels": ["oops"]}'
        else:
            content = '```json\n{"labels": ["edit"], "confirmation": "none", "control_kind": "none", "emotion": "neutral", "navigate": null}\n```'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    patch_client(monkeypatch, llm_mod, handler)
    llm = DashScopeLLM("k", "https://x/compatible-mode/v1", "qwen-test")
    out = await llm.structured([{"role": "user", "content": "hi"}], UtteranceAnalysis)
    assert out.labels == ["edit"]
    assert len(calls) == 3
    assert calls[1]["response_format"] == {"type": "json_object"}
    assert "JSON" in calls[1]["messages"][-1]["content"]  # json_object 模式要求提示词包含 JSON
    assert any("不符合要求" in m["content"] for m in calls[2]["messages"])  # 带错误信息重试


async def test_wanx_messages_api(monkeypatch):
    png = io.BytesIO()
    Image.new("RGB", (8, 4), (200, 100, 50)).save(png, "PNG")

    def handler(req: httpx.Request):
        if req.url.path.endswith("/multimodal-generation/generation"):
            body = json.loads(req.content)
            assert body["model"] == "wan2.6-t2i"
            w, h = map(int, body["parameters"]["size"].split("*"))
            assert 1280 * 1280 <= w * h <= 1440 * 1440 and w % 16 == 0 and h % 16 == 0 and 1.9 < w / h < 2.1
            assert body["input"]["messages"][0]["content"][0]["text"]
            return httpx.Response(200, json={"output": {"choices": [{"message": {"content": [
                {"type": "image", "image": "https://oss.example/x.png"}]}}]}})
        if req.url.host == "oss.example":
            return httpx.Response(200, content=png.getvalue())
        return httpx.Response(404)

    patch_client(monkeypatch, image_mod, handler)
    img = WanxImage("k", "https://x/api/v1", "wan2.6-t2i")
    data = await img.generate("全景", "", 2048, 1024)
    assert data == png.getvalue()


def test_wanx_fit_size():
    for w, h in ((2048, 1024), (1280, 1280), (512, 2048)):
        fw, fh = WanxImage.fit_size(w, h)
        assert 1280 * 1280 <= fw * fh <= 1440 * 1440
        assert fw % 16 == 0 and fh % 16 == 0


async def _ws_server(handler):
    import websockets
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}"


async def test_asr_protocol_and_time_offsets():
    received = {"run": None, "bytes": 0}

    async def handler(ws):
        async for msg in ws:
            if isinstance(msg, bytes):
                received["bytes"] += len(msg)
                if received["bytes"] >= 32000 and not received.get("sent"):
                    received["sent"] = True
                    for final in (False, True):
                        await ws.send(json.dumps({"header": {"event": "result-generated", "task_id": "t"}, "payload": {
                            "output": {"sentence": {"begin_time": 200, "end_time": 900 if final else None,
                                                    "text": "门口有棵树", "sentence_end": final, "heartbeat": False}},
                            "usage": {"duration": 1} if final else None}}))
                continue
            data = json.loads(msg)
            if data["header"]["action"] == "run-task":
                received["run"] = data
                await ws.send(json.dumps({"header": {"event": "task-started", "task_id": data["header"]["task_id"]}}))
            elif data["header"]["action"] == "finish-task":
                await ws.send(json.dumps({"header": {"event": "task-finished"}}))

    server, url = await _ws_server(handler)
    results: list[ASRResult] = []

    async def cb(r):
        results.append(r)

    s = DashScopeASRSession("k", url, "fun-asr-realtime")
    await s.start(cb)
    for _ in range(12):
        await s.send(b"\x00" * 3200)
    await asyncio.sleep(0.3)
    await s.close()
    server.close()
    run = received["run"]
    assert run["payload"]["task"] == "asr" and run["payload"]["function"] == "recognition"
    assert run["payload"]["parameters"]["format"] == "pcm" and run["payload"]["parameters"]["sample_rate"] == 16000
    assert [r.final for r in results] == [False, True]
    assert results[1].text == "门口有棵树" and results[1].start_s == pytest.approx(0.2) and results[1].end_s == pytest.approx(0.9)


async def test_tts_protocol(tmp_path):
    seen = []

    async def handler(ws):
        async for msg in ws:
            data = json.loads(msg)
            act = data["header"]["action"]
            seen.append(act)
            if act == "run-task":
                p = data["payload"]["parameters"]
                assert data["payload"]["function"] == "SpeechSynthesizer" and p["voice"] == "longanrou_v3" and p["rate"] == 0.85
                await ws.send(json.dumps({"header": {"event": "task-started"}}))
            elif act == "continue-task":
                assert data["payload"]["input"]["text"] == "您好"
                await ws.send(b"ID3fake")
                await ws.send(b"mp3data")
            elif act == "finish-task":
                await ws.send(json.dumps({"header": {"event": "task-finished"}}))

    server, url = await _ws_server(handler)
    tts = CosyVoiceTTS("k", url, "cosyvoice-v3-flash", "longanrou_v3", 0.85, tmp_path)
    audio = await tts.synthesize("您好")
    assert audio == b"ID3fakemp3data"
    assert seen == ["run-task", "continue-task", "finish-task"]
    # 第二次走缓存，不再连接
    assert await tts.synthesize("您好") == audio and seen.count("run-task") == 1
    server.close()
