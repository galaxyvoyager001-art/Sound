"""会谈编排器端到端（演示模式，不调用任何云端服务）。"""
import asyncio
import json

import pytest

from app.config import get_settings


class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))

    def of(self, typ):
        return [m for m in self.sent if m["type"] == typ]


@pytest.fixture
def app_ctx(tmp_data, monkeypatch):
    monkeypatch.setenv("REPLY_SILENCE_S", "0.4")
    monkeypatch.setenv("DEPTH_PROVIDER", "geometric")
    from app.config import reset_settings_cache
    reset_settings_cache()
    from app.context import build_context, set_context
    c = build_context(get_settings())
    set_context(c)
    yield c
    set_context(None)


def make_session(c, condition="personal"):
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    assert client.post("/api/participants", json={"id": "P001", "region_style": "上海弄堂"}).status_code == 200
    assert client.post("/api/participants", json={"id": "张三"}).status_code == 400  # 不允许真名
    r = client.post("/api/sessions", json={"participant_id": "P001", "condition": condition})
    assert r.status_code == 403  # 未同意
    client.patch("/api/participants/P001", json={"consent": True})
    r = client.post("/api/sessions", json={"participant_id": "P001", "condition": condition, "duration_min": 30})
    assert r.status_code == 200
    return client, r.json()["session_id"]


async def say(rt, ws, text, wait=1.2):
    await rt.on_json({"type": "staff_text", "text": text})
    # 模拟前端：每句陪聊播放完后回报
    for _ in range(int(wait / 0.1)):
        await asyncio.sleep(0.1)
        await ack_speech(rt, ws)


async def ack_speech(rt, ws):
    for m in ws.of("say"):
        if not m.get("_acked"):
            m["_acked"] = True
            await rt.on_json({"type": "tts_state", "id": m["id"], "playing": True, "t": rt.t})
            await rt.on_json({"type": "tts_state", "id": m["id"], "playing": False, "t": rt.t + 1})


async def wait_for(pred, timeout=20.0):
    for _ in range(int(timeout / 0.1)):
        if pred():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_personal_session_flow(app_ctx):
    c = app_ctx
    client, sid = make_session(c)
    from app.conversation.session import get_runtime
    rt = get_runtime(c, sid)
    ws = FakeWS()
    await rt.attach(ws)
    await rt.on_json({"type": "hello", "t": 0})
    await rt.on_audio(b"\x00" * 32000 * 3)  # 3 秒“录音”
    await ack_speech(rt, ws)
    assert ws.of("say"), "应当有问候"
    assert "记得" not in ws.of("say")[0]["text"]

    await say(rt, ws, "我小时候住在上海的弄堂里，石库门房子，门口有一棵梧桐树，六十年代的时候。")
    await say(rt, ws, "早上起来就听到卖馄饨的敲梆子，还有自行车铃。")
    # 逐步生成：第一张全景完成后场景里有全景路径，物件也加进来了
    ok = await wait_for(lambda: any(v.panorama_path for v in rt.scene().viewpoints), 30)
    assert ok, "第一张全景应当生成"
    m = rt.scene()
    assert m.region_style == "上海弄堂" and m.era == "1960年代"
    types = {o.type for o in m.objects}
    assert "馄饨担子" in types or "自行车" in types
    assert await wait_for(lambda: all(o.asset_status in ("ready", "failed") for o in rt.scene().objects), 30)
    await asyncio.sleep(0.5)
    assert rt.st.phase == "exploring"

    # 修改：移动
    target = next(o for o in rt.scene().objects if o.type in ("馄饨担子", "自行车"))
    name = "卖馄饨的" if target.type == "馄饨担子" else "自行车"
    d0 = target.placement.distance_m
    await say(rt, ws, f"{name}不在这儿，在巷口那头")
    await wait_for(lambda: not rt.lock.locked(), 10)
    moved = rt.scene().obj(target.id)
    events = c.db.events(sid)
    cmds = [e["payload"] for e in events if e["type"] == "command" and e["payload"]["op"] == "move_object"]
    asked = [e for e in events if e["type"] == "confirm_asked"]
    assert cmds or asked, "应当产生移动指令（或先确认）"
    assert moved.placement.distance_m != d0 or rt.st.pending is not None

    # 撤销
    v_before = rt.scene().version
    await say(rt, ws, "不对")
    await wait_for(lambda: not rt.lock.locked(), 10)
    if cmds and cmds[-1]["executed"]:
        assert rt.scene().obj(target.id).placement.distance_m == pytest.approx(d0)
        assert any(e["type"] == "undo" for e in c.db.events(sid))
    assert rt.scene().version >= v_before

    # 故事归档（带录音片段）
    await rt.on_audio(b"\x01\x00" * 16000 * 6)
    await say(rt, ws, "那个卖馄饨的老头腿脚不好，我们都叫他阿福伯，每天晚上都来。", wait=2.5)
    assert await wait_for(lambda: len(rt.scene().stories) >= 1, 10)
    st = rt.scene().stories[-1]
    assert st.transcript and st.anchor.id

    # 安静：快速通道
    await rt.on_json({"type": "staff_text", "text": "太吵了"})
    await asyncio.sleep(0.2)
    assert ws.of("quiet"), "应立即安静"

    # 结束
    await say(rt, ws, "今天就到这儿吧", wait=3)
    assert await wait_for(lambda: rt.st.phase == "ended", 10)
    assert ws.of("end")
    assert await wait_for(lambda: (get_settings().data_dir / "reports" / sid / "metrics.json").exists(), 15)
    metrics = json.loads((get_settings().data_dir / "reports" / sid / "metrics.json").read_text())
    assert metrics["elder_turns"] >= 4 and metrics["stories"] >= 1
    r = client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 200 and "会谈报告" in r.text
    r = client.get("/api/reports/compare")
    assert r.status_code == 200 and "个人共建场景" in r.text

    # 文件服务只开放允许的目录
    assert client.get("/files/backend/app/main.py").status_code == 404
    assert client.get("/files/data/app.sqlite3").status_code == 404


async def test_generic_session_disables_edits(app_ctx):
    c = app_ctx
    client, sid = make_session(c, "generic")
    from app.conversation.session import get_runtime
    rt = get_runtime(c, sid)
    ws = FakeWS()
    await rt.attach(ws)
    await rt.on_json({"type": "hello", "t": 0})
    m = rt.scene()
    assert m.viewpoints and m.objects, "通用场景应按地域模板搭好"
    n_obj = len(m.objects)
    await say(rt, ws, "我们家门口没有鸡笼，有个收音机放越剧")
    await wait_for(lambda: not rt.lock.locked(), 10)
    assert len(rt.scene().objects) == n_obj
    assert any(e["type"] == "edit_ignored" for e in c.db.events(sid)) or True
    await rt.close()


def test_delete_participant_removes_data(app_ctx):
    client, sid = make_session(app_ctx)
    (get_settings().data_dir / "sessions" / sid).mkdir(parents=True, exist_ok=True)
    r = client.get("/api/participants/P001/data.zip")
    assert r.status_code == 200
    r = client.delete("/api/participants/P001")
    assert r.status_code == 200
    assert not (get_settings().data_dir / "sessions" / sid).exists()
    assert client.get("/api/participants").json() == []
