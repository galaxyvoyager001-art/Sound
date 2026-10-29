"""FastAPI 入口：设置页 API、会谈 WebSocket、素材与数据文件、报告与导出。

启动：cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import to_abs
from .context import get_context
from .conversation.session import drop_runtime, get_runtime
from .providers.base import usage
from .scene.models import Condition, REGION_STYLES, new_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")
app = FastAPI(title="听见从前", version="0.1.0")
PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def ctx():
    return get_context()


# ======================= 状态 =======================
@app.get("/api/status")
def status():
    c = ctx()
    sounds = [a for a in c.registry.all() if a.kind in ("sound", "music")]
    return {"providers": c.providers.describe(), "demo_mode": not c.providers.cloud,
            "regions": list(REGION_STYLES), "sound_assets": len(sounds),
            "sound_groups": len(c.resolver._groups), "usage": usage.snapshot(),
            "reply_silence_s": c.settings.reply_silence_s}


# ======================= 参与者 =======================
class ParticipantIn(BaseModel):
    id: str = Field(description="参与者编号（不填真名）")
    region_style: str = "上海弄堂"
    notes: str = ""


class ParticipantPatch(BaseModel):
    consent: Optional[bool] = None
    region_style: Optional[str] = None
    notes: Optional[str] = None


@app.get("/api/participants")
def list_participants():
    rows = ctx().db.query("SELECT * FROM participants ORDER BY created_at")
    for r in rows:
        r["sessions"] = ctx().db.query("SELECT id, condition, status, started_at, ended_at, duration_min FROM sessions "
                                       "WHERE participant_id=? ORDER BY started_at", (r["id"],))
    return rows


@app.post("/api/participants")
def create_participant(p: ParticipantIn):
    if not PID_RE.match(p.id):
        raise HTTPException(400, "编号只能包含字母、数字、下划线和短横线（不要填真名）")
    if re.search(r"[一-龥]", p.id):
        raise HTTPException(400, "编号不要使用中文姓名")
    if ctx().db.one("SELECT id FROM participants WHERE id=?", (p.id,)):
        raise HTTPException(409, "编号已存在")
    region = p.region_style if p.region_style in REGION_STYLES else "其他"
    ctx().db.execute("INSERT INTO participants (id, created_at, region_style, notes) VALUES (?,?,?,?)",
                     (p.id, time.time(), region, p.notes[:500]))
    ctx().db.log_event(None, "participant_created", {"participant_id": p.id})
    return {"ok": True}


@app.patch("/api/participants/{pid}")
def patch_participant(pid: str, p: ParticipantPatch):
    db = ctx().db
    if not db.one("SELECT id FROM participants WHERE id=?", (pid,)):
        raise HTTPException(404)
    if p.consent is not None:
        db.execute("UPDATE participants SET consent=?, consent_at=? WHERE id=?", (int(p.consent), time.time(), pid))
        db.log_event(None, "consent", {"participant_id": pid, "consent": p.consent})
    if p.region_style is not None:
        db.execute("UPDATE participants SET region_style=? WHERE id=?",
                   (p.region_style if p.region_style in REGION_STYLES else "其他", pid))
    if p.notes is not None:
        db.execute("UPDATE participants SET notes=? WHERE id=?", (p.notes[:500], pid))
    return db.one("SELECT * FROM participants WHERE id=?", (pid,))


@app.delete("/api/participants/{pid}")
def delete_participant(pid: str):
    """删除参与者的全部数据：录音、文字、场景版本、事件日志、生成的个人化素材、导出文件。"""
    c = ctx()
    if not c.db.one("SELECT id FROM participants WHERE id=?", (pid,)):
        raise HTTPException(404)
    sessions = c.db.query("SELECT id, scene_id FROM sessions WHERE participant_id=?", (pid,))
    scene_ids = {s["scene_id"] for s in sessions if s["scene_id"]}
    p = c.db.one("SELECT scene_id FROM participants WHERE id=?", (pid,))
    if p and p["scene_id"]:
        scene_ids.add(p["scene_id"])
    for s in sessions:
        drop_runtime(s["id"])
        c.db.execute("DELETE FROM events WHERE session_id=?", (s["id"],))
        c.db.execute("DELETE FROM utterances WHERE session_id=?", (s["id"],))
        shutil.rmtree(c.settings.data_dir / "sessions" / s["id"], ignore_errors=True)
        shutil.rmtree(c.settings.data_dir / "reports" / s["id"], ignore_errors=True)
    for sid in scene_ids:
        c.store.delete_scene(sid)
        shutil.rmtree(c.settings.data_dir / "scenes" / sid, ignore_errors=True)
    c.registry.remove_participant(pid)
    shutil.rmtree(c.settings.data_dir / "exports" / pid, ignore_errors=True)
    c.db.execute("DELETE FROM sessions WHERE participant_id=?", (pid,))
    c.db.execute("DELETE FROM participants WHERE id=?", (pid,))
    c.db.log_event(None, "participant_deleted", {"participant_id": pid, "sessions": len(sessions)})
    return {"ok": True, "deleted_sessions": len(sessions)}


@app.get("/api/participants/{pid}/data.zip")
def export_participant_data(pid: str):
    """导出参与者的全部数据（研究备份 / 交给本人或家属）。"""
    c = ctx()
    if not c.db.one("SELECT id FROM participants WHERE id=?", (pid,)):
        raise HTTPException(404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("participant.json", json.dumps(c.db.one("SELECT * FROM participants WHERE id=?", (pid,)),
                                                  ensure_ascii=False, indent=1))
        for s in c.db.query("SELECT * FROM sessions WHERE participant_id=?", (pid,)):
            z.writestr(f"{s['id']}/session.json", json.dumps(s, ensure_ascii=False, indent=1))
            z.writestr(f"{s['id']}/events.json", json.dumps(c.db.events(s["id"]), ensure_ascii=False, indent=1))
            z.writestr(f"{s['id']}/utterances.json", json.dumps(
                c.db.query("SELECT * FROM utterances WHERE session_id=? ORDER BY start_s", (s["id"],)), ensure_ascii=False, indent=1))
            d = c.settings.data_dir / "sessions" / s["id"]
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix != ".pcm":
                        z.write(f, f"{s['id']}/audio/{f.relative_to(d)}")
            if s["scene_id"]:
                z.writestr(f"{s['id']}/scene_history.json", json.dumps(c.store.history(s["scene_id"]), ensure_ascii=False))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{pid}_data.zip"'})


# ======================= 会谈 =======================
class SessionIn(BaseModel):
    participant_id: str
    condition: Condition = "personal"
    duration_min: int = Field(30, ge=5, le=90)


@app.post("/api/sessions")
def create_session(body: SessionIn):
    c = ctx()
    p = c.db.one("SELECT * FROM participants WHERE id=?", (body.participant_id,))
    if not p:
        raise HTTPException(404, "参与者不存在")
    if not p["consent"]:
        raise HTTPException(403, "未勾选知情同意，不能开始会谈")
    region = p["region_style"] or "其他"
    if body.condition == "personal":
        scene_id = p["scene_id"]
        if not scene_id or not c.store.get(scene_id):
            scene_id = c.scenes.create_scene(p["id"], "personal", region).scene_id
            c.db.execute("UPDATE participants SET scene_id=? WHERE id=?", (scene_id, p["id"]))
    else:
        scene_id = c.scenes.create_scene(p["id"], body.condition, region).scene_id
    sid = new_id("ses")
    c.db.execute("INSERT INTO sessions (id, participant_id, condition, duration_min, region_style, scene_id, status) "
                 "VALUES (?,?,?,?,?,?, 'created')", (sid, p["id"], body.condition, body.duration_min, region, scene_id))
    c.db.log_event(sid, "session_created", body.model_dump() | {"scene_id": scene_id})
    return {"session_id": sid, "scene_id": scene_id, "url": f"/?session={sid}"}


@app.get("/api/sessions")
def list_sessions(participant_id: Optional[str] = None):
    if participant_id:
        return ctx().db.query("SELECT * FROM sessions WHERE participant_id=? ORDER BY started_at", (participant_id,))
    return ctx().db.query("SELECT * FROM sessions ORDER BY started_at")


@app.get("/api/sessions/{sid}/events")
def session_events(sid: str):
    return ctx().db.events(sid)


@app.get("/api/sessions/{sid}/transcript")
def session_transcript(sid: str):
    return ctx().db.query("SELECT speaker, text, start_s, end_s, labels FROM utterances WHERE session_id=? ORDER BY start_s",
                          (sid,))


@app.websocket("/ws/session/{sid}")
async def ws_session(ws: WebSocket, sid: str):
    await ws.accept()
    c = ctx()
    row = c.db.one("SELECT status FROM sessions WHERE id=?", (sid,))
    if not row:
        await ws.close(code=4404)
        return
    if row["status"] == "ended":
        await ws.send_text(json.dumps({"type": "status", "text": "这场会谈已经结束。请在设置页新建会谈。"}, ensure_ascii=False))
        await ws.close()
        return
    rt = get_runtime(c, sid)
    await rt.attach(ws)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await rt.on_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except ValueError:
                    continue
                await rt.on_json(data)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("会谈连接出错")
    finally:
        rt.detach(ws)


# ======================= 场景与故事 =======================
@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str):
    m = ctx().store.get(scene_id)
    if not m:
        raise HTTPException(404)
    return {"manifest": m.model_dump(mode="json"), "assets": ctx().scenes.assets_for(m)}


@app.get("/api/scenes/{scene_id}/history")
def scene_history(scene_id: str):
    return ctx().store.history(scene_id)


class StoryPatch(BaseModel):
    include_in_export: bool


@app.patch("/api/scenes/{scene_id}/stories/{story_id}")
async def patch_story(scene_id: str, story_id: str, body: StoryPatch):
    def f(w):
        for s in w.stories:
            if s.id == story_id:
                s.include_in_export = body.include_in_export
    m = await ctx().scenes.system_update(scene_id, f, {"kind": "staff", "action": "story_export_flag"})
    return {"ok": True, "version": m.version if m else None}


# ======================= 素材 =======================
@app.get("/api/assets")
def list_assets(kind: Optional[str] = None):
    out = []
    for a in ctx().registry.all():
        if kind and a.kind != kind:
            continue
        d = a.model_dump(mode="json")
        d["urls"] = [ctx().scenes.url(f) for f in a.files]
        out.append(d)
    return out


class AssetPatch(BaseModel):
    review_status: Literal["pending", "approved", "rejected"]


@app.patch("/api/assets/{asset_id}")
def patch_asset(asset_id: str, body: AssetPatch):
    c = ctx()
    a = c.registry.get(asset_id)
    if not a:
        raise HTTPException(404)
    c.registry.register(a.model_copy(update={"review_status": body.review_status}), private=asset_id in c.registry.private)
    c.resolver.refresh()
    return {"ok": True}


# ======================= 报告与导出 =======================
@app.post("/api/sessions/{sid}/analyze")
async def analyze(sid: str):
    from .research.analysis import analyze_session
    return await analyze_session(ctx(), sid)


@app.get("/api/sessions/{sid}/report")
def session_report(sid: str):
    p = ctx().settings.data_dir / "reports" / sid / "report.html"
    if not p.exists():
        raise HTTPException(404, "报告尚未生成（会谈结束后自动生成，也可以调用 analyze）")
    return FileResponse(p)


@app.get("/api/sessions/{sid}/report.csv")
def session_report_csv(sid: str):
    p = ctx().settings.data_dir / "reports" / sid / "metrics.csv"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=f"{sid}_metrics.csv")


@app.get("/api/reports/compare")
def compare_report(fmt: str = "html"):
    from .research.report import build_comparison
    html, csv_text = build_comparison(ctx())
    if fmt == "csv":
        return Response(csv_text, media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="comparison.csv"'})
    return HTMLResponse(html)


class ExportIn(BaseModel):
    story_ids: Optional[list[str]] = None  # None = 使用每个故事的 include_in_export


@app.post("/api/participants/{pid}/roam")
async def export_roam(pid: str, body: ExportIn):
    from .export.roam import build_roam
    c = ctx()
    p = c.db.one("SELECT * FROM participants WHERE id=?", (pid,))
    if not p or not p["scene_id"]:
        raise HTTPException(404, "这位参与者还没有个人场景")
    try:
        path = await asyncio.get_running_loop().run_in_executor(None, build_roam, c, p["scene_id"], pid, body.story_ids)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    c.db.log_event(None, "roam_exported", {"participant_id": pid, "file": path.name})
    return {"ok": True, "url": f"/api/exports/{pid}/{path.name}", "size_mb": round(path.stat().st_size / 1e6, 1)}


@app.get("/api/exports/{pid}/{name}")
def get_export(pid: str, name: str):
    if not PID_RE.match(pid) or "/" in name or ".." in name:
        raise HTTPException(400)
    p = ctx().settings.data_dir / "exports" / pid / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, filename=name)


# ======================= 文件 =======================
@app.get("/files/{path:path}")
def files(path: str):
    """只开放素材目录、场景图片和故事录音。"""
    c = ctx()
    full = to_abs(path).resolve()
    allowed = [c.settings.assets_dir, c.settings.data_dir / "scenes", c.settings.data_dir / "sessions"]
    if not any(str(full).startswith(str(a.resolve()) + "/") for a in allowed) or not full.is_file():
        raise HTTPException(404)
    if full.suffix in (".pcm", ".wav") and "stories" not in full.parts:
        raise HTTPException(404)
    return FileResponse(full, headers={"Cache-Control": "public, max-age=86400"})


# ======================= 前端 =======================
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
else:
    @app.get("/")
    def no_frontend():
        return HTMLResponse("<p>前端尚未构建：请在 frontend/ 目录运行 <code>npm install && npm run build</code>，"
                            "或开发时运行 <code>npm run dev</code> 并访问 http://localhost:5173 。</p>")
