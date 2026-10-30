"""会谈编排器：一场会谈的全部实时逻辑。

- 麦克风音频 → 录音文件 + 实时识别；老人沉默 ≥ REPLY_SILENCE_S 秒才处理这一轮并回应；
- 控制短语（太吵了/停一下/今天就到这儿）走快速通道，立即执行；
- 每轮：话语理解（多标签）→ 控制 / 确认 / 导航 / 修改（高置信度直接执行，低置信度先确认）/ 故事归档 → 陪聊回应；
- personal 条件下，第一次会谈边聊边抽取元素、逐步生成场景；
- 所有事件带时间戳写入事件日志。任何外部服务失败都降级，绝不把错误显示给老人。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import subprocess
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..config import to_rel
from ..context import AppContext
from ..providers.asr import ASRResult, BYTES_PER_SECOND, NullASRSession
from ..scene.commands import CommandError, EditCommand, RawCommand, to_typed
from ..scene.executor import create_viewpoint
from ..scene.models import Anchor, SceneManifest, Story, new_id
from . import demo_brain, prompts
from .schemas import CompanionTurn, SceneUpdate, StoryArchive, UtteranceAnalysis

log = logging.getLogger("session")


@dataclass
class Sentence:
    text: str
    start_s: float
    end_s: float


@dataclass
class Pending:
    commands: list[EditCommand]
    question: str
    at: float
    raw_text: str
    kind: str = "edit"  # edit | end（问老人要不要今天先聊到这儿）


@dataclass
class SessionState:
    phase: str = "waiting"  # waiting → greeting → gathering → exploring → closing → ended
    turn: list[Sentence] = field(default_factory=list)
    elder_speaking: bool = False
    vad_since: float = 0.0
    gate_seen: bool = False  # 前端是否发送“可能有人说话”的门控信号
    gate_open: bool = False
    gate_changed: float = 0.0
    last_asr_activity: float = 0.0
    last_elder_activity: float = 0.0
    companion_speaking: bool = False
    last_companion_end: float = 0.0
    last_idle_prompt: float = 0.0
    pending: Optional[Pending] = None
    current_vp: Optional[str] = None
    prev_vp: Optional[str] = None
    quiet: bool = False
    dialogue: deque = field(default_factory=lambda: deque(maxlen=12))
    asked: list[str] = field(default_factory=list)
    emotions: deque = field(default_factory=lambda: deque(maxlen=4))
    narration_turns: int = 0
    last_edit: Optional[dict] = None
    pending_nav_vp: Optional[str] = None
    scene_ready_announced: bool = False
    fast_quiet_handled: bool = False
    end_requested: bool = False
    tired_offer_made: bool = False


class SessionRuntime:
    IDLE_PROMPT_S = 50.0

    def __init__(self, ctx: AppContext, session_id: str):
        self.ctx = ctx
        self.db = ctx.db
        self.s = ctx.settings
        row = self.db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
        if not row:
            raise KeyError(session_id)
        self.row = row
        self.session_id = session_id
        self.scene_id: str = row["scene_id"]
        self.condition: str = row["condition"]
        self.duration_s = int(row["duration_min"]) * 60
        self.st = SessionState()
        self.ws = None
        self.asr = None
        self.dir = self.s.data_dir / "sessions" / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pcm_path = self.dir / "mic.pcm"
        self._pcm = open(self.pcm_path, "ab")
        self.audio_bytes = self.pcm_path.stat().st_size
        self.started_mono = 0.0
        self.lock = asyncio.Lock()
        self.tick_task: Optional[asyncio.Task] = None
        self.tasks: set[asyncio.Task] = set()
        self._say_ids: dict[str, dict] = {}
        self.reply_silence = self.s.reply_silence_s
        ctx.scenes.subscribe(self.scene_id, self._on_scene_change)

    # ================= 基础 =================
    @property
    def t(self) -> float:
        """会谈录音时间（秒）。"""
        return self.audio_bytes / BYTES_PER_SECOND

    def now(self) -> float:
        return time.monotonic()

    def log(self, type_: str, payload: dict) -> None:
        self.db.log_event(self.session_id, type_, payload, t_rel=round(self.t, 2))

    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def send(self, msg: dict) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def scene(self) -> SceneManifest:
        m = self.ctx.store.get(self.scene_id)
        assert m is not None
        return m

    async def push_scene(self, m: Optional[SceneManifest] = None, highlight: Optional[list[str]] = None) -> None:
        m = m or self.scene()
        await self.send({"type": "scene", "manifest": m.model_dump(mode="json"),
                         "assets": self.ctx.scenes.assets_for(m), "highlight": highlight or []})

    async def staff_status(self) -> None:
        elapsed = int(self.now() - self.started_mono) if self.started_mono else 0
        await self.send({"type": "status", "text": f"条件：{self.condition}｜阶段：{self.st.phase}｜已进行 {elapsed // 60} 分 {elapsed % 60} 秒"
                         f"｜{'安静中' if self.st.quiet else ''}{'｜等待确认' if self.st.pending else ''}"
                         f"｜服务：{'云端' if self.ctx.providers.cloud else '演示模式'}"})

    # ================= 连接 =================
    async def attach(self, ws) -> None:
        self.ws = ws
        await self.push_scene()
        await self.staff_status()

    def detach(self, ws) -> None:
        if self.ws is ws:
            self.ws = None

    async def start(self) -> None:
        if self.st.phase != "waiting":
            return
        self.started_mono = self.now()
        self.st.last_elder_activity = self.now()
        self.db.execute("UPDATE sessions SET started_at=?, status='running' WHERE id=?", (time.time(), self.session_id))
        self.log("session_start", {"condition": self.condition, "scene_id": self.scene_id,
                                   "providers": self.ctx.providers.describe(), "scene_version": self.scene().version})
        m = self.scene()
        self.st.current_vp = m.entry_viewpoint
        if self.condition in ("generic", "audio_only") and not m.viewpoints:
            m = await self.ctx.scenes.build_generic(self.scene_id, self.session_id)
            self.st.current_vp = m.entry_viewpoint
            await self.push_scene(m)
        self.st.phase = "gathering" if (self.condition == "personal" and not m.viewpoints) else "exploring"
        if self.st.phase == "exploring":
            self.st.scene_ready_announced = True
        try:
            self.asr = self.ctx.providers.asr_factory()
            await self.asr.start(self.on_asr)
        except Exception as e:
            log.warning("语音识别启动失败：%s", e)
            self.log("asr_error", {"error": str(e)[:200]})
        self.tick_task = asyncio.create_task(self.tick_loop())
        await self.staff_status()
        greet_phase = "greeting" if self.st.phase == "gathering" else "exploring"
        await self.companion_say(greet_phase, opening=True)

    # ================= 输入 =================
    async def on_audio(self, data: bytes) -> None:
        self._pcm.write(data)
        self.audio_bytes += len(data)
        if self.asr is not None and self.st.phase not in ("ended",):
            try:
                await self.asr.feed(data, self.asr_active())
            except Exception:
                pass

    def asr_active(self) -> bool:
        """省钱：只在“可能有人说话”时把音频送去识别（门槛比 VAD 更低，关上后再多送 2 秒）。"""
        if not self.s.asr_gating or not self.st.gate_seen:
            return True
        return self.st.gate_open or self.now() - self.st.gate_changed < 2.0

    async def on_json(self, msg: dict) -> None:
        typ = msg.get("type")
        if typ == "hello":
            # 断线重连：前端的录音时间更靠后时补静音，保证时间轴一致
            t = float(msg.get("t") or 0)
            gap = int((t - self.t) * BYTES_PER_SECOND) // 2 * 2
            if 0 < gap < BYTES_PER_SECOND * 600:
                await self.on_audio(b"\x00" * gap)
            if msg.get("viewpoint_id"):
                self.st.current_vp = msg["viewpoint_id"]
            await self.start()
        elif typ == "vad":
            self.st.elder_speaking = bool(msg.get("speaking"))
            self.st.vad_since = self.now()
            self.st.last_elder_activity = self.now()
            self.log("vad", {"speaking": self.st.elder_speaking, "t": msg.get("t")})
        elif typ == "gate":
            self.st.gate_seen = True
            self.st.gate_open = bool(msg.get("open"))
            self.st.gate_changed = self.now()
        elif typ == "barge_in":
            self.log("barge_in", {"t": msg.get("t")})
        elif typ == "tts_state":
            await self.on_tts_state(msg)
        elif typ == "viewpoint_changed":
            vp = msg.get("viewpoint_id")
            if vp and vp != self.st.current_vp:
                self.st.prev_vp, self.st.current_vp = self.st.current_vp, vp
            self.log("viewpoint_changed", {"viewpoint_id": vp})
        elif typ == "staff_text":
            # 演示/测试：工作人员输入老人的话（等同于一句识别结果）
            text = str(msg.get("text", "")).strip()
            if text:
                t0 = self.t
                await self.on_asr(ASRResult(text, t0, t0 + max(1.0, len(text) * 0.25), True), typed=True)
                self.st.last_elder_activity = self.now() - max(0.0, self.reply_silence - 0.8)
        elif typ == "staff":
            action = msg.get("action")
            self.log("staff_action", {"action": action})
            if action == "end":
                await self.begin_closing("staff")
            elif action == "quiet":
                await self.set_quiet(True, "staff")
            elif action == "unquiet":
                await self.set_quiet(False, "staff")

    async def on_tts_state(self, msg: dict) -> None:
        sid = msg.get("id", "")
        info = self._say_ids.get(sid)
        if msg.get("playing"):
            self.st.companion_speaking = True
            if info is not None:
                info["start_s"] = float(msg.get("t") or self.t)
        else:
            self.st.companion_speaking = False
            self.st.last_companion_end = self.now()
            if info is not None:
                start = info.get("start_s", float(msg.get("t") or self.t))
                end = float(msg.get("t") or self.t)
                self.db.execute("INSERT OR REPLACE INTO utterances (id, session_id, speaker, text, start_s, end_s, labels, turn_id, created_at) "
                                "VALUES (?,?,?,?,?,?,?,?,?)",
                                (sid, self.session_id, "companion", info["text"], start, max(end, start), json.dumps([info["phase"]]),
                                 None, time.time()))
                if sid == "goodbye":
                    await self.finish()

    async def on_asr(self, r: ASRResult, typed: bool = False) -> None:
        if self.st.phase == "ended":
            return
        await self.send({"type": "transcript", "speaker": "elder", "text": r.text, "final": r.final})
        self.st.last_elder_activity = self.now()
        self.st.last_asr_activity = self.now()
        if not r.final:
            return
        # 识别出的系统自己的声音（回声）：陪聊说话期间且与最近一句陪聊高度重合时丢弃
        if self.st.companion_speaking and self._looks_like_echo(r.text):
            self.log("echo_dropped", {"text": r.text})
            return
        self.st.turn.append(Sentence(r.text, r.start_s, r.end_s))
        self.db.execute("INSERT INTO utterances (id, session_id, speaker, text, start_s, end_s, labels, turn_id, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (new_id("u"), self.session_id, "elder", r.text, r.start_s, r.end_s, "[]", None, time.time()))
        self.log("asr_final", {"text": r.text, "start_s": r.start_s, "end_s": r.end_s, "typed": typed})
        await self.fast_path(r.text)

    def _looks_like_echo(self, text: str) -> bool:
        recent = [d["text"] for d in list(self.st.dialogue)[-2:] if d["speaker"] == "companion"]
        def grams(s):
            s = re.sub(r"[^一-龥]", "", s)
            return {s[i:i + 2] for i in range(len(s) - 1)}
        g = grams(text)
        return any(g and len(g & grams(c)) / len(g) > 0.6 for c in recent)

    async def fast_path(self, text: str) -> None:
        """控制短语快速通道：不等 4 秒沉默。"""
        cp = self.ctx.catalog.control_phrases
        short = len(re.sub(r"[^一-龥]", "", text)) <= 8
        if any(p in text for p in cp.get("quiet", [])) and (short or "太吵" in text or "安静" in text):
            await self.set_quiet(True, "elder")
            self.st.fast_quiet_handled = True
        elif any(p in text for p in cp.get("end", [])):
            self.st.end_requested = True
        elif self.st.quiet and short and any(p in text for p in cp.get("resume", [])):
            await self.set_quiet(False, "elder")

    async def set_quiet(self, on: bool, by: str) -> None:
        if self.st.quiet == on:
            return
        self.st.quiet = on
        await self.send({"type": "quiet" if on else "unquiet"})
        self.log("quiet" if on else "unquiet", {"by": by})
        await self.staff_status()

    # ================= 轮次 =================
    async def tick_loop(self) -> None:
        try:
            while self.st.phase != "ended":
                await asyncio.sleep(0.25)
                await self.tick()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("tick 出错")
            if self.st.phase != "ended":
                self.tick_task = asyncio.create_task(self.tick_loop())

    def elder_is_speaking(self, now: float) -> bool:
        """老人是否正在说话。真实说话时识别服务每几百毫秒就会给出中间结果；
        VAD 说在说话、但 3 秒内识别没有任何结果 → 多半是持续噪声（电视、风扇），不再相信 VAD。
        演示模式没有识别服务，VAD 不参与判断（以工作人员输入为准）。"""
        if not self.st.elder_speaking or self.asr is None or isinstance(self.asr, NullASRSession):
            return False
        return now - max(self.st.last_asr_activity, self.st.vad_since) < 3.0

    async def tick(self) -> None:
        now = self.now()
        if self.st.companion_speaking and now - self.st.last_companion_end > 45:
            self.st.companion_speaking = False  # 前端没有回报播放结束时的兜底
        speaking = self.elder_is_speaking(now)
        last = self.st.last_elder_activity if speaking or not self.st.elder_speaking else \
            max(self.st.last_asr_activity, self.st.vad_since)
        silence = now - last
        if self.st.phase in ("closing", "ended") or self.lock.locked():
            return
        if self.st.end_requested and not speaking and silence > 1.5:
            self.st.end_requested = False
            await self.begin_closing("elder")
            return
        if self.st.turn and not speaking and silence >= self.reply_silence:
            sentences, self.st.turn = self.st.turn, []
            self.spawn(self.process_turn(sentences))
            return
        if self.started_mono and now - self.started_mono > self.duration_s and not speaking and silence > 2:
            await self.begin_closing("time_limit")
            return
        # 长时间安静：偶尔轻轻地邀请看看/听听（大部分时间安静陪伴）
        quiet_for = min(silence, now - self.st.last_companion_end if self.st.last_companion_end else silence)
        if (self.st.phase in ("gathering", "exploring") and not self.st.quiet and not self.st.companion_speaking
                and not self.st.turn and quiet_for > self.IDLE_PROMPT_S and now - self.st.last_idle_prompt > self.IDLE_PROMPT_S * 1.5):
            self.st.last_idle_prompt = now
            self.spawn(self.companion_say("idle"))

    async def process_turn(self, sentences: list[Sentence]) -> None:
        async with self.lock:
            text = "".join(s.text for s in sentences).strip()
            if not text:
                return
            start, end = sentences[0].start_s, sentences[-1].end_s
            turn_id = new_id("turn")
            self.st.dialogue.append({"speaker": "elder", "text": text})
            await self.staff_status()
            m = self.scene()
            analysis = await self.understand(text, m)
            self.log("utterance", {"turn_id": turn_id, "text": text, "start_s": start, "end_s": end,
                                   "analysis": analysis.model_dump()})
            self.db.execute("UPDATE utterances SET labels=?, turn_id=? WHERE session_id=? AND speaker='elder' "
                            "AND start_s>=? AND end_s<=?", (json.dumps(analysis.labels), turn_id, self.session_id,
                                                            start - 0.01, end + 0.01))
            self.st.emotions.append(analysis.emotion)
            actions: list[str] = []
            confirm_q: Optional[str] = None

            # ---- 控制 ----
            ck = analysis.control_kind
            if ck == "quiet" and not self.st.fast_quiet_handled:
                await self.set_quiet(True, "elder")
            self.st.fast_quiet_handled = False
            if ck == "quiet":
                return  # 安静时陪聊者也不说话
            if ck == "resume":
                await self.set_quiet(False, "elder")
            if ck == "end":
                await self.begin_closing("elder")
                return
            # ---- 待确认的修改 ----
            if self.st.pending:
                p = self.st.pending
                self.st.pending = None
                if analysis.confirmation == "yes" and p.kind == "end":
                    self.log("confirm", {"answer": "yes", "question": p.question, "kind": "end"})
                    await self.begin_closing("tired")
                    return
                if analysis.confirmation == "yes":
                    await self.run_commands(p.commands, p.raw_text, actions, confirmed=True)
                    self.log("confirm", {"answer": "yes", "question": p.question})
                elif analysis.confirmation == "no":
                    self.log("confirm", {"answer": "no", "question": p.question, "kind": p.kind})
                    actions.append("老人还想接着聊，那就慢慢接着聊" if p.kind == "end" else
                                   "老人说不是这样，没有修改；可以请老人说说应该是什么样")
                else:
                    self.log("confirm", {"answer": "none", "question": p.question})
            elif ck == "undo" and self.condition == "personal":
                new = await self.ctx.scenes.undo(self.scene_id)
                self.log("undo", {"ok": bool(new), "version": new.version if new else None})
                actions.append("把刚才的修改改回去了" if new else "没有可以改回去的修改")
            # ---- 疲惫 ----
            recent = list(self.st.emotions)[-3:]
            if recent.count("tired") >= 2 and not self.st.tired_offer_made and not self.st.pending:
                self.st.tired_offer_made = True
                q = "您是不是有点累了？要不我们今天就先聊到这儿？"
                self.st.pending = Pending([], q, self.now(), text, kind="end")
                self.log("tired_offer", {"emotions": recent})
                confirm_q = q
            elif sum(1 for e in recent if e in ("sad", "anxious")) >= 2:
                actions.append("老人有些难过：先温和地安慰一句，然后轻轻换一个轻松愉快的话题（比如小时候好玩的事、好吃的东西），不要追问")
            # ---- 导航 ----
            if "navigate" in analysis.labels and analysis.navigate:
                desc = await self.navigate(analysis.navigate, text)
                if desc:
                    actions.append(desc)
            # ---- 修改 / 讲述抽取 ----
            if self.condition == "personal" and ck not in ("undo",) and analysis.confirmation == "none":
                if self.st.phase == "gathering":
                    confirm_q = await self.narration(text, actions) or confirm_q
                elif "edit" in analysis.labels:
                    confirm_q = await self.edit(text, actions) or confirm_q
            elif "edit" in analysis.labels:
                self.log("edit_ignored", {"reason": f"条件 {self.condition} 不做修改", "text": text})
            # ---- 故事 ----
            if "story" in analysis.labels or (self.st.phase == "gathering" and len(text) >= 12):
                self.spawn(self.archive_story(text, start, end, turn_id))
            # ---- 回应 ----
            if self.st.turn or self.elder_is_speaking(self.now()):
                self.log("reply_skipped", {"reason": "老人又开口了"})
                return
            emotion = analysis.emotion
            await self.companion_say(self.st.phase, actions=actions, confirm_q=confirm_q, emotion=emotion)

    async def understand(self, text: str, m: SceneManifest) -> UtteranceAnalysis:
        ctx = {"last_companion": next((d["text"] for d in reversed(self.st.dialogue) if d["speaker"] == "companion"), None),
               "confirm_question": self.st.pending.question if self.st.pending else None,
               "last_edit": self.st.last_edit, "scene": m.summary_for_llm(self.st.current_vp)}
        msgs = prompts.understand_messages(text, ctx)
        try:
            return await self.ctx.providers.llm.structured(msgs, UtteranceAnalysis, model=self.ctx.providers.llm_fast_model)
        except Exception as e:
            self.log("llm_error", {"step": "understand", "error": str(e)[:200]})
            return UtteranceAnalysis.model_validate(demo_brain.understand(json.loads(msgs[-1]["content"])))

    # ================= 修改 =================
    def _typed(self, raws: list[RawCommand]) -> tuple[list[EditCommand], list[dict]]:
        ok, bad = [], []
        for r in raws:
            try:
                ok.append(to_typed(r, self.st.current_vp))
            except (CommandError, ValueError) as e:
                bad.append({"raw": r.model_dump(), "error": str(e)})
        return ok, bad

    async def edit(self, text: str, actions: list[str]) -> Optional[str]:
        m = self.scene()
        ctx = {"dialogue": list(self.st.dialogue), "current_viewpoint": self.st.current_vp,
               "scene": m.summary_for_llm(self.st.current_vp)}
        try:
            upd = await self.ctx.providers.llm.structured(prompts.command_messages(text, ctx, "edit"), SceneUpdate,
                                                         model=self.ctx.providers.llm_model)
        except Exception as e:
            self.log("llm_error", {"step": "commands", "error": str(e)[:200]})
            return None
        return await self.handle_commands(upd.commands, text, actions)

    async def handle_commands(self, raws: list[RawCommand], text: str, actions: list[str],
                              added_by: str = "edit") -> Optional[str]:
        cmds, bad = self._typed(raws)
        for b in bad:
            self.log("command_rejected", b)
        thr = self.s.confidence_threshold
        clar = [c for c in cmds if c.op == "clarify"]
        high = [c for c in cmds if c.op != "clarify" and c.confidence >= thr]
        low = [c for c in cmds if c.op != "clarify" and c.confidence < thr]
        if high:
            await self.run_commands(high, text, actions, added_by=added_by)
        if low:
            q = next((c.confirm_question for c in low if c.confirm_question), None) or "您是说要这样改，对吗？"
            self.st.pending = Pending(low, q, self.now(), text)
            self.log("confirm_asked", {"question": q, "commands": [c.model_dump() for c in low]})
            return q
        if clar:
            self.log("clarify", {"question": clar[0].question})
            return clar[0].question
        return None

    async def run_commands(self, cmds: list[EditCommand], text: str, actions: list[str], confirmed: bool = False,
                           added_by: str = "edit") -> None:
        cause = {"kind": "narration" if added_by == "narration" else "edit", "session_id": self.session_id,
                 "text": text, "confirmed": confirmed}
        m, results = await self.ctx.scenes.apply(self.scene_id, cmds, cause, self.st.current_vp, added_by,  # type: ignore[arg-type]
                                                 self.session_id)
        for cmd, r, err in results:
            self.log("command", {"op": cmd.op, "command": cmd.model_dump(), "confidence": cmd.confidence,
                                 "executed": bool(r and r.applied), "message": r.message if r else err,
                                 "confirmed": confirmed, "added_by": added_by, "version": m.version})
            if r and r.applied:
                actions.append(r.message)
                self.st.last_edit = {"op": cmd.op, "message": r.message}
                for j in r.jobs:
                    if j.kind == "panorama" and cmd.op == "add_viewpoint":
                        self.st.pending_nav_vp = j.target_id

    async def narration(self, text: str, actions: list[str]) -> Optional[str]:
        """第一次会谈：从讲述中抽取元素，逐步生成场景。"""
        self.st.narration_turns += 1
        m = self.scene()
        ctx = {"dialogue": list(self.st.dialogue), "current_viewpoint": self.st.current_vp,
               "scene": m.summary_for_llm(self.st.current_vp)}
        try:
            upd = await self.ctx.providers.llm.structured(prompts.command_messages(text, ctx, "narration"), SceneUpdate,
                                                         model=self.ctx.providers.llm_model)
        except Exception as e:
            self.log("llm_error", {"step": "narration", "error": str(e)[:200]})
            return None
        self.log("narration_extract", upd.model_dump())

        def meta(w: SceneManifest):
            if upd.meta:
                for k, v in upd.meta.model_dump().items():
                    if v:
                        setattr(w, k, v)
            if upd.viewpoint_description and w.viewpoints:
                vp = w.viewpoint(self.st.current_vp or w.entry_viewpoint) or w.viewpoints[0]
                vp.description_zh = (vp.description_zh + "；" + upd.viewpoint_description).strip("；")[-600:]
            return []

        await self.ctx.scenes.system_update(self.scene_id, meta, {"kind": "narration", "step": "meta"})
        m = self.scene()
        # 第一张全景：知道地域、并且有了一些描述（或聊了两轮）就开始画
        if not m.viewpoints and (upd.viewpoint_description or self.st.narration_turns >= 2 or
                                 (upd.meta and upd.meta.region_style)):
            name = self.ctx.catalog.style["regions"].get(m.region_style, {}).get("default_viewpoints", ["家门口"])[0]
            desc = upd.viewpoint_description or ""
            holder: dict = {}

            def first_vp(w: SceneManifest):
                vp = create_viewpoint(w, name, desc)
                holder["id"] = vp.id
                from ..scene.executor import refresh_auto_ambience
                refresh_auto_ambience(w, self.ctx.resolver)
                return [vp.id]

            await self.ctx.scenes.system_update(self.scene_id, first_vp, {"kind": "narration", "step": "first_viewpoint"})
            self.st.current_vp = holder["id"]
            from ..scene.executor import Job
            self.ctx.scenes.spawn(self.ctx.scenes.run_job(self.scene_id, Job("panorama", holder["id"], {"mode": "new"}),
                                                          self.session_id))
            self.log("first_panorama_requested", {"viewpoint_id": holder["id"], "description": desc})
        if self.scene().viewpoints and upd.commands:
            # 只取物件/环境声/新点位类指令，逐个加入
            keep = [c for c in upd.commands if c.op in ("add_object", "add_ambience", "add_viewpoint", "set_season",
                                                         "set_time", "set_weather")]
            # 讲述阶段边聊边加：不需要陪聊者说“改好了”，所以不把这些动作告诉陪聊者
            return await self.handle_commands(keep[:4], text, [], added_by="narration")
        return None

    # ================= 导航 =================
    async def navigate(self, nav, text: str) -> Optional[str]:
        m = self.scene()
        cur = self.st.current_vp
        turn = {"left": -70, "right": 70, "back": 180, "front": 0}
        self.log("navigate_intent", nav.model_dump())
        if nav.kind == "look_direction":
            await self.send({"type": "navigate", "turn": turn.get(nav.direction or "left", -70)})
            return None
        if nav.kind == "go_back":
            if self.st.prev_vp and m.viewpoint(self.st.prev_vp):
                await self.send({"type": "navigate", "viewpoint_id": self.st.prev_vp})
                return f"回到了{m.viewpoint(self.st.prev_vp).name_zh}"
            return None
        lm = m.landmark(nav.landmark_id) if nav.landmark_id else None
        vp = m.viewpoint(nav.viewpoint_id) if nav.viewpoint_id else None
        name = nav.place_name or ""
        if not vp and name:
            vp = m.find_viewpoint_by_name(name)
        if not lm and name:
            lm = m.find_landmark_by_name(name, cur) or m.find_landmark_by_name(name)
        if vp and vp.id != cur and vp.panorama_path:
            await self.send({"type": "navigate", "viewpoint_id": vp.id})
            return f"去了{vp.name_zh}"
        if lm:
            if nav.kind == "go_place" and lm.linked_viewpoint_id and m.viewpoint(lm.linked_viewpoint_id).panorama_path:
                await self.send({"type": "navigate", "viewpoint_id": lm.linked_viewpoint_id})
                return f"走到了{lm.name_zh}"
            if lm.viewpoint_id != cur:
                await self.send({"type": "navigate", "viewpoint_id": lm.viewpoint_id, "yaw": lm.yaw_deg})
            else:
                await self.send({"type": "navigate", "yaw": lm.yaw_deg})
            if nav.kind == "look_place" or self.condition != "personal":
                return f"转过去看{lm.name_zh}"
        if nav.kind == "go_place" and name and self.condition == "personal" and m.viewpoints:
            if vp and not vp.panorama_path:
                return f"{vp.name_zh}还在画，画好了就过去"
            from ..scene.commands import AddViewpoint
            actions: list[str] = []
            await self.run_commands([AddViewpoint(name_zh=name, description="", connect_to=cur, confidence=0.9)], text, actions)
            return f"正在照着老人说的去画{name}，画好了就带老人过去"
        return None

    async def _on_scene_change(self, m: SceneManifest, highlight: list[str]) -> None:
        await self.push_scene(m, highlight)
        # 第一张全景完成 → 邀请老人看看
        if self.condition == "personal" and not self.st.scene_ready_announced and any(v.panorama_path for v in m.viewpoints):
            self.st.scene_ready_announced = True
            if self.st.phase == "gathering":
                self.st.phase = "exploring"
            self.log("scene_ready", {"version": m.version})
            self.spawn(self._announce_ready())
        if self.st.pending_nav_vp:
            vp = m.viewpoint(self.st.pending_nav_vp)
            if vp and vp.panorama_path:
                self.st.pending_nav_vp = None
                await self.send({"type": "navigate", "viewpoint_id": vp.id})
                self.log("auto_navigate", {"viewpoint_id": vp.id})

    async def _announce_ready(self) -> None:
        # 等老人说完、等当前一轮处理完再说
        for _ in range(240):
            if not self.elder_is_speaking(self.now()) and not self.st.turn and not self.lock.locked() \
                    and self.now() - self.st.last_elder_activity > 2.0 and not self.st.companion_speaking:
                break
            await asyncio.sleep(0.5)
        await self.companion_say("scene_ready")

    # ================= 故事 =================
    async def archive_story(self, text: str, start_s: float, end_s: float, turn_id: str) -> None:
        try:
            m = self.scene()
            if not m.viewpoints:
                # 场景还没有点位：先记下来，等第一张全景出来后挂到入口点位
                for _ in range(120):
                    await asyncio.sleep(1)
                    m = self.scene()
                    if m.viewpoints:
                        break
                if not m.viewpoints:
                    self.log("story_deferred_failed", {"text": text})
                    return
            ctx = {"current_viewpoint": self.st.current_vp or m.entry_viewpoint, "scene": m.summary_for_llm(self.st.current_vp)}
            msgs = prompts.story_messages(text, ctx)
            try:
                sa = await self.ctx.providers.llm.structured(msgs, StoryArchive, model=self.ctx.providers.llm_fast_model)
            except Exception as e:
                self.log("llm_error", {"step": "story", "error": str(e)[:200]})
                sa = StoryArchive.model_validate(demo_brain.story(json.loads(msgs[-1]["content"])))
            if not sa.is_story:
                return
            pool = {"object": m.obj, "landmark": m.landmark, "viewpoint": m.viewpoint}[sa.anchor_kind]
            anchor = Anchor(kind=sa.anchor_kind, id=sa.anchor_id) if pool(sa.anchor_id) else \
                Anchor(kind="viewpoint", id=self.st.current_vp or m.entry_viewpoint)
            sid = new_id("st")
            audio_rel = await asyncio.get_running_loop().run_in_executor(None, self.cut_audio, start_s, end_s, sid)
            story = Story(id=sid, title_zh=sa.title_zh[:12], transcript=text, audio_path=audio_rel, start_s=start_s,
                          end_s=end_s, anchor=anchor, session_id=self.session_id)

            def add(w: SceneManifest):
                pool2 = {"object": w.obj, "landmark": w.landmark, "viewpoint": w.viewpoint}[story.anchor.kind]
                if not pool2(story.anchor.id):
                    story.anchor = Anchor(kind="viewpoint", id=w.entry_viewpoint)
                w.stories.append(story)
                return [story.id]

            await self.ctx.scenes.system_update(self.scene_id, add, {"kind": "story", "session_id": self.session_id})
            self.log("story", {"story_id": sid, "turn_id": turn_id, "title": story.title_zh, "anchor": anchor.model_dump(),
                               "start_s": start_s, "end_s": end_s, "audio_path": audio_rel, "chars": len(text)})
        except Exception as e:
            log.exception("故事归档失败")
            self.log("story_failed", {"error": str(e)[:200]})

    def cut_audio(self, start_s: float, end_s: float, story_id: str) -> Optional[str]:
        """从会谈录音里截出这段讲述（前后各留一点余量），存为 mp3。"""
        try:
            self._pcm.flush()
            a = max(0, int((start_s - 0.4) * BYTES_PER_SECOND) // 2 * 2)
            b = int((end_s + 0.6) * BYTES_PER_SECOND) // 2 * 2
            with open(self.pcm_path, "rb") as f:
                f.seek(a)
                pcm = f.read(max(0, b - a))
            if len(pcm) < BYTES_PER_SECOND * 0.3:
                return None
            out_dir = self.dir / "stories"
            out_dir.mkdir(exist_ok=True)
            wav_path = out_dir / f"{story_id}.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm)
            mp3 = wav_path.with_suffix(".mp3")
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav_path), "-af", "highpass=f=80,loudnorm=I=-18",
                                "-b:a", "64k", str(mp3)], capture_output=True, timeout=60)
            final = mp3 if r.returncode == 0 else wav_path
            if final == mp3:
                wav_path.unlink(missing_ok=True)
            return to_rel(final)
        except Exception as e:
            log.warning("截取讲述录音失败：%s", e)
            return None

    # ================= 陪聊 =================
    async def companion_say(self, phase: str, actions: Optional[list[str]] = None, confirm_q: Optional[str] = None,
                            emotion: str = "neutral", opening: bool = False) -> None:
        if self.st.quiet and phase not in ("closing",):
            return
        m = self.scene()
        if self.condition != "personal" and actions:
            actions = [a for a in actions if "画" not in a]
        ctx = {"phase": "greeting" if opening and phase == "greeting" else phase, "scene": m.summary_for_llm(self.st.current_vp),
               "dialogue": list(self.st.dialogue)[-8:], "actions": actions or [], "confirm_question": confirm_q,
               "emotion": emotion, "asked": self.st.asked}
        if self.condition != "personal":
            ctx["actions"] = (ctx["actions"] or []) + ["（这次活动里画面保持原样，不做修改；不要说“改好了”）"]
        try:
            turn = await self.ctx.providers.llm.structured(prompts.companion_messages(ctx), CompanionTurn,
                                                          model=self.ctx.providers.llm_model, temperature=0.7)
        except Exception as e:
            self.log("llm_error", {"step": "companion", "error": str(e)[:200]})
            if confirm_q:
                turn = CompanionTurn(speak=True, text=confirm_q)
            elif phase == "closing":
                turn = CompanionTurn(speak=True, text="今天听您讲了这么多，真好。我们下次再聊。")
            else:
                return  # 想不出合适的话就安静陪着
        text = self.clean_text(turn.text)
        if not turn.speak or not text:
            self.log("companion_silent", {"phase": phase})
            if phase == "closing":
                text = "今天就聊到这儿，谢谢您。我们下次再聊。"
            else:
                return
        if self.st.turn or (self.elder_is_speaking(self.now()) and phase not in ("closing",)):
            self.log("reply_skipped", {"reason": "老人正在说话", "text": text})
            return
        await self.say(text, phase, "goodbye" if phase == "closing" else None)

    @staticmethod
    def clean_text(text: str) -> str:
        text = (text or "").strip().strip("“”\"")
        # 安全网：去掉“您还记得吗”这类考记忆的问法
        text = re.sub(r"您?还记得[吗么]?[？?]?", "", text)
        text = re.sub(r"(您)?记不记得", "", text)
        return text.strip()[:120]

    async def say(self, text: str, phase: str, sid: Optional[str] = None) -> None:
        sid = sid or uuid.uuid4().hex[:10]
        self._say_ids[sid] = {"text": text, "phase": phase}
        self.st.dialogue.append({"speaker": "companion", "text": text})
        if "？" in text or "?" in text:
            self.st.asked.append(text)
        audio_b64 = None
        try:
            audio = await self.ctx.providers.tts.synthesize(text)
            if audio:
                audio_b64 = base64.b64encode(audio).decode()
        except Exception as e:
            self.log("tts_error", {"error": str(e)[:200]})
        self.log("companion", {"id": sid, "text": text, "phase": phase, "tts": bool(audio_b64)})
        self.st.companion_speaking = True
        self.st.last_companion_end = self.now()
        await self.send({"type": "say", "id": sid, "text": text, "audio_b64": audio_b64})
        await self.send({"type": "transcript", "speaker": "companion", "text": text, "final": True})

    # ================= 结束 =================
    async def begin_closing(self, reason: str) -> None:
        if self.st.phase in ("closing", "ended"):
            return
        self.st.phase = "closing"
        self.log("closing", {"reason": reason})
        self.db.execute("UPDATE sessions SET end_reason=? WHERE id=?", (reason, self.session_id))
        if self.st.quiet:
            await self.set_quiet(False, "closing")
        await self.companion_say("closing")
        # 如果前端没有回报播放结束，30 秒后也结束
        self.spawn(self._finish_after(30))

    async def _finish_after(self, s: float) -> None:
        await asyncio.sleep(s)
        await self.finish()

    def closing_song(self) -> Optional[dict]:
        reg = self.ctx.registry
        songs = [a for a in reg.all() if a.kind in ("music", "sound") and "收尾老歌" in a.tags and a.usable and not a.sensitive]
        if not songs:
            songs = [a for a in reg.all() if a.kind == "music" and "老歌" in a.tags and a.usable and not a.sensitive]
        if not songs:
            return None
        a = sorted(songs, key=lambda x: x.review_status != "approved")[0]
        return {"url": self.ctx.scenes.url(a.files[0]), "profile": a.sound.acoustic_profile if a.sound else "gramophone",
                "asset_id": a.id}

    async def finish(self) -> None:
        if self.st.phase == "ended":
            return
        self.st.phase = "ended"
        song = self.closing_song()
        await self.send({"type": "end", "music_url": song["url"] if song else None,
                         "profile": song["profile"] if song else None})
        self.log("session_end", {"song": song["asset_id"] if song else None, "duration_s": round(self.t, 1)})
        self.db.execute("UPDATE sessions SET ended_at=?, status='ended' WHERE id=?", (time.time(), self.session_id))
        await self.close()
        try:
            from ..research.analysis import analyze_session
            self.ctx.scenes.spawn(analyze_session(self.ctx, self.session_id))
        except Exception as e:
            log.warning("会后分析启动失败：%s", e)

    async def close(self) -> None:
        if self.tick_task:
            self.tick_task.cancel()
        if self.asr is not None:
            try:
                await self.asr.close()
            except Exception:
                pass
            self.asr = None
        try:
            self._pcm.flush()
            self._pcm.close()
        except Exception:
            pass
        self.ctx.scenes.unsubscribe(self.scene_id, self._on_scene_change)
        # 生成完整录音 wav（研究用）
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._write_wav)
        except Exception:
            pass

    def _write_wav(self) -> None:
        with open(self.pcm_path, "rb") as f:
            pcm = f.read()
        with wave.open(str(self.dir / "mic.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)


_runtimes: dict[str, SessionRuntime] = {}


def get_runtime(ctx: AppContext, session_id: str) -> SessionRuntime:
    rt = _runtimes.get(session_id)
    if rt is None or rt.st.phase == "ended":
        rt = SessionRuntime(ctx, session_id)
        _runtimes[session_id] = rt
    return rt


def drop_runtime(session_id: str) -> None:
    _runtimes.pop(session_id, None)
