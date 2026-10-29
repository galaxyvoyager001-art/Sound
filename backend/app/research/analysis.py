"""会后自动分析：讲述时长、段数、主动开口次数、细节计数、修改与导航、故事。

定义（写进报告，便于论文方法部分引用）：
- 讲述总时长：老人所有识别句子的（结束 − 开始）之和（秒）。陪聊者的声音来自系统，天然不计入。
- 讲述段数：老人的“轮次”数（两次回应之间老人说的话合为一段，以沉默 ≥ 回应阈值切分）。
- 主动开口：老人的一段讲述，满足以下任一条件即计为主动开口——
  (a) 它前面最近的一句陪聊不是问句；(b) 距离最近一句陪聊结束已超过 10 秒；(c) 会谈中第一次开口前陪聊者没有提问。
- 细节：由 LLM 从老人的全部讲述中抽取人名、地名、时间、感官描述（声音/气味/颜色/温度/味道/触感）、事件，
  逐条保留原话片段，存为 details.json / details.csv，供人工核对后再用于统计。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from collections import Counter
from pathlib import Path

from ..conversation import demo_brain, prompts
from ..conversation.schemas import DetailExtraction

log = logging.getLogger("analysis")


def _turns(utts: list[dict]) -> list[dict]:
    """按 turn_id（缺失时按时间间隔）把老人的句子合并成段。"""
    turns: list[dict] = []
    for u in utts:
        if u["speaker"] != "elder":
            continue
        tid = u.get("turn_id")
        if turns and ((tid and turns[-1]["turn_id"] == tid) or (not tid and u["start_s"] - turns[-1]["end_s"] < 4.0)):
            turns[-1]["end_s"] = max(turns[-1]["end_s"], u["end_s"])
            turns[-1]["text"] += u["text"]
            turns[-1]["speech_s"] += max(0.0, u["end_s"] - u["start_s"])
        else:
            turns.append({"turn_id": tid, "start_s": u["start_s"], "end_s": u["end_s"], "text": u["text"],
                          "speech_s": max(0.0, u["end_s"] - u["start_s"])})
    return turns


def initiated_flags(turns: list[dict], companion: list[dict]) -> list[bool]:
    flags = []
    for t in turns:
        prev = [c for c in companion if c["end_s"] <= t["start_s"] + 0.5]
        if not prev:
            flags.append(True)
            continue
        last = prev[-1]
        is_q = any(q in last["text"] for q in ("？", "?", "吗", "呢"))
        gap = t["start_s"] - last["end_s"]
        flags.append((not is_q) or gap > 10.0)
    return flags


async def extract_details(ctx, transcript: str) -> DetailExtraction:
    if not transcript.strip():
        return DetailExtraction(persons=[], places=[], times=[], sensory=[], events=[])
    try:
        return await ctx.providers.llm.structured(prompts.details_messages(transcript), DetailExtraction,
                                                  model=ctx.providers.llm_model, temperature=0.0)
    except Exception as e:
        log.warning("细节抽取失败，使用规则版：%s", e)
        return DetailExtraction.model_validate(demo_brain.details(transcript))


async def analyze_session(ctx, session_id: str) -> dict:
    db = ctx.db
    s = db.one("SELECT * FROM sessions WHERE id=?", (session_id,))
    if not s:
        return {}
    utts = db.query("SELECT * FROM utterances WHERE session_id=? ORDER BY start_s", (session_id,))
    events = db.events(session_id)
    companion = [u for u in utts if u["speaker"] == "companion"]
    turns = _turns(utts)
    flags = initiated_flags(turns, companion)
    transcript = "\n".join(t["text"] for t in turns)
    details = await extract_details(ctx, transcript)
    sens = Counter(x.kind for x in details.sensory)

    cmds = [e["payload"] for e in events if e["type"] == "command"]
    executed = [c for c in cmds if c.get("executed")]
    op_counts = Counter(c["op"] for c in executed)
    by_source = Counter(c.get("added_by", "edit") for c in executed)
    confirms = Counter(e["payload"].get("answer") for e in events if e["type"] == "confirm")
    vp_switches = sum(1 for e in events if e["type"] == "viewpoint_changed")
    stories = [e["payload"] for e in events if e["type"] == "story"]
    start = next((e["ts"] for e in events if e["type"] == "session_start"), None)
    end = next((e["ts"] for e in events if e["type"] == "session_end"), None)

    metrics = {
        "session_id": session_id, "participant_id": s["participant_id"], "condition": s["condition"],
        "region_style": s["region_style"], "end_reason": s["end_reason"],
        "session_minutes": round((end - start) / 60, 2) if start and end else None,
        "elder_speech_s": round(sum(t["speech_s"] for t in turns), 1),
        "elder_turns": len(turns),
        "elder_initiated_turns": sum(flags),
        "elder_chars": sum(len(t["text"]) for t in turns),
        "mean_turn_s": round(sum(t["speech_s"] for t in turns) / len(turns), 1) if turns else 0,
        "companion_utterances": len(companion),
        "details_persons": len(details.persons), "details_places": len(details.places),
        "details_times": len(details.times), "details_sensory": len(details.sensory),
        "details_sensory_sound": sens.get("sound", 0), "details_sensory_smell": sens.get("smell", 0),
        "details_sensory_color": sens.get("color", 0), "details_sensory_temperature": sens.get("temperature", 0),
        "details_events": len(details.events),
        "details_total": len(details.persons) + len(details.places) + len(details.times) + len(details.sensory) + len(details.events),
        "edits_executed": sum(1 for c in executed if c.get("added_by", "edit") == "edit"),
        "narration_additions": by_source.get("narration", 0),
        "edits_by_op": dict(op_counts),
        "edits_rejected": sum(1 for c in cmds if not c.get("executed")),
        "confirm_asked": sum(1 for e in events if e["type"] == "confirm_asked"),
        "confirm_yes": confirms.get("yes", 0), "confirm_no": confirms.get("no", 0),
        "undo": sum(1 for e in events if e["type"] == "undo"),
        "viewpoint_switches": vp_switches,
        "stories": len(stories),
        "quiet_requests": sum(1 for e in events if e["type"] == "quiet"),
        "barge_ins": sum(1 for e in events if e["type"] == "barge_in"),
        "llm_errors": sum(1 for e in events if e["type"] == "llm_error"),
    }
    out = ctx.settings.data_dir / "reports" / session_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), "utf-8")
    (out / "details.json").write_text(details.model_dump_json(indent=1), "utf-8")
    (out / "turns.json").write_text(json.dumps([{**t, "initiated": f} for t, f in zip(turns, flags)], ensure_ascii=False,
                                               indent=1), "utf-8")
    write_metrics_csv(out / "metrics.csv", [metrics])
    with open(out / "details.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["类别", "子类", "内容", "原话", "人工核对（填 1/0）"])
        for cat, items in (("人名", details.persons), ("地名", details.places), ("时间", details.times),
                           ("事件", details.events)):
            for it in items:
                w.writerow([cat, "", it.text, it.quote, ""])
        for it in details.sensory:
            w.writerow(["感官", it.kind, it.text, it.quote, ""])
    from .report import session_html
    (out / "report.html").write_text(session_html(metrics, details, turns, flags, stories), "utf-8")
    db.log_event(session_id, "analysis_done", {"metrics": metrics})
    return metrics


FLAT_KEYS = ["session_id", "participant_id", "condition", "region_style", "end_reason", "session_minutes",
             "elder_speech_s", "elder_turns", "elder_initiated_turns", "elder_chars", "mean_turn_s", "companion_utterances",
             "details_total", "details_persons", "details_places", "details_times", "details_sensory",
             "details_sensory_sound", "details_sensory_smell", "details_sensory_color", "details_sensory_temperature",
             "details_events", "edits_executed", "narration_additions", "edits_rejected", "confirm_asked", "confirm_yes",
             "confirm_no", "undo", "viewpoint_switches", "stories", "quiet_requests", "barge_ins", "llm_errors", "edits_by_op"]


def write_metrics_csv(path: Path, rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(FLAT_KEYS)
    for r in rows:
        w.writerow([json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), dict) else r.get(k) for k in FLAT_KEYS])
    text = buf.getvalue()
    path.write_text("﻿" + text, "utf-8")
    return text
