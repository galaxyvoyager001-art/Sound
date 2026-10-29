"""演示模式（没有 API Key 或 DEMO_MODE=true）下的规则版“LLM”。

只用于本地演示和自动化测试：用关键词规则模拟话语理解、指令生成、陪聊回复和故事归档，
保证没有云端服务时整个流程也能跑通。真实会谈请使用百炼的大语言模型。
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Optional

from ..assets.catalog import get_catalog
from ..scene.commands import RawCommand

REGION_WORDS = [("上海", "上海弄堂"), ("弄堂", "上海弄堂"), ("石库门", "上海弄堂"), ("北京", "北京胡同"), ("胡同", "北京胡同"),
                ("四合院", "北京胡同"), ("苏州", "江南水乡"), ("杭州", "江南水乡"), ("绍兴", "江南水乡"), ("水乡", "江南水乡"),
                ("河边", "江南水乡"), ("村", "北方村庄"), ("农村", "北方村庄"), ("乡下", "北方村庄"), ("厂", "厂区大院"),
                ("大院", "厂区大院")]
ERA_WORDS = [("五十年代", "1950年代"), ("五几年", "1950年代"), ("六十年代", "1960年代"), ("六几年", "1960年代"),
             ("七十年代", "1970年代"), ("七几年", "1970年代"), ("八十年代", "1980年代"), ("八几年", "1980年代")]
TIME_WORDS = [("清早", "清晨"), ("早上", "清晨"), ("清晨", "清晨"), ("中午", "中午"), ("下午", "下午"), ("傍晚", "傍晚"),
              ("黄昏", "傍晚"), ("天黑", "夜晚"), ("晚上", "夜晚"), ("夜里", "夜晚"), ("上午", "上午")]
WEATHER_WORDS = [("下大雨", "大雨"), ("大雨", "大雨"), ("下雨", "小雨"), ("小雨", "小雨"), ("下雪", "雪"), ("雪", "雪"),
                 ("阴天", "阴"), ("雾", "雾"), ("晴", "晴"), ("太阳", "晴")]
SEASON_WORDS = [("春天", "春"), ("夏天", "夏"), ("秋天", "秋"), ("冬天", "冬"), ("过年", "冬")]
DIRS = [("左边", "left"), ("左手", "left"), ("右边", "right"), ("右手", "right"), ("后面", "back"), ("身后", "back"),
        ("背后", "back"), ("前面", "front"), ("对面", "front")]
OPEN_QUESTIONS = [
    "早上醒来，最先听到的是什么声音？",
    "那时候家门口，经常有谁经过？",
    "家里是在哪儿听广播、听歌的？",
    "夏天的晚上，大家都在哪儿乘凉？",
    "门口那条路，是什么样子的？",
    "那时候您最喜欢去哪儿玩？",
]


def _payload(messages) -> Any:
    try:
        return json.loads(messages[-1]["content"])
    except Exception:
        return messages[-1]["content"]


def _find(words, text) -> Optional[str]:
    for w, v in words:
        if w in text:
            return v
    return None


def _match_object(scene: dict, text: str) -> Optional[dict]:
    cat = get_catalog()
    for o in scene.get("objects", []):
        names = [o["type"], o.get("label") or ""]
        entry = cat.find_object(o["type"])
        if entry:
            names += entry.synonyms
        if any(n and n in text for n in names):
            return o
    return None


def _match_landmark(scene: dict, text: str, vp: Optional[str] = None) -> Optional[dict]:
    best = None
    for l in scene.get("landmarks", []):
        if vp and l["viewpoint"] != vp:
            continue
        if l["name"] in text and (best is None or len(l["name"]) > len(best["name"])):
            best = l
    return best


def _raw(**kw) -> dict:
    base = {k: None for k in RawCommand.model_fields}
    base["confidence"] = 0.85
    base.update(kw)
    return base


def _place_phrase(text: str) -> Optional[str]:
    for m in re.finditer(r"(?:在|去|到)([一-龥]{1,4}?)(?:那头|那边|那儿|边上|旁边|门口|看看|$|，|。)", text):
        name = m.group(1)
        if name not in ("这儿", "那儿", "这里", "那里", "哪儿", "这", "那"):
            return name + ("边" if name in ("井",) else "")
    return None


def understand(p: dict) -> dict:
    text = p["老人刚说的话"]
    cat = get_catalog()
    scene = p.get("场景清单") or {}
    labels, control, conf, nav = [], "none", "none", None
    cp = cat.control_phrases
    short = len(re.sub(r"[^一-龥]", "", text)) <= 8
    if any(x in text for x in cp.get("quiet", [])) and (short or "太吵" in text or "安静" in text):
        labels.append("control"); control = "quiet"
    elif any(x in text for x in cp.get("end", [])):
        labels.append("control"); control = "end"
    elif any(x in text for x in cp.get("resume", [])):
        labels.append("control"); control = "resume"
    if p.get("等待确认的问题"):
        if any(text.strip("。！ ").startswith(x) for x in cp.get("no", [])):
            conf = "no"
        elif any(text.strip("。！ ").startswith(x) for x in cp.get("yes", [])):
            conf = "yes"
    elif len(text) <= 10 and any(x in text for x in cp.get("undo", [])) and p.get("最近一次画面修改"):
        labels.append("control"); control = "undo"
    if re.search(r"(看看|瞧瞧|去.{1,4}看|回到|回家门口)", text):
        labels.append("navigate")
        d = _find(DIRS, text)
        lm = _match_landmark(scene, text)
        place = _place_phrase(text)
        if "回" in text:
            nav = {"kind": "go_back", "direction": None, "place_name": place, "landmark_id": None, "viewpoint_id": None}
        elif "去" in text or "到" in text:
            nav = {"kind": "go_place", "direction": None, "place_name": (lm or {}).get("name") or place,
                   "landmark_id": (lm or {}).get("id"), "viewpoint_id": None}
        elif lm:
            nav = {"kind": "look_place", "direction": None, "place_name": lm["name"], "landmark_id": lm["id"], "viewpoint_id": None}
        else:
            nav = {"kind": "look_direction", "direction": d or "left", "place_name": None, "landmark_id": None, "viewpoint_id": None}
    edit_pat = r"(不在这|在.{1,6}那头|没有|没养|不要|去掉|还有|有个|有一个|应该|挪|换个|不是这个|大声|小声|下雨|下雪|晚上|傍晚|冬天|夏天|阴天|太响|太轻)"
    if re.search(edit_pat, text) and control == "none" and "navigate" not in labels:
        labels.append("edit")
    if len(text) >= 14 and re.search(r"(那时候|小时候|我们叫|以前|我妈|我爸|每天|总是|常常|记得|有一回|那年)", text):
        labels.append("story")
    if not labels:
        labels.append("other")
    emotion = "tired" if re.search(r"(累|困|乏)", text) else "sad" if re.search(r"(难过|伤心|想哭|走了)", text) else \
        "happy" if re.search(r"(开心|高兴|好玩|笑)", text) else "neutral"
    return {"labels": labels, "confirmation": conf, "control_kind": control, "emotion": emotion, "navigate": nav}


def commands(p: dict, narration: bool) -> dict:
    text = p.get("老人的话", "")
    scene = p.get("场景清单") or {}
    vp = p.get("当前点位")
    cat = get_catalog()
    out: list[dict] = []
    meta = None
    if narration:
        meta = {"region_style": _find(REGION_WORDS, text), "era": _find(ERA_WORDS, text), "time_of_day": None,
                "weather": None, "season": _find(SEASON_WORDS, text)}
    # 天气 / 时段 / 季节
    if not narration:
        w = _find(WEATHER_WORDS, text)
        if w:
            out.append(_raw(op="set_weather", weather=w))
        t = _find(TIME_WORDS, text)
        if t and re.search(r"(天黑|晚上|傍晚|早上|清晨|中午|下午)", text):
            out.append(_raw(op="set_time", time_of_day=t))
        s = _find(SEASON_WORDS, text)
        if s:
            out.append(_raw(op="set_season", season=s))
    obj = _match_object(scene, text)
    place = _place_phrase(text)
    lm = _match_landmark(scene, text)
    neg = re.search(r"(没有|没养|不要|去掉|没)", text)
    if obj and neg:
        out.append(_raw(op="remove_object", target_id=obj["id"]))
    elif obj and re.search(r"(不在这|在.{1,6}那头|应该在|挪)", text):
        rel = "farther" if "那头" in text or "远" in text else "nearer" if "近" in text else "at"
        out.append(_raw(op="move_object", target_id=obj["id"], landmark_id=(lm or {}).get("id"),
                        landmark_name=None if lm else place, direction="front" if not lm else None,
                        distance="far" if rel == "farther" else None, relation=rel,
                        confidence=0.9 if (lm or place) else 0.5,
                        confirm_question=f"是说{obj['label'] or obj['type']}应该在{(lm or {}).get('name') or place or '别处'}吗？"))
    elif obj and re.search(r"(大声|太轻|听不见)", text):
        out.append(_raw(op="set_gain", target_id=obj["id"], gain_change="louder"))
    elif obj and re.search(r"(小声|太响|太吵)", text):
        out.append(_raw(op="set_gain", target_id=obj["id"], gain_change="quieter"))
    elif obj and re.search(r"(不是这个|换个)", text):
        out.append(_raw(op="replace_sound", target_id=obj["id"], next_variant=True))
    elif not neg:
        existing = {o["type"] for o in scene.get("objects", [])}
        for entry in cat.objects:
            names = [entry.type] + entry.synonyms
            hit = next((n for n in names if len(n) >= 2 and n in text), None)
            if hit and entry.type not in existing:
                d = _find(DIRS, text)
                out.append(_raw(op="add_object", object_type=entry.type, label_zh=hit if hit != entry.type else None,
                                landmark_id=(lm or {}).get("id"), landmark_name=None if lm else place,
                                direction=d or ("front" if place else None),
                                distance="far" if ("那头" in text or "远" in text) else "mid" if place else None,
                                relation="at", sound_query=entry.sound))
                existing.add(entry.type)
                if len(out) >= 3:
                    break
    if narration and re.search(r"(后院|井边|河边|弄堂口|胡同口|村口|厂门口)", text):
        name = re.search(r"(后院|井边|河边|弄堂口|胡同口|村口|厂门口)", text).group(1)
        if name not in [v["name"] for v in scene.get("viewpoints", [])] and len(scene.get("viewpoints", [])) >= 1:
            out.append(_raw(op="add_viewpoint", landmark_name=name, description=text[:60], confidence=0.75))
    desc = text if narration and re.search(r"(房子|墙|门|路|树|弄堂|胡同|院子|河|桥)", text) else None
    return {"meta": meta, "viewpoint_description": desc, "commands": out}


def story(p: dict) -> dict:
    text = p["老人讲的话"]
    scene = p.get("场景清单") or {}
    obj = _match_object(scene, text)
    vp = p.get("当前点位") or (scene.get("viewpoints") or [{}])[0].get("id")
    m = re.search(r"叫他([一-龥]{2,4})|([一-龥]{1,2}[伯叔婶姨爷奶])", text)
    title = (m.group(1) or m.group(2)) if m else (obj["type"] if obj else text[:6])
    if obj:
        return {"is_story": True, "title_zh": f"{title}的故事"[:10], "anchor_kind": "object", "anchor_id": obj["id"]}
    lm = _match_landmark(scene, text, vp)
    if lm:
        return {"is_story": True, "title_zh": title[:10], "anchor_kind": "landmark", "anchor_id": lm["id"]}
    return {"is_story": bool(vp), "title_zh": title[:10], "anchor_kind": "viewpoint", "anchor_id": vp or ""}


def companion(p: dict) -> dict:
    task = p.get("当前任务", "")
    if p.get("需要向老人确认的问题"):
        return {"speak": True, "text": p["需要向老人确认的问题"]}
    asked = p.get("已经问过的问题", [])
    q = next((x for x in OPEN_QUESTIONS if x not in asked), random.choice(OPEN_QUESTIONS))
    actions = p.get("刚才对画面/声音做的改动") or []
    if "要结束了" in task:
        return {"speak": True, "text": "今天听您讲了这么多，真好。我们下次再接着聊，您慢慢休息。"}
    if "会谈刚开始" in task:
        return {"speak": True, "text": "您好呀，今天想听您讲讲小时候住的地方。您小时候住在哪儿呀？"}
    if "刚刚准备好" in task:
        return {"speak": True, "text": "您看，这是照着您说的样子画的。您看看、听听，哪里不像，就跟我说。"}
    if "安静了一会儿" in task:
        return {"speak": random.random() < 0.5, "text": "您听，这是什么声音？"}
    if actions:
        return {"speak": True, "text": "好，您看，改好了。" + ("" if random.random() < 0.6 else q)}
    if p.get("老人当前的情绪") in ("sad", "anxious"):
        return {"speak": True, "text": "那时候真不容易。我们慢慢聊，不着急。"}
    dialogue = p.get("最近的对话（旧→新）") or []
    last = dialogue[-1]["text"] if dialogue else ""
    if len(last) > 25:
        return {"speak": True, "text": random.choice(["嗯，真好。后来呢？", "听着就很热闹。", "嗯，您接着说。"])}
    return {"speak": True, "text": q}


def keywords(query: str) -> dict:
    cat = get_catalog()
    for v in cat.sounds:
        if v.key in query or any(z in query for z in v.zh):
            return {"en": v.en, "zh": v.zh, "mode": v.mode, "is_sensitive": False}
    return {"en": [], "zh": [query], "mode": "event", "is_sensitive": cat.is_sensitive_text(query)}


def details(text: str) -> dict:
    def items(pattern):
        seen, out = set(), []
        for m in re.finditer(pattern, text):
            s = m.group(0)
            if s not in seen:
                seen.add(s)
                out.append({"text": s, "quote": text[max(0, m.start() - 6): m.end() + 6]})
        return out
    sensory = []
    for kind, pat in (("sound", r"(响|声|叫|敲|吆喝|唱)"), ("smell", r"(香|味|臭)"), ("color", r"(红|绿|蓝|白|黑|黄|灰)"),
                      ("temperature", r"(热|冷|凉|暖|烫)")):
        for it in items(pat):
            sensory.append({"kind": kind, **it})
    return {
        "persons": items(r"(阿[一-龥]{1,2}伯|[一-龥]师傅|[王李张刘陈杨赵黄周吴]阿姨|我妈|我爸|奶奶|爷爷|外婆|外公)"),
        "places": items(r"(巷口|弄堂口|井边|河边|胡同口|村口|厂里|学校|门口|后院|灶间|老虎灶)"),
        "times": items(r"([五六七八]十年代|[五六七八]几年|过年|夏天|冬天|早上|傍晚|晚上|小时候|\d+岁|[一二三四五六七八九十]+岁)"),
        "sensory": sensory,
        "events": items(r"[一-龥]{0,4}(跑|买|吃|玩|跳|躲|洗|打水|乘凉)[一-龥]{0,4}"),
    }


def demo_llm_handler(messages, schema) -> Any:
    name = schema.__name__ if schema else "chat"
    p = _payload(messages)
    sys_text = messages[0]["content"] if messages else ""
    if name == "UtteranceAnalysis":
        return understand(p)
    if name == "SceneUpdate":
        return commands(p, narration="逐步画出场景" in sys_text)
    if name == "StoryArchive":
        return story(p)
    if name == "CompanionTurn":
        return companion(p)
    if name == "SoundKeywords":
        return keywords(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))
    if name == "DetailExtraction":
        return details(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))
    return "嗯。"
