"""按统一风格组装提示词。"""
from __future__ import annotations

import hashlib
from typing import Optional

from ..assets.catalog import Catalog, ObjectEntry
from ..scene.models import SceneManifest, Viewpoint

DIRECTION_WORDS = [(-22.5, 22.5, "正前方（画面中央）"), (22.5, 67.5, "右前方"), (67.5, 112.5, "右边（画面右侧四分之一处）"),
                   (112.5, 157.5, "右后方"), (-67.5, -22.5, "左前方"), (-112.5, -67.5, "左边（画面左侧四分之一处）"),
                   (-157.5, -112.5, "左后方")]


def direction_word(yaw: float) -> str:
    for lo, hi, w in DIRECTION_WORDS:
        if lo <= yaw < hi:
            return w
    return "身后（画面两端）"


def style_key(catalog: Catalog) -> str:
    s = catalog.style["style"]
    key = catalog.style.get("preset", "") + s["base_zh"] + s["negative"] + catalog.style["object"]["prefix_zh"]
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def panorama_prompt(catalog: Catalog, m: SceneManifest, vp: Viewpoint, restyle: bool = False) -> tuple[str, str]:
    st = catalog.style
    region = st["regions"].get(m.region_style, st["regions"]["其他"])
    era = st["eras"].get(m.era, "")
    parts = [st["panorama"]["prefix_zh"], st["style"]["base_zh"] + "。",
             f"地点：{vp.name_zh}。{region['visual_zh']}。", f"年代特征：{era}。" if era else ""]
    if vp.description_zh:
        parts.append(f"老人描述的样子（优先遵循）：{vp.description_zh}。")
    lms = [l for l in m.landmarks_of(vp.id) if not l.builtin]
    if lms:
        parts.append("方位：" + "；".join(f"{direction_word(l.yaw_deg)}是{l.name_zh}" for l in lms[:6]) + "。")
    # 天气/季节只有在“重绘”时才画进图里；轻量变化由前端后期处理
    if restyle or m.season == "冬" or m.weather == "雪":
        parts.append(f"{st['time_of_day_zh'][m.time_of_day]}，{st['weather_zh'][m.weather]}，{st['season_zh'][m.season]}。")
    else:
        parts.append(f"{st['time_of_day_zh']['上午']}，{st['season_zh'][m.season]}。")
    parts.append(st["panorama"]["suffix_zh"])
    parts.append(st["style"]["people_policy_zh"] + "。")
    negative = st["style"]["negative"] + ", " + st["panorama"]["negative_extra"]
    return "".join(p for p in parts if p), negative


def object_prompt(catalog: Catalog, object_type: str, entry: Optional[ObjectEntry], region: str, era: str) -> tuple[str, str]:
    st = catalog.style
    desc = entry.prompt if entry and entry.prompt else object_type
    parts = [st["object"]["prefix_zh"], f"{desc}，", f"{st['eras'].get(era, '')}，" if (era and entry and entry.person) else f"{era}的样式，" if era else "",
             st["style"]["base_zh"] + "，", st["object"]["suffix_zh"]]
    if entry and entry.person:
        parts.append("，" + st["object"]["person_suffix_zh"] + "，" + st["style"]["people_policy_zh"])
    negative = st["style"]["negative"] + ", " + st["object"]["negative_extra"]
    return "".join(parts), negative
