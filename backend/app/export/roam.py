"""导出“老家漫游”：把参与者的最终场景打包成一个可离线打开的 HTML 文件。

手机上用 file:// 打开本地网页时，WebGL 不能读取旁边的图片文件，所以所有素材都以 data URI 内嵌。
只包含设置页勾选的故事；有版权限制（非公有领域 / 非 CC）的音乐和戏曲录音不导出。
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import time
from pathlib import Path
from typing import Optional

from PIL import Image

from ..config import to_abs

MAX_VARIANTS = 2


def _data_uri(path: Path, max_w: Optional[int] = None, quality: int = 80) -> Optional[str]:
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = path.read_bytes()
    if max_w and mime == "image/jpeg":
        img = Image.open(io.BytesIO(data))
        if img.width > max_w:
            img = img.resize((max_w, max_w * img.height // img.width), Image.LANCZOS)
        b = io.BytesIO()
        img.convert("RGB").save(b, "JPEG", quality=quality, optimize=True, progressive=True)
        data = b.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _template(root: Path) -> str:
    p = root / "frontend" / "dist-roam" / "roam.html"
    if not p.exists():
        raise FileNotFoundError("找不到漫游页模板：请先在 frontend/ 目录运行 npm run build:roam")
    return p.read_text("utf-8")


def build_roam(ctx, scene_id: str, participant_id: str, story_ids: Optional[list[str]] = None) -> Path:
    m = ctx.store.get(scene_id)
    if not m:
        raise FileNotFoundError("场景不存在")
    tpl = _template(ctx.settings.root)
    m = m.model_copy(deep=True)
    if story_ids is not None:
        m.stories = [s for s in m.stories if s.id in story_ids]
    else:
        m.stories = [s for s in m.stories if s.include_in_export]
    # 只保留已经画好的点位
    ready = {v.id for v in m.viewpoints if v.panorama_path}
    m.viewpoints = [v for v in m.viewpoints if v.id in ready]
    for v in m.viewpoints:
        v.neighbors = [n for n in v.neighbors if n in ready]
    if m.entry_viewpoint not in ready:
        m.entry_viewpoint = next(iter(ready), None)
    m.landmarks = [l for l in m.landmarks if l.viewpoint_id in ready]
    lm_ids = {l.id for l in m.landmarks}
    m.objects = [o for o in m.objects if o.landmark in lm_ids]
    m.unresolved = []

    table = ctx.scenes.assets_for(m)
    credits = []
    seen_assets = set()
    out_assets: dict = {}
    reg = ctx.registry

    def credit(rec):
        if rec.id in seen_assets:
            return
        seen_assets.add(rec.id)
        src = rec.source
        where = (f"AI 生成（{src.service}）" if src.type == "ai" else src.site or
                 {"procedural": "程序合成", "own_recording": "自有录音", "restoration": "自行修复"}.get(src.type, src.type))
        credits.append({"name": rec.name_zh, "source": where, "license": rec.license, "attribution": rec.attribution or ""})

    def exportable(rec) -> bool:
        if rec.kind == "music" or rec.copyright_status in ("copyrighted", "unknown"):
            return rec.copyright_status in ("public_domain", "cc")
        return not rec.sensitive

    for key, info in table.items():
        uris = []
        if key.startswith("file:"):
            rel = key[5:]
            uri = _data_uri(to_abs(rel), max_w=4096 if rel.endswith(".jpg") else None, quality=78)
            if uri:
                uris.append(uri)
        elif key.startswith("grp:"):
            g = ctx.resolver.group(key)
            if g:
                n = 0
                for rec in sorted(g.assets, key=lambda a: a.review_status != "approved"):
                    if not exportable(rec):
                        continue
                    for f in rec.files:
                        if n >= MAX_VARIANTS:
                            break
                        uri = _data_uri(to_abs(f))
                        if uri:
                            uris.append(uri)
                            n += 1
                            credit(rec)
        else:
            rec = reg.get(key.removeprefix("ir:")) if not key.startswith("ir:") else None
            if key.startswith("ir:"):
                rec = next((a for a in reg.by_kind("ir") if a.tags and f"ir:{a.tags[0]}" == key), None)
            if rec and exportable(rec):
                for f in rec.files[:1]:
                    uri = _data_uri(to_abs(f))
                    if uri:
                        uris.append(uri)
                credit(rec)
        if uris:
            out_assets[key] = {**{k: v for k, v in info.items() if k != "urls"}, "urls": uris}
    # 声音不能导出的物件：去掉声音，画面保留
    for o in m.objects:
        if o.sound and o.sound.clip_id not in out_assets:
            o.sound = None
    m.ambience = [a for a in m.ambience if a.clip_id in out_assets]

    payload = {"manifest": m.model_dump(mode="json"), "assets": out_assets,
               "title": f"{m.region_style} · {m.era}", "credits": credits}
    script = "<script>window.ROAM_DATA=" + json.dumps(payload, ensure_ascii=False).replace("</", "<\\/") + ";</script>"
    html = tpl.replace("<!--ROAM_DATA-->", script, 1)
    out_dir = ctx.settings.data_dir / "exports" / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"laojia_manyou_{participant_id}_{time.strftime('%Y%m%d_%H%M')}.html"
    path.write_text(html, "utf-8")
    return path
