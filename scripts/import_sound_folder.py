"""把以前下载过的一批声音素材导入本项目（例如 WSL 里的 ~/amy-home-hearingaid）。

三步：
  1) 看看还缺哪些声音：
       python scripts/import_sound_folder.py needs
  2) 扫描文件夹，生成候选表（不会复制任何文件）：
       python scripts/import_sound_folder.py scan ~/amy-home-hearingaid
     结果写到 data/import_candidates.csv（用 Excel/WPS 打开）。每一行是一个音频文件，系统已经猜好：
       key      —— 对应声音词表里的哪个声音（如 木梆子、自行车铃），猜不出来是空的，可以自己填
       license  —— 授权。能自动识别的有：Freesound 文件名（12345__作者__标题.wav）+ FREESOUND_API_KEY 查询，
                   或 Freesound 打包下载自带的 _readme_and_license.txt
       keep     —— 1 = 导入，0 = 不导入
     授权不明确的文件 keep 默认为 0：请回想当初是从哪儿下载的，把 license（如 CC0 / CC BY 4.0）
     和 source_url、author 填上，再把 keep 改成 1。不确定的就别导入。
  3) 导入勾选的文件（转码、响度归一化、登记来源和授权）：
       python scripts/import_sound_folder.py apply data/import_candidates.csv

导入的素材状态为“待审核”，请在设置页“素材审核”里试听。
"""
import argparse
import asyncio
import csv
import hashlib
import re
import sys
import tempfile
from pathlib import Path

from _common import ROOT, registry

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif"}
FREESOUND_NAME = re.compile(r"^(\d+)__([^_]+(?:_[^_]+)*?)__(.+)$")
COLUMNS = ["keep", "path", "key", "match_score", "duration_s", "site", "source_url", "author", "license",
           "attribution", "original_title", "notes"]


def setup():
    sys.path.insert(0, str(ROOT / "backend"))
    from app.assets.catalog import get_catalog
    from app.config import get_settings
    return get_settings(), get_catalog()


def parse_freesound_readme(folder: Path) -> dict[str, dict]:
    """Freesound 打包下载里的 _readme_and_license.txt：文件名 → {url, license}。"""
    info: dict[str, dict] = {}
    for readme in folder.rglob("_readme_and_license.txt"):
        cur = None
        for line in readme.read_text("utf-8", errors="ignore").splitlines():
            m = re.match(r"\s*\*\s*(\S.*\.(?:wav|mp3|flac|ogg|aif+|m4a))\s*$", line, re.I)
            if m:
                cur = m.group(1).strip()
                info[cur] = {}
                continue
            if cur:
                m = re.match(r"\s*\*\s*(url|license)\s*:\s*(.+)$", line, re.I)
                if m:
                    info[cur][m.group(1).lower()] = m.group(2).strip()
    return info


def guess_key(path: Path, vocab, catalog) -> tuple[str, float]:
    """按文件名和所在文件夹名猜对应的声音词条。"""
    text = " ".join(list(path.parts[-4:])).lower().replace("_", " ").replace("-", " ")
    best, best_score = "", 0.0
    for v in vocab:
        score = 0.0
        for w in [v.key] + v.zh:
            if w and w in text:
                score = max(score, 1.0)
        for phrase in v.en:
            words = [x for x in phrase.lower().split() if len(x) > 2]
            if words:
                hit = sum(1 for x in words if x in text) / len(words)
                score = max(score, hit * 0.9)
        if score > best_score:
            best, best_score = v.key, score
    # 物件同义词（如“脚踏车”）→ 物件的默认声音
    for o in catalog.objects:
        if o.sound and any(n and n in text for n in [o.type] + o.synonyms):
            k = o.sound.split()[0]
            if best_score < 0.8 and any(v.key == k for v in vocab):
                best, best_score = k, 0.8
    return (best, round(best_score, 2)) if best_score >= 0.5 else ("", round(best_score, 2))


async def freesound_lookup(ids: list[str], key: str) -> dict[str, dict]:
    import httpx
    out = {}
    async with httpx.AsyncClient(timeout=20) as c:
        for i in ids:
            try:
                r = await c.get(f"https://freesound.org/apiv2/sounds/{i}/",
                                params={"token": key, "fields": "id,name,username,license,url"})
                if r.status_code == 200:
                    out[i] = r.json()
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return out


def cmd_needs():
    _, cat = setup()
    reg = registry()
    print("声音词表（config/catalog.yaml）里各声音的素材情况：\n")
    print(f"{'声音':<10}{'真实录音':>6}{'合成占位':>8}   检索关键词")
    for v in cat.sounds:
        recs = [a for a in reg.all() if a.kind in ("sound", "music") and a.tags and a.tags[0] == v.key and a.usable]
        real = sum(1 for a in recs if a.source.type != "procedural")
        synth = len(recs) - real
        mark = "  ← 缺" if real == 0 else ""
        print(f"{v.key:<10}{real:>6}{synth:>8}   {' / '.join(v.zh[:3] + v.en[:2])}{mark}")
    print("\n每个声音建议有 2–3 段不同的真实录音（事件类声音会随机轮换，避免机械重复）。")


def cmd_scan(folder: Path, out: Path):
    s, cat = setup()
    from app.assets.online import ffprobe_duration, normalize_license
    files = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT and p.is_file()]
    print(f"找到 {len(files)} 个音频文件")
    readme = parse_freesound_readme(folder)
    fs_ids = {}
    for p in files:
        m = FREESOUND_NAME.match(p.stem)
        if m:
            fs_ids[str(p)] = m.group(1)
    fs_info = {}
    if fs_ids and s.freesound_api_key:
        print(f"用 Freesound API 查询 {len(fs_ids)} 个文件的授权…")
        fs_info = asyncio.run(freesound_lookup(sorted(set(fs_ids.values())), s.freesound_api_key))
    elif fs_ids:
        print(f"有 {len(fs_ids)} 个文件像是从 Freesound 下载的；配置 FREESOUND_API_KEY 后重新扫描可以自动查到授权。")
    rows = []
    for p in files:
        key, score = guess_key(p, cat.sounds, cat)
        row = dict.fromkeys(COLUMNS, "")
        row.update(path=str(p), key=key, match_score=score, duration_s=round(ffprobe_duration(p), 1))
        rd = readme.get(p.name)
        fid = fs_ids.get(str(p))
        if fid and fid in fs_info:
            x = fs_info[fid]
            lic = x["license"]
            row.update(site="Freesound", source_url=x["url"], author=x["username"], original_title=x["name"],
                       license={"Creative Commons 0": "CC0"}.get(lic, lic))
        elif rd:
            row.update(site="Freesound", source_url=rd.get("url", ""), license=rd.get("license", ""))
            m = FREESOUND_NAME.match(p.stem)
            if m:
                row.update(author=m.group(2), original_title=m.group(3))
        elif fid:
            m = FREESOUND_NAME.match(p.stem)
            row.update(site="Freesound", source_url=f"https://freesound.org/s/{fid}/", author=m.group(2),
                       original_title=m.group(3), notes="授权待查：配置 FREESOUND_API_KEY 后重新扫描")
        lic_ok = bool(row["license"]) and normalize_license(_lic_text(row["license"])) is not None
        if row["license"] and not lic_ok:
            row["notes"] = (row["notes"] + "；" if row["notes"] else "") + "授权不允许使用（如 NC 非商业）或无法识别"
        dur = float(row["duration_s"] or 0)
        row["keep"] = 1 if (key and lic_ok and 0.3 <= dur <= 600 and not cat.is_sensitive_text(str(p))) else 0
        if cat.is_sensitive_text(str(p)):
            row["notes"] = (row["notes"] + "；" if row["notes"] else "") + "可能是敏感内容，不导入"
        rows.append(row)
    rows.sort(key=lambda r: (-int(r["keep"]), r["key"] == "", r["key"], r["path"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_keep = sum(int(r["keep"]) for r in rows)
    n_key = sum(1 for r in rows if r["key"])
    print(f"已写入 {out}：{n_key} 个文件猜到了对应的声音，其中 {n_keep} 个授权明确、默认勾选导入。")
    print("请打开表格检查/补充 key 和 license，然后运行：python scripts/import_sound_folder.py apply", out)


def _lic_text(lic: str) -> str:
    lic = lic.strip()
    if "creativecommons.org/publicdomain/zero" in lic:
        return "CC0"
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/(\d\.\d)", lic)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    return {"Attribution": "CC BY 4.0", "Attribution NonCommercial": "CC BY-NC 4.0"}.get(lic, lic)


def cmd_apply(table: Path):
    s, cat = setup()
    from app.assets.online import ffprobe_duration, normalize_license, transcode, vocab_props
    from app.assets.registry import AssetRecord, AssetSource
    from app.config import to_rel
    from app.generation.service import slug
    reg = registry()
    lib = s.assets_dir / "sounds" / "library"
    lib.mkdir(parents=True, exist_ok=True)
    ok = skipped = 0
    with open(table, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if str(r.get("keep", "")).strip() != "1":
            continue
        src = Path(r["path"])
        key = r["key"].strip()
        vocab = cat.sound_vocab(key)
        lic = normalize_license(_lic_text(r["license"]))
        if not src.exists() or not vocab or not lic:
            print(f"跳过 {src.name}：" + ("文件不存在" if not src.exists() else "key 不在声音词表里" if not vocab
                                         else "授权缺失或不允许使用"))
            skipped += 1
            continue
        if cat.is_sensitive_text(f"{src} {r.get('original_title', '')}"):
            print(f"跳过 {src.name}：可能是敏感内容")
            skipped += 1
            continue
        digest = hashlib.sha1(str(src).encode()).hexdigest()[:8]
        aid = f"snd_{slug(key)}_imp_{digest}"
        dst = lib / f"{slug(key)}_imp_{digest}.mp3"
        loop_like = vocab.mode == "loop" or vocab.ambience
        if not transcode(src, dst, not (vocab.ambience or vocab.music), -26.0 if vocab.ambience else -20.0,
                         60.0 if loop_like else 20.0):
            skipped += 1
            continue
        author = r.get("author") or "未署名"
        title = r.get("original_title") or src.stem
        site = r.get("site") or "本地收藏"
        attribution = r.get("attribution") or (f"“{title}” by {author}, {lic}, via {site}" if lic.startswith("CC-BY") else None)
        rec = AssetRecord(
            id=aid, kind="music" if vocab.music else "sound", name_zh=key, tags=[key] + vocab.zh, synonyms=vocab.en,
            files=[to_rel(dst)],
            source=AssetSource(type="web" if r.get("source_url") else "own_recording", url=r.get("source_url") or None,
                               site=site, author=author, original_title=title,
                               notes=f"从本地收藏导入：{src}；{r.get('notes', '')}".strip("；")),
            license=lic, attribution=attribution,
            copyright_status="public_domain" if lic in ("CC0-1.0", "Public-Domain") else "cc",
            review_status="pending", sound=vocab_props(vocab, ffprobe_duration(dst)))
        try:
            reg.register(rec)
            ok += 1
            print("导入", key, "←", src.name)
        except ValueError as e:
            print(f"跳过 {src.name}：{e}")
            skipped += 1
    print(f"完成：导入 {ok} 个，跳过 {skipped} 个。请到设置页“素材审核”里试听，满意的标为“通过”。")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("needs", help="列出各声音现有素材，看还缺哪些")
    p = sub.add_parser("scan", help="扫描文件夹，生成候选表")
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "import_candidates.csv")
    p = sub.add_parser("apply", help="导入候选表里 keep=1 的文件")
    p.add_argument("table", type=Path)
    a = ap.parse_args()
    if a.cmd == "needs":
        cmd_needs()
    elif a.cmd == "scan":
        cmd_scan(a.folder.expanduser(), a.out)
    else:
        cmd_apply(a.table)


if __name__ == "__main__":
    main()
