"""在线检索并下载有明确授权的声音素材，自动登记来源与授权。

- Wikimedia Commons（无需 Key）：MediaWiki API，授权信息来自 extmetadata（LicenseShortName / Artist / …）。
- Freesound（需要 FREESOUND_API_KEY）：/apiv2/search/，只取 “Creative Commons 0” 与 “Attribution”，
  下载 HQ 预览（mp3）。非商业（NC）、授权不明确的一律不下载。
- 文生音效（可选 ElevenLabs）：网上找不到时生成，同样登记来源。
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from ..config import to_rel
from .catalog import Catalog, SoundVocab
from .registry import AssetRecord, AssetRegistry, AssetSource, SoundProps

log = logging.getLogger("online")
UA = "TingjianCongqian/0.1 (reminiscence activity research project; contact via repository)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


@dataclass
class Candidate:
    site: str
    title: str
    page_url: str
    download_url: str
    license: str
    license_url: Optional[str]
    author: str
    description: str
    duration_s: Optional[float] = None
    size: Optional[int] = None
    attribution_required: bool = True
    score: float = 0.0
    query: str = ""
    extra: dict = field(default_factory=dict)


def normalize_license(short: str) -> Optional[str]:
    """把各站点的授权写法统一；不允许使用的返回 None。"""
    s = short.strip().lower().replace("_", " ")
    if not s:
        return None
    if "nc" in re.split(r"[-\s]", s) or "noncommercial" in s or "non-commercial" in s or "nd" in re.split(r"[-\s]", s):
        return None
    if s in ("cc0", "cc0 1.0", "creative commons 0") or "cc0" in s or "publicdomain/zero" in s:
        return "CC0-1.0"
    if s.startswith("public domain") or s in ("pd", "pd-self", "pd-old", "pd-usgov") or s.startswith("pd-"):
        return "Public-Domain"
    m = re.match(r"cc[ -]by[ -]sa[ -]?(\d\.\d)?", s)
    if m:
        return f"CC-BY-SA-{m.group(1) or '4.0'}"
    m = re.match(r"cc[ -]by[ -]?(\d\.\d)?", s)
    if m:
        return f"CC-BY-{m.group(1) or '4.0'}"
    if s == "attribution":
        return "CC-BY-4.0"
    return None


def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def relevance(c: Candidate, words: list[str], sensitive: list[str]) -> float:
    text = f"{c.title} {c.description}".lower()
    hits = sum(1 for w in words if w.lower() in text)
    score = hits / max(1, len(words))
    if any(k.lower() in text for k in sensitive):
        score -= 1.0
    # 偏好适中时长
    if c.duration_s:
        if c.duration_s < 0.5 or c.duration_s > 600:
            score -= 1
        elif 2 <= c.duration_s <= 90:
            score += 0.2
    return score


async def polite_get(client: httpx.AsyncClient, url: str, params: dict, attempts: int = 4) -> httpx.Response:
    """公共 API 限流（403/429）时指数退避重试。"""
    delay = 15.0
    for i in range(attempts):
        r = await client.get(url, params=params)
        if r.status_code not in (403, 429, 503) or i == attempts - 1:
            r.raise_for_status()
            return r
        log.info("%s 限流（%d），%.0f 秒后重试", url.split('/')[2], r.status_code, delay)
        await asyncio.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


class CommonsSource:
    site = "Wikimedia Commons"

    async def search(self, client: httpx.AsyncClient, query: str, limit: int = 12) -> list[Candidate]:
        r = await polite_get(client, COMMONS_API, {
            "action": "query", "list": "search", "srnamespace": 6, "srsearch": f"{query} filetype:audio",
            "srlimit": limit, "format": "json", "maxlag": 5})
        titles = [x["title"] for x in r.json().get("query", {}).get("search", [])]
        if not titles:
            return []
        await asyncio.sleep(1.0)
        r = await polite_get(client, COMMONS_API, {
            "action": "query", "titles": "|".join(titles), "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|metadata", "format": "json", "maxlag": 5})
        out = []
        for page in r.json().get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            lic = normalize_license(meta.get("LicenseShortName", {}).get("value", ""))
            if not lic or not info.get("url"):
                continue
            dur = None
            for m in info.get("metadata") or []:
                if m.get("name") in ("length", "playtime_seconds"):
                    try:
                        dur = float(m.get("value"))
                    except (TypeError, ValueError):
                        pass
            author = strip_html(meta.get("Artist", {}).get("value", "")) or "未署名"
            out.append(Candidate(
                site=self.site, title=page["title"].removeprefix("File:"),
                page_url=info.get("descriptionurl", ""), download_url=info["url"], license=lic,
                license_url=meta.get("LicenseUrl", {}).get("value"), author=author[:120],
                description=strip_html(meta.get("ImageDescription", {}).get("value", ""))[:400],
                duration_s=dur, size=info.get("size"),
                attribution_required=meta.get("AttributionRequired", {}).get("value", "true") != "false",
                query=query))
        return out


class OpenverseSource:
    """Openverse（WordPress 基金会的开放授权媒体检索，聚合 Freesound / Wikimedia / Jamendo 等，匿名可用但限流）。"""
    site = "Openverse"

    async def search(self, client: httpx.AsyncClient, query: str, limit: int = 12) -> list[Candidate]:
        r = await polite_get(client, "https://api.openverse.org/v1/audio/", {
            "q": query, "license": "cc0,pdm,by,by-sa", "page_size": limit}, attempts=2)
        out = []
        for x in r.json().get("results", []):
            lic = normalize_license({"cc0": "cc0", "pdm": "pd", "by": f"cc by {x.get('license_version') or '4.0'}",
                                     "by-sa": f"cc by-sa {x.get('license_version') or '4.0'}"}.get(x.get("license", ""), ""))
            if not lic or not x.get("url"):
                continue
            dur = x.get("duration")
            out.append(Candidate(site=f"Openverse/{x.get('source') or x.get('provider')}", title=x.get("title") or "",
                                 page_url=x.get("foreign_landing_url") or "", download_url=x["url"], license=lic,
                                 license_url=x.get("license_url"), author=(x.get("creator") or "未署名")[:120],
                                 description=" ".join(t.get("name", "") for t in (x.get("tags") or []))[:400],
                                 duration_s=dur / 1000 if dur else None, size=x.get("filesize"), query=query))
        return out


class FreesoundSource:
    site = "Freesound"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, client: httpx.AsyncClient, query: str, limit: int = 12) -> list[Candidate]:
        r = await client.get("https://freesound.org/apiv2/search/", params={
            "query": query, "token": self.api_key, "page_size": limit,
            "filter": 'license:("Creative Commons 0" OR "Attribution") duration:[0.8 TO 180]',
            "fields": "id,name,description,license,username,url,duration,previews,tags"})
        if r.status_code in (401, 403):
            log.warning("Freesound Key 无效")
            return []
        r.raise_for_status()
        out = []
        for x in r.json().get("results", []):
            lic = normalize_license({"http://creativecommons.org/publicdomain/zero/1.0/": "cc0"}.get(x["license"], x["license"])
                                    if x["license"].startswith("http") else x["license"])
            if not lic and "creativecommons.org/licenses/by/" in x["license"]:
                lic = "CC-BY-" + (re.search(r"/by/(\d\.\d)", x["license"]).group(1) if re.search(r"/by/(\d\.\d)", x["license"]) else "4.0")
            if not lic:
                continue
            prev = (x.get("previews") or {}).get("preview-hq-mp3")
            if not prev:
                continue
            out.append(Candidate(site=self.site, title=x["name"], page_url=x["url"], download_url=prev, license=lic,
                                 license_url=x["license"], author=x["username"],
                                 description=(x.get("description") or "")[:400] + " " + " ".join(x.get("tags") or []),
                                 duration_s=x.get("duration"), query=query))
        return out


def ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=30)
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return 0.0


def transcode(src: Path, dst: Path, mono: bool, target_lufs: float, max_s: float) -> bool:
    """转成 mp3（44.1 kHz，128 kbps），响度归一化，截取到 max_s 秒，首尾短淡入淡出。"""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("需要安装 ffmpeg")
    dur = ffprobe_duration(src)
    length = min(dur, max_s) if dur else max_s
    fade_out_start = max(0.0, length - 0.15)
    af = (f"atrim=0:{length:.2f},afade=t=in:d=0.02,afade=t=out:st={fade_out_start:.2f}:d=0.15,"
          f"loudnorm=I={target_lufs}:TP=-2:LRA=11")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", "44100", "-ac", "1" if mono else "2",
           "-b:a", "128k", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        log.warning("转码失败 %s：%s", src.name, r.stderr[-300:])
        return False
    return True


def vocab_props(v: SoundVocab, duration: float) -> SoundProps:
    return SoundProps(mode=v.mode if v.mode in ("loop", "event") else "event",  # type: ignore[arg-type]
                      interval_range_s=(float(v.interval[0]), float(v.interval[1])), gain_db=v.gain,
                      acoustic_profile=v.profile if v.profile in ("clean", "old_radio", "loudspeaker_horn", "gramophone")
                      else "clean", duration_s=round(duration, 2), is_ambience=v.ambience)  # type: ignore[arg-type]


class SoundFetcher:
    def __init__(self, root: Path, registry: AssetRegistry, catalog: Catalog, freesound_key: str = "",
                 sfx=None):
        self.root = root
        self.registry = registry
        self.catalog = catalog
        self.sources: list = [CommonsSource(), OpenverseSource()]
        if freesound_key:
            self.sources.insert(0, FreesoundSource(freesound_key))
        self.sfx = sfx
        from ..config import get_settings
        self.lib_dir = get_settings().assets_dir / "sounds" / "library"
        self.lib_dir.mkdir(parents=True, exist_ok=True)

    def known_urls(self) -> set[str]:
        return {a.source.download_url for a in self.registry.all() if a.source.download_url}

    async def candidates(self, vocab: SoundVocab, extra_queries: list[str] = ()) -> list[Candidate]:
        queries = list(dict.fromkeys(list(extra_queries) + vocab.en + vocab.zh))
        words = [w for q in vocab.en for w in q.split()] + vocab.zh + list(extra_queries)
        found: dict[str, Candidate] = {}
        async with httpx.AsyncClient(timeout=30, headers={"User-Agent": UA}, follow_redirects=True) as c:
            for q in queries:
                for src in self.sources:
                    try:
                        for cand in await src.search(c, q):
                            cand.score = relevance(cand, words, self.catalog.sensitive_keywords)
                            if cand.download_url not in found or found[cand.download_url].score < cand.score:
                                found[cand.download_url] = cand
                    except Exception as e:
                        log.warning("%s 检索“%s”失败：%s", src.site, q, e)
                    await asyncio.sleep(2.0)  # 对公共 API 友好：串行、低频
        known = self.known_urls()
        cands = [x for x in found.values() if x.download_url not in known and x.score > 0
                 and (x.size is None or x.size < 40_000_000)]
        cands.sort(key=lambda x: -x.score)
        return cands

    async def download_and_register(self, vocab: SoundVocab, cand: Candidate, index: int) -> Optional[AssetRecord]:
        digest = hashlib.sha1(cand.download_url.encode()).hexdigest()[:8]
        from ..generation.service import slug
        aid = f"snd_{slug(vocab.key)}_{digest}"
        if self.registry.get(aid):
            return self.registry.get(aid)
        dst = self.lib_dir / f"{slug(vocab.key)}_{digest}.mp3"
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / ("src" + Path(cand.download_url.split("?")[0]).suffix)
            try:
                async with httpx.AsyncClient(timeout=120, headers={"User-Agent": UA}, follow_redirects=True) as c:
                    r = await c.get(cand.download_url, params={"token": getattr(self.sources[0], "api_key", None)}
                                    if cand.site == "Freesound" else None)
                    r.raise_for_status()
                    raw.write_bytes(r.content)
            except Exception as e:
                log.warning("下载失败 %s：%s", cand.download_url, e)
                return None
            loop_like = vocab.mode == "loop" or vocab.ambience
            ok = await asyncio.get_running_loop().run_in_executor(
                None, transcode, raw, dst, not (vocab.ambience or vocab.music),
                -26.0 if vocab.ambience else -20.0, 60.0 if loop_like else 20.0)
            if not ok:
                return None
        dur = ffprobe_duration(dst)
        sensitive = self.catalog.is_sensitive_text(f"{cand.title} {cand.description}")
        attribution = f"“{cand.title}” by {cand.author}, {cand.license}, via {cand.site}"
        rec = AssetRecord(
            id=aid, kind="music" if vocab.music else "sound", name_zh=vocab.key,
            tags=[vocab.key] + vocab.zh, synonyms=vocab.en, files=[to_rel(dst)],
            source=AssetSource(type="web", url=cand.page_url, download_url=cand.download_url, site=cand.site,
                               author=cand.author, original_title=cand.title,
                               notes=f"检索词：{cand.query}；原始描述：{cand.description[:200]}"),
            license=cand.license, license_url=cand.license_url,
            attribution=attribution if (cand.attribution_required or cand.license.startswith("CC-BY")) else None,
            sensitive=sensitive, copyright_status="public_domain" if cand.license in ("CC0-1.0", "Public-Domain") else "cc",
            review_status="pending", sound=vocab_props(vocab, dur))
        self.registry.register(rec)
        log.info("入库 %s ← %s（%s，%s）", aid, cand.title, cand.license, cand.site)
        return rec

    async def fetch_vocab(self, vocab: SoundVocab, per_key: int = 2, extra_queries: list[str] = ()) -> list[AssetRecord]:
        have = [a for a in self.registry.all() if a.kind in ("sound", "music") and vocab.key in a.tags and a.usable]
        need = max(0, per_key - len(have))
        if need == 0:
            return []
        got = []
        for cand in await self.candidates(vocab, extra_queries):
            if len(got) >= need:
                break
            rec = await self.download_and_register(vocab, cand, len(got))
            if rec and not rec.sensitive:
                got.append(rec)
        if not got and self.sfx is not None and getattr(self.sfx, "available", False):
            rec = await self.generate(vocab)
            if rec:
                got.append(rec)
        return got

    async def generate(self, vocab: SoundVocab) -> Optional[AssetRecord]:
        prompt = (vocab.en[0] if vocab.en else vocab.key) + ", recorded outdoors in an old Chinese town, natural, no music"
        try:
            data = await self.sfx.generate(prompt, duration_s=12 if vocab.mode == "loop" else 4, loop=vocab.mode == "loop")
        except Exception as e:
            log.warning("文生音效失败：%s", e)
            return None
        from ..generation.service import slug
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        dst = self.lib_dir / f"{slug(vocab.key)}_gen_{digest}.mp3"
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "gen.mp3"
            raw.write_bytes(data)
            transcode(raw, dst, not vocab.ambience, -26.0 if vocab.ambience else -20.0, 60.0)
        rec = AssetRecord(id=f"snd_{slug(vocab.key)}_gen_{digest}", kind="sound", name_zh=vocab.key,
                          tags=[vocab.key] + vocab.zh, synonyms=vocab.en, files=[to_rel(dst)],
                          source=AssetSource(type="ai", service=f"{self.sfx.name} 文生音效", prompt=prompt),
                          license="AI 生成（服务条款允许使用）", review_status="pending",
                          sound=vocab_props(vocab, ffprobe_duration(dst)))
        self.registry.register(rec)
        return rec
