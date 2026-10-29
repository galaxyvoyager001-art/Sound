"""把老人的说法映射到声音素材：标签 + 同义词 + 拼音模糊匹配 + 文本向量相似度，融合排序。

声音以“组”为单位使用：同一逻辑名称（如“木梆子”）的多段录音组成一组 `grp:木梆子`，
事件模式下随机轮换、避免机械重复；“换一个”就是换到组内下一段。
找不到足够好的匹配时返回 None（放进 unresolved），绝不硬塞不相关的素材；敏感素材永远不返回。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pypinyin import Style, lazy_pinyin

from ..providers.embedding import Embedder, HashEmbedder
from ..scene.executor import ObjectMeta, SoundMatch
from ..scene.models import AmbienceLayer, SceneManifest
from .catalog import Catalog
from .registry import AssetRecord, AssetRegistry

log = logging.getLogger("retrieval")


def pinyin_str(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))


INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def _fuzz(syl: str) -> str:
    """方言与识别常见混淆：平翘舌（zh/z）、前后鼻音（ng/n）、n/l、f/h。"""
    syl = re.sub(r"^([zcs])h", r"\1", syl).replace("ng", "n")
    syl = re.sub(r"^l", "n", syl)
    return re.sub(r"^f", "h", syl)


def _split(syl: str) -> tuple[str, str]:
    for i in INITIALS:
        if syl.startswith(i):
            return i, syl[len(i):]
    return "", syl


def _syl_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ia, fa = _split(a)
    ib, fb = _split(b)
    if fa == fb:
        return 0.6  # 韵母相同（如 dun / tun）
    if ia and ia == ib:
        return 0.25
    return 0.0


def pinyin_sim(a: str, b: str) -> float:
    """拼音相似度：逐音节对齐（容忍同音字、识别错字和口音），取最佳滑动对齐的平均分。
    如“混沌 hun dun”≈“馄饨 hun tun”。"""
    pa = [_fuzz(x) for x in pinyin_str(a).split()]
    pb = [_fuzz(x) for x in pinyin_str(b).split()]
    if not pa or not pb:
        return 0.0
    short, long_ = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    best = 0.0
    for off in range(len(long_) - len(short) + 1):
        sc = sum(_syl_sim(x, long_[off + i]) for i, x in enumerate(short)) / len(long_)
        best = max(best, sc)
    return best


def tokens(text: str) -> list[str]:
    parts = re.split(r"[\s,，、。;；/的了在那个这个]+", text.strip())
    return [p for p in parts if p]


@dataclass
class SoundGroup:
    key: str
    assets: list[AssetRecord]
    text: str = ""
    names: list[str] = field(default_factory=list)

    @property
    def clip_id(self) -> str:
        return f"grp:{self.key}"

    @property
    def files(self) -> list[str]:
        out = []
        for a in sorted(self.assets, key=lambda a: (a.review_status != "approved", a.created_at)):
            out.extend(a.files)
        return out

    @property
    def lead(self) -> AssetRecord:
        return sorted(self.assets, key=lambda a: (a.review_status != "approved", a.created_at))[0]


class AssetResolverImpl:
    MIN_SCORE = 0.42

    def __init__(self, registry: AssetRegistry, catalog: Catalog, embedder: Optional[Embedder] = None):
        self.registry = registry
        self.catalog = catalog
        self.embedder = embedder or HashEmbedder()
        self._local = HashEmbedder()
        self._groups: dict[str, SoundGroup] = {}
        self._emb: Optional[np.ndarray] = None
        self._emb_keys: list[str] = []
        self._emb_name = ""
        self.refresh()

    # ---------- 组 ----------
    def refresh(self) -> None:
        groups: dict[str, SoundGroup] = {}
        for a in self.registry.all():
            if a.kind not in ("sound", "music") or not a.usable or a.sensitive:
                continue
            key = a.tags[0] if a.tags else a.name_zh
            groups.setdefault(key, SoundGroup(key, []))
            groups[key].assets.append(a)
        for g in groups.values():
            vocab = self.catalog.sound_vocab(g.key)
            names = {g.key}
            for a in g.assets:
                names.update(a.tags)
                names.update(a.synonyms)
                names.add(a.name_zh)
            if vocab:
                names.update(vocab.zh)
                names.update(vocab.en)
            # 物件目录里以这个声音为默认声音的物件，它们的叫法也算同义词
            for o in self.catalog.objects:
                if o.sound and g.key in o.sound.split():
                    names.add(o.type)
                    names.update(o.synonyms)
            g.names = sorted(n for n in names if n)
            g.text = " ".join(g.names)
        self._groups = groups
        self._emb = None

    def group(self, clip_id: str) -> Optional[SoundGroup]:
        return self._groups.get(clip_id.removeprefix("grp:")) if clip_id.startswith("grp:") else None

    def clip_files(self, clip_id: str) -> list[str]:
        g = self.group(clip_id)
        if g:
            return g.files
        a = self.registry.get(clip_id)
        return list(a.files) if a else []

    async def _embeddings(self) -> tuple[Embedder, np.ndarray]:
        if self._emb is not None:
            return (self.embedder if self._emb_name == self.embedder.name else self._local), self._emb
        keys = list(self._groups)
        texts = [self._groups[k].text for k in keys]
        try:
            emb = await self.embedder.embed(texts) if texts else np.zeros((0, self.embedder.dim))
            used = self.embedder
        except Exception as e:
            log.warning("文本向量服务不可用，检索改用本地向量：%s", e)
            emb = await self._local.embed(texts) if texts else np.zeros((0, self._local.dim))
            used = self._local
        self._emb, self._emb_keys, self._emb_name = emb, keys, used.name
        return used, emb

    # ---------- 检索 ----------
    def expand_query(self, query: str) -> list[str]:
        """同义词扩展：物件叫法 → 默认声音；声音词表的中英文关键词。"""
        terms = [query] + tokens(query)
        obj = self.catalog.find_object(query)
        if obj and obj.sound:
            terms += obj.sound.split()
        for v in self.catalog.sounds:
            if any(z in query for z in v.zh) or v.key in query:
                terms.append(v.key)
        return list(dict.fromkeys(t for t in terms if t))

    async def rank(self, query: str, purpose: str = "object", exclude: tuple[str, ...] = ()) -> list[tuple[float, SoundGroup, dict]]:
        if not self._groups:
            return []
        if self.catalog.is_sensitive_text(query):
            return []
        terms = self.expand_query(query)
        used, emb = await self._embeddings()
        try:
            q = (await used.embed([query]))[0]
            sims = emb @ q if len(emb) else np.zeros(0)
        except Exception:
            sims = np.zeros(len(self._emb_keys))
        out = []
        for i, key in enumerate(self._emb_keys):
            g = self._groups[key]
            if g.clip_id in exclude or any(a.id in exclude for a in g.assets):
                continue
            tag = 0.0
            for t in terms:
                if t == g.key or t in g.names:
                    tag = max(tag, 1.0 if t in (query, g.key) else 0.85)
                elif len(t) >= 2 and any(t in n or (len(n) >= 2 and n in t) for n in g.names):
                    tag = max(tag, 0.6)
            py = max((pinyin_sim(t, n) for t in terms[:4] for n in g.names if not n.isascii()), default=0.0)
            em = float(sims[i]) if i < len(sims) else 0.0
            # 本地哈希向量只是兜底，权重降低
            em_w = 0.35 if used.name != "hash" else 0.15
            # 词面证据：标签/同义词命中，或拼音几乎完全相同（同音字、识别错字）
            lex = max(tag, 0.9 * py if py >= 0.95 else 0.55 * py)
            score = 0.6 * lex + em_w * max(0.0, em)
            lead = g.lead
            amb = bool(lead.sound and lead.sound.is_ambience)
            if purpose == "ambience" and amb or purpose == "object" and not amb:
                score += 0.05
            if lead.review_status == "approved":
                score += 0.03
            out.append((score, g, {"tag": tag, "pinyin": round(py, 3), "embed": round(em, 3)}))
        out.sort(key=lambda x: -x[0])
        return out

    async def find_sound(self, query: str, manifest: SceneManifest, purpose: str = "object",
                         exclude: tuple[str, ...] = ()) -> Optional[SoundMatch]:
        ranked = await self.rank(query, purpose, exclude)
        if not ranked:
            return None
        score, g, parts = ranked[0]
        # 必须有“词面”证据（标签/同义词/拼音）或很高的语义相似度，才接受
        strong_text = parts["tag"] >= 0.6 or parts["pinyin"] >= 0.5
        strong_sem = parts["embed"] >= 0.72 and self.embedder.name != "hash"
        if score < self.MIN_SCORE or not (strong_text or strong_sem):
            log.info("“%s”没有足够好的声音匹配（最高 %s %.2f %s）", query, g.key, score, parts)
            return None
        lead = g.lead
        sp = lead.sound
        return SoundMatch(clip_id=g.clip_id, name_zh=g.key, mode=sp.mode if sp else "event",
                          interval_range_s=tuple(sp.interval_range_s) if sp else (10.0, 30.0),
                          acoustic_profile=sp.acoustic_profile if sp else "clean",
                          gain_db=sp.gain_db if sp else -6.0, variants=len(g.files), score=round(score, 3))

    # ---------- 其他 AssetResolver 接口 ----------
    def object_meta(self, object_type: str) -> ObjectMeta:
        o = self.catalog.find_object(object_type)
        if not o:
            return ObjectMeta(1.2, 0.0, object_type, False)
        return ObjectMeta(o.height_m, o.elevation_m, o.sound, bool(o.sound))

    def sound_variants(self, clip_id: str) -> int:
        return max(1, len(self.clip_files(clip_id)))

    def auto_ambience(self, m: SceneManifest) -> list[AmbienceLayer]:
        out: list[AmbienceLayer] = []
        seen = set()
        for rule in self.catalog.ambience_rules:
            w = rule.get("when", {})
            ok = all([
                m.region_style in w.get("region", [m.region_style]),
                m.time_of_day in w.get("time", [m.time_of_day]),
                m.weather in w.get("weather", [m.weather]),
                m.season in w.get("season", [m.season]),
            ])
            if not ok:
                continue
            g = self._groups.get(rule["sound"])
            if not g or g.clip_id in seen:
                continue
            seen.add(g.clip_id)
            from ..generation.service import slug
            out.append(AmbienceLayer(id=f"amb_auto_{slug(g.key)}", clip_id=g.clip_id, name_zh=g.key,
                                     gain_db=float(rule.get("gain", -16)), role=rule.get("role", "base"), auto=True))
        return out
