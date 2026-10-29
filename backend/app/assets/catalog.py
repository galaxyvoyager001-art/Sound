"""读取 config/catalog.yaml 与 config/style.yaml。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from ..config import ROOT


@dataclass
class ObjectEntry:
    type: str
    synonyms: list[str] = field(default_factory=list)
    height_m: float = 1.2
    elevation_m: float = 0.0
    sound: Optional[str] = None
    person: bool = False
    prompt: str = ""


@dataclass
class SoundVocab:
    key: str
    zh: list[str]
    en: list[str]
    mode: str = "event"
    interval: tuple[float, float] = (10.0, 30.0)
    gain: float = -6.0
    profile: str = "clean"
    ambience: bool = False
    music: bool = False


@dataclass
class Catalog:
    objects: list[ObjectEntry]
    sounds: list[SoundVocab]
    ambience_rules: list[dict]
    sensitive_keywords: list[str]
    control_phrases: dict[str, list[str]]
    style: dict

    def find_object(self, name: str) -> Optional[ObjectEntry]:
        name = name.strip()
        for o in self.objects:
            if name == o.type or name in o.synonyms:
                return o
        best, best_len = None, 0
        for o in self.objects:  # 包含关系：“阿福伯的馄饨担子” → 馄饨担子
            for cand in [o.type] + o.synonyms:
                if len(cand) >= 2 and cand in name and len(cand) > best_len:
                    best, best_len = o, len(cand)
        return best

    def sound_vocab(self, key: str) -> Optional[SoundVocab]:
        return next((s for s in self.sounds if s.key == key), None)

    def is_sensitive_text(self, text: str) -> bool:
        t = text.lower()
        return any(k.lower() in t for k in self.sensitive_keywords)


def load_catalog(config_dir: Path = ROOT / "config") -> Catalog:
    data = yaml.safe_load((config_dir / "catalog.yaml").read_text(encoding="utf-8"))
    style = yaml.safe_load((config_dir / "style.yaml").read_text(encoding="utf-8"))
    objects = [ObjectEntry(type=o["type"], synonyms=o.get("synonyms") or [], height_m=o.get("height_m", 1.2),
                           elevation_m=o.get("elevation_m", 0.0), sound=o.get("sound"), person=o.get("person", False),
                           prompt=o.get("prompt", "")) for o in data["objects"]]
    sounds = [SoundVocab(key=s["key"], zh=s.get("zh") or [], en=s.get("en") or [], mode=s.get("mode", "event"),
                         interval=tuple(s.get("interval", (10, 30))), gain=s.get("gain", -6.0),
                         profile=s.get("profile", "clean"), ambience=s.get("ambience", False),
                         music=s.get("music", False)) for s in data["sounds"]]
    return Catalog(objects=objects, sounds=sounds, ambience_rules=data.get("ambience_rules", []),
                   sensitive_keywords=data.get("sensitive_keywords", []),
                   control_phrases=data.get("control_phrases", {}), style=style)


@lru_cache
def get_catalog() -> Catalog:
    return load_catalog()
