"""统一素材登记：每个素材都必须有来源和授权记录。

- 共享素材库：assets/manifest.json（可提交到 git），自动生成 assets/REFERENCES.md
- 参与者专属的生成素材（个人化全景图）：data/generated_manifest.json（不提交）
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AssetKind = Literal["panorama", "depth", "object_image", "model3d", "sound", "ir", "music"]
SourceType = Literal["web", "ai", "own_recording", "restoration", "procedural"]
CopyrightStatus = Literal["public_domain", "cc", "copyrighted", "unknown", "not_applicable"]


class AssetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: SourceType
    url: Optional[str] = None  # 原始网页
    download_url: Optional[str] = None
    site: Optional[str] = None  # Wikimedia Commons / Freesound / …
    author: Optional[str] = None
    service: Optional[str] = None  # AI 生成服务和模型
    prompt: Optional[str] = None
    original_title: Optional[str] = None
    notes: Optional[str] = None


class SoundProps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["loop", "event"] = "event"
    interval_range_s: tuple[float, float] = (10.0, 30.0)
    gain_db: float = -6.0
    acoustic_profile: Literal["clean", "old_radio", "loudspeaker_horn", "gramophone"] = "clean"
    duration_s: float = 0.0
    loudness_lufs: Optional[float] = None
    is_ambience: bool = False


class AssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: AssetKind
    name_zh: str
    tags: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=lambda: ["any"])
    era: list[str] = Field(default_factory=lambda: ["any"])
    files: list[str]  # 相对项目根目录；多个文件 = 多个变体
    source: AssetSource
    license: str  # 如 CC0-1.0 / CC-BY-4.0 / CC-BY-SA-4.0 / AI-generated / 自有
    license_url: Optional[str] = None
    attribution: Optional[str] = None  # 署名要求（需要署名时必填）
    sensitive: bool = False
    copyright_status: CopyrightStatus = "not_applicable"
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    sound: Optional[SoundProps] = None
    style_key: Optional[str] = None  # 生成图片时的风格指纹（同风格可复用）
    participant_id: Optional[str] = None  # 只有个人化素材才有
    created_at: float = Field(default_factory=lambda: round(time.time(), 3))
    notes: str = ""

    @property
    def usable(self) -> bool:
        return self.review_status != "rejected" and bool(self.files)


class AssetRegistry:
    def __init__(self, root: Path, shared_path: Path, private_path: Path):
        self.root = root
        self.shared_path = shared_path
        self.private_path = private_path
        self._lock = threading.RLock()
        self.shared: dict[str, AssetRecord] = self._load(shared_path)
        self.private: dict[str, AssetRecord] = self._load(private_path)

    @staticmethod
    def _load(path: Path) -> dict[str, AssetRecord]:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return {a["id"]: AssetRecord.model_validate(a) for a in data.get("assets", [])}

    def reload(self) -> None:
        with self._lock:
            self.shared = self._load(self.shared_path)
            self.private = self._load(self.private_path)

    def all(self) -> Iterable[AssetRecord]:
        yield from self.shared.values()
        yield from self.private.values()

    def get(self, asset_id: str) -> Optional[AssetRecord]:
        return self.shared.get(asset_id) or self.private.get(asset_id)

    def by_kind(self, kind: str) -> list[AssetRecord]:
        return [a for a in self.all() if a.kind == kind and a.usable]

    def register(self, rec: AssetRecord, private: bool = False) -> AssetRecord:
        AssetRecord.model_validate(rec.model_dump())
        if rec.source.type == "web" and rec.license.lower() in ("", "unknown", "未知"):
            raise ValueError("网上素材必须有明确的授权协议")
        if rec.license.upper().startswith("CC-BY") and not rec.attribution:
            raise ValueError("CC-BY 类素材必须填写署名")
        with self._lock:
            self._merge_from_disk()  # 其他进程（如下载脚本）可能刚写过
            (self.private if private else self.shared)[rec.id] = rec
            self.save()
        return rec

    def _merge_from_disk(self) -> None:
        for path, items in ((self.shared_path, self.shared), (self.private_path, self.private)):
            try:
                for k, v in self._load(path).items():
                    items.setdefault(k, v)
            except Exception:
                pass

    def remove(self, asset_id: str) -> None:
        with self._lock:
            self.shared.pop(asset_id, None)
            self.private.pop(asset_id, None)
            self.save()

    def remove_participant(self, participant_id: str) -> list[AssetRecord]:
        with self._lock:
            gone = [a for a in self.private.values() if a.participant_id == participant_id]
            for a in gone:
                self.private.pop(a.id, None)
            self.save()
            return gone

    def save(self) -> None:
        with self._lock:
            for path, items in ((self.shared_path, self.shared), (self.private_path, self.private)):
                path.parent.mkdir(parents=True, exist_ok=True)
                data = {"generated_at": round(time.time()), "count": len(items),
                        "assets": [a.model_dump(mode="json") for a in sorted(items.values(), key=lambda x: x.id)]}
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(path)
            write_references_md(self.shared_path.parent / "REFERENCES.md", self.shared.values())

    def path(self, rec: AssetRecord, variant: int = 0) -> Path:
        from ..config import to_abs
        return to_abs(rec.files[variant % len(rec.files)])


KIND_ZH = {"panorama": "全景图", "depth": "深度图", "object_image": "物件插画", "model3d": "3D 模型",
           "sound": "声音", "ir": "冲激响应", "music": "音乐/戏曲"}


def write_references_md(path: Path, assets: Iterable[AssetRecord]) -> None:
    assets = sorted(assets, key=lambda a: (a.kind, a.id))
    lines = ["# 素材来源与授权（自动生成，请勿手改）", "",
             "本文件由 `assets/manifest.json` 自动生成。每个素材的来源、授权和署名要求如下。", ""]
    by_kind: dict[str, list[AssetRecord]] = {}
    for a in assets:
        by_kind.setdefault(a.kind, []).append(a)
    for kind, items in by_kind.items():
        lines += [f"## {KIND_ZH.get(kind, kind)}（{len(items)}）", "",
                  "| ID | 名称 | 来源 | 作者 | 授权 | 署名 | 版权状态 | 敏感 | 审核 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for a in items:
            src = a.source
            if src.type == "ai":
                where = f"AI 生成（{src.service or ''}）"
            elif src.url:
                where = f"[{src.site or '链接'}]({src.url})"
            else:
                where = {"own_recording": "自有录音", "restoration": "自行修复", "procedural": "程序合成"}.get(src.type, src.type)
            cell = lambda s: (s or "").replace("|", "/").replace("\n", " ")[:80]  # noqa: E731
            lines.append(f"| `{a.id}` | {cell(a.name_zh)} | {where} | {cell(src.author)} | {cell(a.license)} | "
                         f"{cell(a.attribution)} | {a.copyright_status} | {'是' if a.sensitive else ''} | {a.review_status} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
