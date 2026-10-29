"""场景清单（SceneManifest）：画面和声音的唯一数据来源。

约定：
- 方位角 yaw_deg：0 = 全景图正中（等距柱状投影 u=0.5），正值向右（顺时针），范围 (-180, 180]。
- 俯仰角 pitch_deg：0 = 地平线，正值向上。
- 距离以米为单位；观看者眼睛高度固定为 EYE_HEIGHT_M。
"""
from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EYE_HEIGHT_M = 1.6
SCHEMA_VERSION = 1

RegionStyle = Literal["上海弄堂", "北京胡同", "江南水乡", "北方村庄", "厂区大院", "其他"]
TimeOfDay = Literal["清晨", "上午", "中午", "下午", "傍晚", "夜晚"]
Weather = Literal["晴", "阴", "小雨", "大雨", "雪", "雾"]
Season = Literal["春", "夏", "秋", "冬"]
AcousticProfile = Literal["clean", "old_radio", "loudspeaker_horn", "gramophone"]
SoundMode = Literal["loop", "event"]
AddedBy = Literal["narration", "edit", "generic", "system"]
Condition = Literal["audio_only", "generic", "personal"]
Relation = Literal["at", "farther", "nearer", "left", "right", "beside", "opposite"]
AssetStatus = Literal["pending", "generating", "ready", "failed"]

TIMES_OF_DAY: tuple[str, ...] = TimeOfDay.__args__  # type: ignore[attr-defined]
WEATHERS: tuple[str, ...] = Weather.__args__  # type: ignore[attr-defined]
SEASONS: tuple[str, ...] = Season.__args__  # type: ignore[attr-defined]
REGION_STYLES: tuple[str, ...] = RegionStyle.__args__  # type: ignore[attr-defined]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def now_ts() -> float:
    return round(time.time(), 3)


def norm_yaw(y: float) -> float:
    """把角度规范到 (-180, 180]。"""
    y = (y + 180.0) % 360.0 - 180.0
    return 180.0 if y == -180.0 else round(y, 3)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Viewpoint(_Base):
    id: str
    name_zh: str
    description_zh: str = ""  # 老人描述的样子（用于生成/重绘）
    panorama_path: Optional[str] = None  # 相对 data/ 或 assets/ 的路径，由服务端提供 URL
    depth_path: Optional[str] = None
    panorama_asset_id: Optional[str] = None
    generation_prompt: str = ""
    neighbors: list[str] = Field(default_factory=list)
    status: AssetStatus = "pending"
    generated_for: dict[str, str] = Field(default_factory=dict)  # 生成时的季节/天气等（用于判断是否需要重绘）


class Landmark(_Base):
    id: str
    name_zh: str
    aliases: list[str] = Field(default_factory=list)
    viewpoint_id: str
    yaw_deg: float
    distance_m: float = Field(gt=0.3, le=200)
    linked_viewpoint_id: Optional[str] = None  # 走过去之后对应的点位（如“井边”点位）
    builtin: bool = False  # 系统预置的方向地标（正前方/左边…）

    @field_validator("yaw_deg")
    @classmethod
    def _yaw(cls, v: float) -> float:
        return norm_yaw(v)


class Offset(_Base):
    relation: Relation = "at"
    dyaw_deg: float = 0.0
    ddist_m: float = 0.0


class Placement(_Base):
    """解析后的具体位置（由后端根据地标 + 偏移 + 深度图计算）。"""
    viewpoint_id: str
    yaw_deg: float
    distance_m: float = Field(gt=0.3, le=200)
    elevation_m: float = 0.0  # 物件底部离地高度（如窗台上的收音机）
    occluded: float = Field(0.0, ge=0, le=1)  # 粗略遮挡系数（前端会逐帧细算）

    @field_validator("yaw_deg")
    @classmethod
    def _yaw(cls, v: float) -> float:
        return norm_yaw(v)


class SoundSpec(_Base):
    clip_id: str
    variant_index: int = Field(0, ge=0)
    gain_db: float = Field(0.0, ge=-40, le=6)
    mode: SoundMode = "loop"
    interval_range_s: tuple[float, float] = (8.0, 25.0)
    acoustic_profile: AcousticProfile = "clean"
    ref_distance_m: float = Field(3.0, gt=0)
    rolloff: float = Field(1.0, ge=0, le=10)
    max_distance_m: float = Field(80.0, gt=0)

    @field_validator("interval_range_s")
    @classmethod
    def _interval(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo <= 0 or hi < lo:
            raise ValueError("interval_range_s 必须满足 0 < 最小值 ≤ 最大值")
        return (float(lo), float(hi))


class SceneObject(_Base):
    id: str
    type: str  # 物件类型（中文），如“馄饨担子”
    label_zh: str = ""  # 老人的叫法，如“阿福伯的馄饨担子”
    asset_id: Optional[str] = None  # 插画素材
    model_asset_id: Optional[str] = None  # 3D 模型素材（可选）
    asset_status: AssetStatus = "pending"
    landmark: str
    offset: Offset = Field(default_factory=Offset)
    placement: Optional[Placement] = None
    height_m: float = Field(1.2, gt=0.05, le=15)
    sound: Optional[SoundSpec] = None
    added_by: AddedBy = "edit"
    created_at: float = Field(default_factory=now_ts)


class AmbienceLayer(_Base):
    id: str
    clip_id: str
    name_zh: str = ""
    gain_db: float = Field(-12.0, ge=-40, le=6)
    role: Literal["base", "time", "weather", "season", "manual"] = "manual"
    auto: bool = False  # 由时段/天气/季节规则自动选出，规则变化时会被替换


class Anchor(_Base):
    kind: Literal["object", "landmark", "viewpoint"]
    id: str


class Story(_Base):
    id: str
    title_zh: str = ""
    transcript: str
    audio_path: Optional[str] = None
    start_s: float = 0.0
    end_s: float = 0.0
    anchor: Anchor
    session_id: Optional[str] = None
    created_at: float = Field(default_factory=now_ts)
    include_in_export: bool = True


class UnresolvedItem(_Base):
    id: str
    mention: str  # 老人的原话片段
    kind: Literal["object", "sound", "viewpoint", "ambience", "sensitive"]
    reason: str = ""
    status: Literal["open", "searching", "resolved", "dismissed"] = "open"
    created_at: float = Field(default_factory=now_ts)


class SceneManifest(_Base):
    # 整体引用校验只在构造/提交时做，执行指令过程中允许暂时不一致
    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    schema_version: int = SCHEMA_VERSION
    scene_id: str
    participant_id: str  # 只存编号，不存真名
    version: int = 0
    condition: Condition = "personal"
    region_style: RegionStyle = "上海弄堂"
    era: str = "1960年代"
    time_of_day: TimeOfDay = "上午"
    weather: Weather = "晴"
    season: Season = "夏"
    entry_viewpoint: Optional[str] = None
    viewpoints: list[Viewpoint] = Field(default_factory=list)
    landmarks: list[Landmark] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    ambience: list[AmbienceLayer] = Field(default_factory=list)
    stories: list[Story] = Field(default_factory=list)
    unresolved: list[UnresolvedItem] = Field(default_factory=list)
    updated_at: float = Field(default_factory=now_ts)

    @model_validator(mode="after")
    def _check_refs(self) -> "SceneManifest":
        def unique(items, what):
            ids = [i.id for i in items]
            dup = {i for i in ids if ids.count(i) > 1}
            if dup:
                raise ValueError(f"{what} 的 id 重复：{sorted(dup)}")
            return set(ids)

        vp_ids = unique(self.viewpoints, "viewpoints")
        lm_ids = unique(self.landmarks, "landmarks")
        obj_ids = unique(self.objects, "objects")
        unique(self.ambience, "ambience")
        unique(self.stories, "stories")
        unique(self.unresolved, "unresolved")
        if self.entry_viewpoint and self.entry_viewpoint not in vp_ids:
            raise ValueError(f"entry_viewpoint {self.entry_viewpoint} 不存在")
        for vp in self.viewpoints:
            for n in vp.neighbors:
                if n not in vp_ids:
                    raise ValueError(f"点位 {vp.id} 的相邻点位 {n} 不存在")
        for lm in self.landmarks:
            if lm.viewpoint_id not in vp_ids:
                raise ValueError(f"地标 {lm.id} 所在点位 {lm.viewpoint_id} 不存在")
            if lm.linked_viewpoint_id and lm.linked_viewpoint_id not in vp_ids:
                raise ValueError(f"地标 {lm.id} 链接的点位不存在")
        for o in self.objects:
            if o.landmark not in lm_ids:
                raise ValueError(f"物件 {o.id} 的地标 {o.landmark} 不存在")
            if o.placement and o.placement.viewpoint_id not in vp_ids:
                raise ValueError(f"物件 {o.id} 的位置点位不存在")
        for s in self.stories:
            pool = {"object": obj_ids, "landmark": lm_ids, "viewpoint": vp_ids}[s.anchor.kind]
            if s.anchor.id not in pool:
                raise ValueError(f"故事 {s.id} 的锚点 {s.anchor.kind}:{s.anchor.id} 不存在")
        return self

    # ---- 便捷查询 ----
    def viewpoint(self, vid: str) -> Optional[Viewpoint]:
        return next((v for v in self.viewpoints if v.id == vid), None)

    def landmark(self, lid: str) -> Optional[Landmark]:
        return next((l for l in self.landmarks if l.id == lid), None)

    def obj(self, oid: str) -> Optional[SceneObject]:
        return next((o for o in self.objects if o.id == oid), None)

    def ambience_layer(self, aid: str) -> Optional[AmbienceLayer]:
        return next((a for a in self.ambience if a.id == aid), None)

    def landmarks_of(self, vid: str) -> list[Landmark]:
        return [l for l in self.landmarks if l.viewpoint_id == vid]

    def objects_of(self, vid: str) -> list[SceneObject]:
        out = []
        for o in self.objects:
            vp = o.placement.viewpoint_id if o.placement else (self.landmark(o.landmark).viewpoint_id if self.landmark(o.landmark) else None)
            if vp == vid:
                out.append(o)
        return out

    def find_landmark_by_name(self, name: str, viewpoint_id: Optional[str] = None) -> Optional[Landmark]:
        name = name.strip()
        cands = [l for l in self.landmarks if viewpoint_id is None or l.viewpoint_id == viewpoint_id]
        for l in cands:
            if l.name_zh == name or name in l.aliases:
                return l
        for l in cands:  # 宽松匹配：“井边” ~ “井”
            if name and (name in l.name_zh or l.name_zh in name):
                return l
        return None

    def find_viewpoint_by_name(self, name: str) -> Optional[Viewpoint]:
        name = name.strip()
        for v in self.viewpoints:
            if v.name_zh == name:
                return v
        for v in self.viewpoints:
            if name and (name in v.name_zh or v.name_zh in name):
                return v
        return None

    def summary_for_llm(self, current_viewpoint: Optional[str] = None) -> dict:
        """给 LLM 的精简上下文（不含路径等无关字段）。"""
        return {
            "region_style": self.region_style, "era": self.era,
            "time_of_day": self.time_of_day, "weather": self.weather, "season": self.season,
            "current_viewpoint": current_viewpoint,
            "viewpoints": [{"id": v.id, "name": v.name_zh, "description": v.description_zh[:120],
                            "neighbors": v.neighbors} for v in self.viewpoints],
            "landmarks": [{"id": l.id, "name": l.name_zh, "viewpoint": l.viewpoint_id,
                           "yaw": round(l.yaw_deg), "distance_m": round(l.distance_m, 1)}
                          for l in self.landmarks],
            "objects": [{"id": o.id, "type": o.type, "label": o.label_zh, "landmark": o.landmark,
                         "relation": o.offset.relation,
                         "sound": (o.sound.clip_id if o.sound else None)} for o in self.objects],
            "ambience": [{"id": a.id, "name": a.name_zh or a.clip_id, "role": a.role} for a in self.ambience],
        }
