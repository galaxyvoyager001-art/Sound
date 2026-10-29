"""LLM 结构化输出的 Pydantic 模型（同时生成 JSON Schema 发给模型）。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..scene.commands import RawCommand
from ..scene.models import RegionStyle, Season, TimeOfDay, Weather

Label = Literal["edit", "navigate", "story", "control", "other"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NavigateIntent(_S):
    kind: Literal["look_direction", "look_place", "go_place", "go_back"]
    direction: Optional[Literal["left", "right", "back", "front"]] = None
    place_name: Optional[str] = None
    landmark_id: Optional[str] = None
    viewpoint_id: Optional[str] = None


class UtteranceAnalysis(_S):
    labels: list[Label]
    confirmation: Literal["yes", "no", "none"] = "none"
    control_kind: Literal["quiet", "resume", "end", "undo", "none"] = "none"
    emotion: Literal["neutral", "happy", "sad", "anxious", "tired"] = "neutral"
    navigate: Optional[NavigateIntent] = None


class SceneMeta(_S):
    region_style: Optional[RegionStyle] = None
    era: Optional[str] = Field(None, description="如“1960年代”")
    time_of_day: Optional[TimeOfDay] = None
    weather: Optional[Weather] = None
    season: Optional[Season] = None


class SceneUpdate(_S):
    meta: Optional[SceneMeta] = None
    viewpoint_description: Optional[str] = None
    commands: list[RawCommand]


class CompanionTurn(_S):
    speak: bool
    text: str = Field(description="要说的话；speak=false 时为空字符串")


class StoryArchive(_S):
    is_story: bool
    title_zh: str
    anchor_kind: Literal["object", "landmark", "viewpoint"]
    anchor_id: str


class SoundKeywords(_S):
    en: list[str]
    zh: list[str]
    mode: Literal["loop", "event"]
    is_sensitive: bool


class DetailItem(_S):
    text: str
    quote: str


class SensoryItem(_S):
    kind: Literal["sound", "smell", "color", "temperature", "taste", "touch"]
    text: str
    quote: str


class DetailExtraction(_S):
    persons: list[DetailItem]
    places: list[DetailItem]
    times: list[DetailItem]
    sensory: list[SensoryItem]
    events: list[DetailItem]
