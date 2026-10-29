"""修改指令。

LLM 只输出 `RawCommand`（扁平结构，便于 JSON Schema 约束），
后端用 `to_typed()` 转成强类型指令并逐条校验，再交给执行器。
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .models import Relation, Season, TimeOfDay, Weather

Op = Literal[
    "add_object", "remove_object", "move_object", "replace_sound", "set_gain",
    "set_time", "set_weather", "set_season", "regenerate_viewpoint", "add_viewpoint",
    "add_ambience", "remove_ambience", "clarify",
]
Direction = Literal["front", "back", "left", "right", "front_left", "front_right", "back_left", "back_right"]
DistanceWord = Literal["near", "mid", "far"]
GainChange = Literal["louder", "quieter", "much_louder", "much_quieter", "mute"]


class RawCommand(BaseModel):
    """LLM 输出的一条指令（所有字段都出现，不用的填 null）。"""
    model_config = ConfigDict(extra="forbid")
    op: Op
    confidence: float = Field(ge=0, le=1, description="0~1，对理解正确的把握")
    target_id: Optional[str] = Field(None, description="要修改的物件/环境声 id（来自当前场景清单）")
    object_type: Optional[str] = Field(None, description="新物件的类型（中文，如“馄饨担子”“自行车”）")
    label_zh: Optional[str] = Field(None, description="老人的叫法，如“阿福伯的馄饨担子”")
    landmark_id: Optional[str] = Field(None, description="已有地标 id")
    landmark_name: Optional[str] = Field(None, description="新地标名称（当前清单里没有时）")
    direction: Optional[Direction] = Field(None, description="新地标相对观看者的方向")
    distance: Optional[DistanceWord] = Field(None, description="新地标的远近")
    relation: Optional[Relation] = Field(None, description="相对地标的位置")
    viewpoint_id: Optional[str] = None
    sound_query: Optional[str] = Field(None, description="想要的声音（中文描述）")
    next_variant: Optional[bool] = Field(None, description="replace_sound：只换同一种声音的另一个版本")
    gain_change: Optional[GainChange] = None
    time_of_day: Optional[TimeOfDay] = None
    weather: Optional[Weather] = None
    season: Optional[Season] = None
    description: Optional[str] = Field(None, description="点位/画面的描述（老人说的样子）")
    confirm_question: Optional[str] = Field(None, description="低置信度时，用一句话向老人确认")


class RawCommandList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    commands: list[RawCommand]


# ---------- 强类型指令 ----------
class _Cmd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence: float = 1.0
    confirm_question: Optional[str] = None


class LocationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    landmark_id: Optional[str] = None
    landmark_name: Optional[str] = None
    direction: Optional[Direction] = None
    distance: Optional[DistanceWord] = None
    relation: Relation = "at"
    viewpoint_id: Optional[str] = None


class AddObject(_Cmd):
    op: Literal["add_object"] = "add_object"
    object_type: str
    label_zh: str = ""
    location: LocationRef = Field(default_factory=LocationRef)
    sound_query: Optional[str] = None


class RemoveObject(_Cmd):
    op: Literal["remove_object"] = "remove_object"
    object_id: str


class MoveObject(_Cmd):
    op: Literal["move_object"] = "move_object"
    object_id: str
    location: LocationRef


class ReplaceSound(_Cmd):
    op: Literal["replace_sound"] = "replace_sound"
    target_id: str
    sound_query: Optional[str] = None
    next_variant: bool = False


class SetGain(_Cmd):
    op: Literal["set_gain"] = "set_gain"
    target_id: str
    gain_change: GainChange


class SetTime(_Cmd):
    op: Literal["set_time"] = "set_time"
    time_of_day: TimeOfDay


class SetWeather(_Cmd):
    op: Literal["set_weather"] = "set_weather"
    weather: Weather


class SetSeason(_Cmd):
    op: Literal["set_season"] = "set_season"
    season: Season


class RegenerateViewpoint(_Cmd):
    op: Literal["regenerate_viewpoint"] = "regenerate_viewpoint"
    viewpoint_id: str
    description: str


class AddViewpoint(_Cmd):
    op: Literal["add_viewpoint"] = "add_viewpoint"
    name_zh: str
    description: str = ""
    connect_to: Optional[str] = None


class AddAmbience(_Cmd):
    op: Literal["add_ambience"] = "add_ambience"
    sound_query: str


class RemoveAmbience(_Cmd):
    op: Literal["remove_ambience"] = "remove_ambience"
    ambience_id: str


class Clarify(_Cmd):
    op: Literal["clarify"] = "clarify"
    question: str


EditCommand = Annotated[
    Union[AddObject, RemoveObject, MoveObject, ReplaceSound, SetGain, SetTime, SetWeather, SetSeason,
          RegenerateViewpoint, AddViewpoint, AddAmbience, RemoveAmbience, Clarify],
    Field(discriminator="op"),
]
EditCommandAdapter: TypeAdapter = TypeAdapter(EditCommand)


class CommandError(ValueError):
    """指令不完整或与场景不符。"""


def _need(v, name: str, op: str):
    if v is None or (isinstance(v, str) and not v.strip()):
        raise CommandError(f"{op} 缺少 {name}")
    return v


def to_typed(raw: RawCommand, current_viewpoint: Optional[str] = None) -> EditCommand:
    """把 LLM 的扁平指令转换为强类型指令；字段缺失时抛 CommandError。"""
    op = raw.op
    base = {"confidence": raw.confidence, "confirm_question": raw.confirm_question}
    loc = LocationRef(
        landmark_id=raw.landmark_id, landmark_name=raw.landmark_name, direction=raw.direction,
        distance=raw.distance, relation=raw.relation or "at",
        viewpoint_id=raw.viewpoint_id or current_viewpoint,
    )
    if op == "add_object":
        return AddObject(**base, object_type=_need(raw.object_type, "object_type", op),
                         label_zh=raw.label_zh or "", location=loc, sound_query=raw.sound_query)
    if op == "remove_object":
        return RemoveObject(**base, object_id=_need(raw.target_id, "target_id", op))
    if op == "move_object":
        if not (raw.landmark_id or raw.landmark_name or raw.direction or raw.relation):
            raise CommandError("move_object 缺少目标位置")
        return MoveObject(**base, object_id=_need(raw.target_id, "target_id", op), location=loc)
    if op == "replace_sound":
        return ReplaceSound(**base, target_id=_need(raw.target_id, "target_id", op),
                            sound_query=raw.sound_query, next_variant=bool(raw.next_variant))
    if op == "set_gain":
        return SetGain(**base, target_id=_need(raw.target_id, "target_id", op),
                       gain_change=_need(raw.gain_change, "gain_change", op))
    if op == "set_time":
        return SetTime(**base, time_of_day=_need(raw.time_of_day, "time_of_day", op))
    if op == "set_weather":
        return SetWeather(**base, weather=_need(raw.weather, "weather", op))
    if op == "set_season":
        return SetSeason(**base, season=_need(raw.season, "season", op))
    if op == "regenerate_viewpoint":
        return RegenerateViewpoint(**base, viewpoint_id=_need(raw.viewpoint_id or current_viewpoint, "viewpoint_id", op),
                                   description=_need(raw.description, "description", op))
    if op == "add_viewpoint":
        return AddViewpoint(**base, name_zh=_need(raw.landmark_name or raw.label_zh, "landmark_name", op),
                            description=raw.description or "", connect_to=raw.viewpoint_id or current_viewpoint)
    if op == "add_ambience":
        return AddAmbience(**base, sound_query=_need(raw.sound_query, "sound_query", op))
    if op == "remove_ambience":
        return RemoveAmbience(**base, ambience_id=_need(raw.target_id, "target_id", op))
    if op == "clarify":
        return Clarify(**base, question=_need(raw.confirm_question or raw.description, "confirm_question", op))
    raise CommandError(f"未知指令 {op}")
