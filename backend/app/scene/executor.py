"""执行修改指令：只改场景清单，耗时的生成任务以 Job 形式返回给编排器异步执行。"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .commands import (
    AddAmbience, AddObject, AddViewpoint, Clarify, CommandError, EditCommand, LocationRef,
    MoveObject, RegenerateViewpoint, RemoveAmbience, RemoveObject, ReplaceSound, SetGain,
    SetSeason, SetTime, SetWeather,
)
from .models import (
    AddedBy, AmbienceLayer, Landmark, SceneManifest, SceneObject, SoundSpec, UnresolvedItem,
    Viewpoint, new_id,
)
from .placement import (
    BUILTIN_LANDMARKS, DISTANCE_M, DIRECTION_YAW, DepthSampler, NoDepth, offset_for,
    resolve_placement,
)

GAIN_STEPS = {"louder": 4.0, "quieter": -5.0, "much_louder": 8.0, "much_quieter": -10.0}


@dataclass
class SoundMatch:
    clip_id: str
    name_zh: str = ""
    mode: str = "loop"
    interval_range_s: tuple[float, float] = (8.0, 25.0)
    acoustic_profile: str = "clean"
    gain_db: float = 0.0
    variants: int = 1
    score: float = 1.0


@dataclass
class ObjectMeta:
    height_m: float = 1.2
    elevation_m: float = 0.0
    sound_query: Optional[str] = None
    has_sound: bool = False


class AssetResolver(Protocol):
    async def find_sound(self, query: str, manifest: SceneManifest, purpose: str = "object",
                         exclude: tuple[str, ...] = ()) -> Optional[SoundMatch]: ...

    def object_meta(self, object_type: str) -> ObjectMeta: ...

    def sound_variants(self, clip_id: str) -> int: ...

    def auto_ambience(self, manifest: SceneManifest) -> list[AmbienceLayer]: ...


@dataclass
class Job:
    kind: str  # object_asset | panorama | fetch_sound
    target_id: str
    params: dict = field(default_factory=dict)


@dataclass
class ExecContext:
    resolver: AssetResolver
    depth: DepthSampler = field(default_factory=NoDepth)
    current_viewpoint: Optional[str] = None
    added_by: AddedBy = "edit"
    rng: random.Random = field(default_factory=random.Random)


@dataclass
class ExecResult:
    applied: bool
    message: str = ""
    changed_ids: list[str] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    question: Optional[str] = None  # clarify 时由陪聊者说出


def add_builtin_landmarks(manifest: SceneManifest, viewpoint_id: str) -> None:
    for name, aliases, yaw, dist in BUILTIN_LANDMARKS:
        if manifest.find_landmark_by_name(name, viewpoint_id) and \
                manifest.find_landmark_by_name(name, viewpoint_id).builtin:
            continue
        manifest.landmarks.append(Landmark(id=new_id("lm"), name_zh=name, aliases=list(aliases),
                                           viewpoint_id=viewpoint_id, yaw_deg=yaw, distance_m=dist,
                                           builtin=True))


def create_viewpoint(manifest: SceneManifest, name_zh: str, description: str = "",
                     connect_to: Optional[str] = None) -> Viewpoint:
    vp = Viewpoint(id=new_id("vp"), name_zh=name_zh, description_zh=description)
    manifest.viewpoints.append(vp)
    add_builtin_landmarks(manifest, vp.id)
    if connect_to and manifest.viewpoint(connect_to):
        other = manifest.viewpoint(connect_to)
        other.neighbors = list(dict.fromkeys(other.neighbors + [vp.id]))
        vp.neighbors = [connect_to]
        # 原点位里如果有同名地标，链接过去，“去井边看看”就能走过去
        lm = manifest.find_landmark_by_name(name_zh, connect_to)
        if lm and not lm.builtin:
            lm.linked_viewpoint_id = vp.id
    if not manifest.entry_viewpoint:
        manifest.entry_viewpoint = vp.id
    return vp


def _resolve_landmark(manifest: SceneManifest, loc: LocationRef, current_vp: Optional[str]) -> Landmark:
    vp_id = loc.viewpoint_id or current_vp or manifest.entry_viewpoint
    if not vp_id or not manifest.viewpoint(vp_id):
        raise CommandError("还没有可以放物件的点位")
    if loc.landmark_id:
        lm = manifest.landmark(loc.landmark_id)
        if not lm:
            raise CommandError(f"地标 {loc.landmark_id} 不存在")
        return lm
    if loc.landmark_name:
        lm = manifest.find_landmark_by_name(loc.landmark_name, vp_id) or manifest.find_landmark_by_name(loc.landmark_name)
        if lm:
            return lm
        # 新地标：按方向和远近放置；没给方向时默认正前方偏一点
        yaw = DIRECTION_YAW.get(loc.direction or "front", 0.0)
        dist = DISTANCE_M.get(loc.distance or "mid", 9.0)
        lm = Landmark(id=new_id("lm"), name_zh=loc.landmark_name.strip(), viewpoint_id=vp_id,
                      yaw_deg=yaw, distance_m=dist)
        manifest.landmarks.append(lm)
        return lm
    if loc.direction:
        # 只有方向：映射到预置方向地标
        names = {"front": "正前方", "left": "左边", "right": "右边", "back": "身后",
                 "front_left": "左边", "front_right": "右边", "back_left": "身后", "back_right": "身后"}
        if loc.distance == "far" and loc.direction == "front":
            names["front"] = "远处"
        lm = manifest.find_landmark_by_name(names[loc.direction], vp_id)
        if lm:
            return lm
    lm = manifest.find_landmark_by_name("正前方", vp_id)
    if not lm:
        add_builtin_landmarks(manifest, vp_id)
        lm = manifest.find_landmark_by_name("正前方", vp_id)
    return lm  # type: ignore[return-value]


def _sound_spec(match: SoundMatch, prev: Optional[SoundSpec] = None) -> SoundSpec:
    return SoundSpec(
        clip_id=match.clip_id, variant_index=0,
        gain_db=prev.gain_db if prev else match.gain_db,
        mode=match.mode if match.mode in ("loop", "event") else "loop",  # type: ignore[arg-type]
        interval_range_s=match.interval_range_s,
        acoustic_profile=match.acoustic_profile if match.acoustic_profile in
        ("clean", "old_radio", "loudspeaker_horn", "gramophone") else "clean",  # type: ignore[arg-type]
    )


def _add_unresolved(manifest: SceneManifest, mention: str, kind: str, reason: str) -> UnresolvedItem:
    for u in manifest.unresolved:
        if u.mention == mention and u.kind == kind and u.status in ("open", "searching"):
            return u
    item = UnresolvedItem(id=new_id("un"), mention=mention, kind=kind, reason=reason)  # type: ignore[arg-type]
    manifest.unresolved.append(item)
    return item


def refresh_auto_ambience(manifest: SceneManifest, resolver: AssetResolver) -> list[str]:
    manual = [a for a in manifest.ambience if not a.auto]
    auto = resolver.auto_ambience(manifest)
    manifest.ambience = manual + auto
    return [a.id for a in auto]


async def execute(manifest: SceneManifest, cmd: EditCommand, ctx: ExecContext) -> ExecResult:
    """在 manifest 上原地执行一条指令。调用方负责先深拷贝、执行后提交版本。"""
    m = manifest
    if isinstance(cmd, Clarify):
        return ExecResult(applied=False, message="需要确认", question=cmd.question)

    if isinstance(cmd, AddObject):
        lm = _resolve_landmark(m, cmd.location, ctx.current_viewpoint)
        meta = ctx.resolver.object_meta(cmd.object_type)
        obj = SceneObject(id=new_id("obj"), type=cmd.object_type.strip(), label_zh=cmd.label_zh,
                          landmark=lm.id, offset=offset_for(cmd.location.relation, ctx.rng),
                          height_m=meta.height_m, added_by=ctx.added_by, asset_status="pending")
        obj.placement = resolve_placement(m, lm, obj.offset, ctx.depth, meta.elevation_m)
        jobs = [Job("object_asset", obj.id)]
        query = cmd.sound_query or (meta.sound_query if meta.has_sound else None)
        if query:
            match = await ctx.resolver.find_sound(query, m, "object")
            if match:
                obj.sound = _sound_spec(match)
            elif cmd.sound_query:
                u = _add_unresolved(m, cmd.sound_query, "sound", f"没有合适的“{cmd.sound_query}”声音素材")
                jobs.append(Job("fetch_sound", u.id, {"query": cmd.sound_query, "object_id": obj.id}))
        m.objects.append(obj)
        return ExecResult(True, f"加入了{obj.label_zh or obj.type}", [obj.id], jobs)

    if isinstance(cmd, RemoveObject):
        obj = m.obj(cmd.object_id)
        if not obj:
            raise CommandError(f"物件 {cmd.object_id} 不存在")
        # 挂在这个物件上的故事改挂到它所在的地标，故事不丢
        for s in m.stories:
            if s.anchor.kind == "object" and s.anchor.id == obj.id:
                s.anchor = s.anchor.model_copy(update={"kind": "landmark", "id": obj.landmark})
        m.objects = [o for o in m.objects if o.id != obj.id]
        return ExecResult(True, f"去掉了{obj.label_zh or obj.type}", [obj.id])

    if isinstance(cmd, MoveObject):
        obj = m.obj(cmd.object_id)
        if not obj:
            raise CommandError(f"物件 {cmd.object_id} 不存在")
        lm = _resolve_landmark(m, cmd.location, obj.placement.viewpoint_id if obj.placement else ctx.current_viewpoint)
        obj.landmark = lm.id
        obj.offset = offset_for(cmd.location.relation, ctx.rng)
        elev = obj.placement.elevation_m if obj.placement else 0.0
        obj.placement = resolve_placement(m, lm, obj.offset, ctx.depth, elev, avoid_ids=(obj.id,))
        return ExecResult(True, f"把{obj.label_zh or obj.type}挪到了{lm.name_zh}", [obj.id])

    if isinstance(cmd, ReplaceSound):
        target_obj = m.obj(cmd.target_id)
        target_amb = m.ambience_layer(cmd.target_id)
        if not target_obj and not target_amb:
            raise CommandError(f"{cmd.target_id} 不存在")
        if target_obj:
            if cmd.next_variant and target_obj.sound:
                n = max(1, ctx.resolver.sound_variants(target_obj.sound.clip_id))
                if n > 1:
                    target_obj.sound = target_obj.sound.model_copy(
                        update={"variant_index": (target_obj.sound.variant_index + 1) % n})
                    return ExecResult(True, "换了一个声音", [target_obj.id])
            query = cmd.sound_query or target_obj.type
            exclude = (target_obj.sound.clip_id,) if target_obj.sound else ()
            match = await ctx.resolver.find_sound(query, m, "object", exclude=exclude)
            if not match:
                u = _add_unresolved(m, query, "sound", "暂时没有更合适的声音")
                return ExecResult(False, "暂时没有更合适的声音", [], [Job("fetch_sound", u.id,
                                  {"query": query, "object_id": target_obj.id})])
            target_obj.sound = _sound_spec(match, target_obj.sound)
            return ExecResult(True, "换了一个声音", [target_obj.id])
        assert target_amb
        exclude = (target_amb.clip_id,)
        match = await ctx.resolver.find_sound(cmd.sound_query or target_amb.name_zh or target_amb.clip_id,
                                              m, "ambience", exclude=exclude)
        if not match:
            return ExecResult(False, "暂时没有更合适的环境声")
        target_amb.clip_id = match.clip_id
        target_amb.name_zh = match.name_zh
        target_amb.auto = False
        target_amb.role = "manual"
        return ExecResult(True, "换了环境声", [target_amb.id])

    if isinstance(cmd, SetGain):
        targets: list = []
        if cmd.target_id == "all":
            targets = [o for o in m.objects if o.sound] + list(m.ambience)
        else:
            t = m.obj(cmd.target_id) or m.ambience_layer(cmd.target_id)
            if not t:
                raise CommandError(f"{cmd.target_id} 不存在")
            targets = [t]
        changed = []
        for t in targets:
            if isinstance(t, SceneObject):
                if not t.sound:
                    continue
                g = -40.0 if cmd.gain_change == "mute" else t.sound.gain_db + GAIN_STEPS[cmd.gain_change]
                t.sound = t.sound.model_copy(update={"gain_db": max(-40.0, min(6.0, g))})
            else:
                g = -40.0 if cmd.gain_change == "mute" else t.gain_db + GAIN_STEPS[cmd.gain_change]
                t.gain_db = max(-40.0, min(6.0, g))
            changed.append(t.id)
        if not changed:
            raise CommandError("这个东西没有声音")
        return ExecResult(True, "调了音量", changed)

    if isinstance(cmd, (SetTime, SetWeather, SetSeason)):
        if isinstance(cmd, SetTime):
            m.time_of_day = cmd.time_of_day
        elif isinstance(cmd, SetWeather):
            m.weather = cmd.weather
        else:
            m.season = cmd.season
        changed = refresh_auto_ambience(m, ctx.resolver)
        return ExecResult(True, "换了时间/天气", ["scene"] + changed)

    if isinstance(cmd, RegenerateViewpoint):
        vp = m.viewpoint(cmd.viewpoint_id)
        if not vp:
            raise CommandError(f"点位 {cmd.viewpoint_id} 不存在")
        vp.description_zh = (vp.description_zh + "；" + cmd.description).strip("；")[-600:]
        # 画面在生成期间保持不变，生成完成后淡入；状态不置为 pending 以免前端清空
        return ExecResult(True, f"重新画{vp.name_zh}", [vp.id], [Job("panorama", vp.id, {"mode": "regenerate"})])

    if isinstance(cmd, AddViewpoint):
        existing = m.find_viewpoint_by_name(cmd.name_zh)
        if existing:
            return ExecResult(False, f"{existing.name_zh}已经有了", [existing.id])
        connect = cmd.connect_to if cmd.connect_to and m.viewpoint(cmd.connect_to) else (
            ctx.current_viewpoint or m.entry_viewpoint)
        vp = create_viewpoint(m, cmd.name_zh, cmd.description, connect)
        return ExecResult(True, f"添加了{vp.name_zh}", [vp.id], [Job("panorama", vp.id, {"mode": "new"})])

    if isinstance(cmd, AddAmbience):
        match = await ctx.resolver.find_sound(cmd.sound_query, m, "ambience")
        if not match:
            u = _add_unresolved(m, cmd.sound_query, "ambience", "没有合适的环境声素材")
            return ExecResult(False, "暂时没有合适的环境声", [], [Job("fetch_sound", u.id, {"query": cmd.sound_query})])
        if any(a.clip_id == match.clip_id for a in m.ambience):
            return ExecResult(False, "这个声音已经有了")
        layer = AmbienceLayer(id=new_id("amb"), clip_id=match.clip_id, name_zh=match.name_zh or cmd.sound_query,
                              gain_db=match.gain_db if match.gain_db < 0 else -10.0, role="manual")
        m.ambience.append(layer)
        return ExecResult(True, f"加了{layer.name_zh}", [layer.id])

    if isinstance(cmd, RemoveAmbience):
        layer = m.ambience_layer(cmd.ambience_id)
        if not layer:
            raise CommandError(f"环境声 {cmd.ambience_id} 不存在")
        m.ambience = [a for a in m.ambience if a.id != layer.id]
        return ExecResult(True, f"去掉了{layer.name_zh}", [layer.id])

    raise CommandError(f"不支持的指令 {type(cmd).__name__}")
