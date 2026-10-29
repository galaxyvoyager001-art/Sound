"""场景清单校验、指令解析与执行、撤销。"""
import copy

import pytest
from pydantic import ValidationError

from app.db import Database
from app.scene.commands import CommandError, RawCommand, RawCommandList, to_typed
from app.scene.executor import ExecContext, execute
from app.scene.models import (Anchor, Landmark, SceneManifest, SceneObject, SoundSpec, Story, norm_yaw)
from app.scene.placement import resolve_placement, Offset
from app.scene.store import SceneStore, compute_diff
from app.providers.llm import strict_schema

from helpers import FakeResolver, make_scene


def raw(**kw) -> RawCommand:
    base = {k: None for k in RawCommand.model_fields}
    base.update({"confidence": 0.9})
    base.update(kw)
    return RawCommand(**base)


# ---------------- 清单校验 ----------------
def test_manifest_roundtrip_and_builtin_landmarks():
    m = make_scene()
    data = m.model_dump_json()
    m2 = SceneManifest.model_validate_json(data)
    assert m2 == m
    names = {l.name_zh for l in m.landmarks}
    assert {"正前方", "左边", "右边", "身后", "远处"} <= names


def test_manifest_rejects_dangling_landmark():
    m = make_scene()
    data = m.model_dump()
    data["objects"].append({"id": "obj_x", "type": "鸡笼", "landmark": "lm_missing"})
    with pytest.raises(ValidationError, match="地标"):
        SceneManifest.model_validate(data)


def test_manifest_rejects_duplicate_ids_and_bad_story_anchor():
    m = make_scene()
    d = m.model_dump()
    d["landmarks"].append(d["landmarks"][0])
    with pytest.raises(ValidationError, match="重复"):
        SceneManifest.model_validate(d)
    d = m.model_dump()
    d["stories"].append({"id": "st1", "transcript": "x", "anchor": {"kind": "object", "id": "nope"}})
    with pytest.raises(ValidationError, match="锚点"):
        SceneManifest.model_validate(d)


def test_sound_spec_limits():
    with pytest.raises(ValidationError):
        SoundSpec(clip_id="a", gain_db=20)
    with pytest.raises(ValidationError):
        SoundSpec(clip_id="a", interval_range_s=(10, 5))
    with pytest.raises(ValidationError):
        SceneManifest(scene_id="s", participant_id="p", weather="台风")


def test_norm_yaw():
    assert norm_yaw(190) == -170
    assert norm_yaw(-180) == 180
    assert norm_yaw(540) == 180


def test_strict_schema_all_required():
    js = strict_schema(RawCommandList)
    cmd = js["$defs"]["RawCommand"]
    assert cmd["additionalProperties"] is False
    assert set(cmd["required"]) == set(cmd["properties"])


# ---------------- 指令解析 ----------------
def test_to_typed_requires_fields():
    with pytest.raises(CommandError):
        to_typed(raw(op="remove_object"))
    with pytest.raises(CommandError):
        to_typed(raw(op="add_object"))
    with pytest.raises(CommandError):
        to_typed(raw(op="move_object", target_id="obj_1"))
    c = to_typed(raw(op="set_weather", weather="小雨"))
    assert c.op == "set_weather" and c.weather == "小雨"


def test_raw_command_rejects_unknown_op():
    with pytest.raises(ValidationError):
        raw(op="explode")


# ---------------- 执行 ----------------
async def _exec(m, rc, cur=None, added_by="edit"):
    ctx = ExecContext(resolver=FakeResolver(), current_viewpoint=cur or m.entry_viewpoint, added_by=added_by)
    return await execute(m, to_typed(rc, ctx.current_viewpoint), ctx)


async def test_add_object_with_new_landmark_and_sound():
    m = make_scene()
    r = await _exec(m, raw(op="add_object", object_type="馄饨担子", label_zh="阿福伯的馄饨担子",
                           landmark_name="巷口", direction="front", distance="far"))
    assert r.applied
    obj = m.objects[0]
    assert obj.sound and obj.sound.clip_id == "snd_bangzi" and obj.sound.mode == "event"
    lm = m.landmark(obj.landmark)
    assert lm.name_zh == "巷口" and lm.distance_m == 22.0
    assert obj.placement.distance_m == pytest.approx(22.0)
    assert any(j.kind == "object_asset" for j in r.jobs)
    SceneManifest.model_validate(m.model_dump())


async def test_add_object_unknown_sound_goes_unresolved():
    m = make_scene()
    r = await _exec(m, raw(op="add_object", object_type="爆米花机", sound_query="爆米花机砰的一声"))
    assert r.applied
    assert m.objects[0].sound is None
    assert m.unresolved and m.unresolved[0].kind == "sound"
    assert any(j.kind == "fetch_sound" for j in r.jobs)


async def test_move_object_farther_and_remove():
    m = make_scene()
    await _exec(m, raw(op="add_object", object_type="馄饨担子", landmark_name="巷口", direction="front"))
    oid = m.objects[0].id
    d0 = m.objects[0].placement.distance_m
    r = await _exec(m, raw(op="move_object", target_id=oid, landmark_name="巷口", relation="farther"))
    assert r.applied and m.objects[0].placement.distance_m > d0
    r = await _exec(m, raw(op="remove_object", target_id=oid))
    assert r.applied and not m.objects


async def test_remove_object_keeps_story_on_landmark():
    m = make_scene()
    await _exec(m, raw(op="add_object", object_type="鸡笼"))
    o = m.objects[0]
    m.stories.append(Story(id="st1", transcript="小时候养过鸡", anchor=Anchor(kind="object", id=o.id)))
    await _exec(m, raw(op="remove_object", target_id=o.id))
    assert m.stories[0].anchor.kind == "landmark" and m.stories[0].anchor.id == o.landmark
    SceneManifest.model_validate(m.model_dump())


async def test_set_gain_and_variant_and_weather():
    m = make_scene()
    await _exec(m, raw(op="add_object", object_type="馄饨担子"))
    oid = m.objects[0].id
    await _exec(m, raw(op="set_gain", target_id=oid, gain_change="quieter"))
    assert m.objects[0].sound.gain_db == -8
    for _ in range(10):
        await _exec(m, raw(op="set_gain", target_id=oid, gain_change="much_louder"))
    assert m.objects[0].sound.gain_db == 6  # 上限
    await _exec(m, raw(op="replace_sound", target_id=oid, next_variant=True))
    assert m.objects[0].sound.variant_index == 1
    await _exec(m, raw(op="set_weather", weather="小雨"))
    assert m.weather == "小雨" and any(a.clip_id == "snd_rain_tiles" for a in m.ambience)


async def test_add_viewpoint_links_landmark():
    m = make_scene()
    await _exec(m, raw(op="add_object", object_type="水井", landmark_name="井边", direction="left"))
    r = await _exec(m, raw(op="add_viewpoint", landmark_name="井边", description="青石井栏"))
    assert r.applied and r.jobs[0].kind == "panorama"
    new_vp = m.find_viewpoint_by_name("井边")
    lm = m.find_landmark_by_name("井边", m.entry_viewpoint)
    assert lm.linked_viewpoint_id == new_vp.id
    assert new_vp.id in m.viewpoint(m.entry_viewpoint).neighbors


async def test_clarify_not_applied():
    m = make_scene()
    r = await _exec(m, raw(op="clarify", confirm_question="是说卖馄饨的在更远的巷口吗？"))
    assert not r.applied and r.question


async def test_remove_missing_object_raises():
    m = make_scene()
    with pytest.raises(CommandError):
        await _exec(m, raw(op="remove_object", target_id="obj_nope"))


def test_depth_clamps_placement():
    m = make_scene()
    lm = m.find_landmark_by_name("远处")

    class Wall:
        def depth_at(self, vp, yaw, pitch):
            return 6.0  # 前方 6 米有墙

    p = resolve_placement(m, lm, Offset(), Wall())
    assert p.distance_m <= 6.0


# ---------------- 版本与撤销 ----------------
async def test_store_commit_and_undo_restores_only_edit():
    db = Database(":memory:")
    store = SceneStore(db)
    m = store.create(make_scene())

    before = m
    after = copy.deepcopy(m)
    await _exec(after, raw(op="add_object", object_type="馄饨担子", landmark_name="巷口"))
    v2 = store.commit(before, after, {"kind": "edit", "op": "add_object"})
    assert v2.version == 2 and len(v2.objects) == 1
    oid = v2.objects[0].id

    # 后台任务：插画完成（system 版本，不可撤销）
    sys_after = copy.deepcopy(v2)
    sys_after.weather = "阴"  # 同时有一个无关的系统变化
    v3 = store.commit(v2, sys_after, {"kind": "system"})

    # 一次移动
    mv = copy.deepcopy(v3)
    await _exec(mv, raw(op="move_object", target_id=oid, relation="farther", landmark_name="巷口"))
    v4 = store.commit(v3, mv, {"kind": "edit", "op": "move_object"})
    assert v4.objects[0].placement.distance_m > v3.objects[0].placement.distance_m

    # 撤销移动 → 回到 v3 的位置，系统变化保留
    v5 = store.undo_last(v4.scene_id)
    assert v5.version == 5
    assert v5.objects[0].placement == v3.objects[0].placement
    assert v5.weather == "阴"
    # 再撤销 → 去掉物件
    v6 = store.undo_last(v4.scene_id)
    assert not v6.objects and v6.weather == "阴"
    # 没有可撤销的了
    assert store.undo_last(v4.scene_id) is None
    hist = store.history(v4.scene_id)
    assert [h["cause"]["kind"] for h in hist] == ["create", "edit", "system", "edit", "undo", "undo"]


def test_commit_rejects_stale_version_and_noop():
    db = Database(":memory:")
    store = SceneStore(db)
    m = store.create(make_scene())
    same = copy.deepcopy(m)
    assert store.commit(m, same, {"kind": "edit"}) is None
    a = copy.deepcopy(m)
    a.weather = "雪"
    store.commit(m, a, {"kind": "edit"})
    b = copy.deepcopy(m)
    b.weather = "雾"
    with pytest.raises(RuntimeError):
        store.commit(m, b, {"kind": "edit"})


def test_commit_rejects_invalid_manifest():
    db = Database(":memory:")
    store = SceneStore(db)
    m = store.create(make_scene())
    bad = copy.deepcopy(m)
    bad.objects.append(SceneObject.model_construct(id="o1", type="x", landmark="lm_missing",
                                                   offset=Offset(), placement=None, height_m=1.0, sound=None,
                                                   added_by="edit", label_zh="", asset_id=None,
                                                   model_asset_id=None, asset_status="pending", created_at=0))
    with pytest.raises(ValidationError):
        store.commit(m, bad, {"kind": "edit"})


def test_diff_detects_changes():
    a = make_scene()
    b = copy.deepcopy(a)
    b.landmarks.append(Landmark(id="lm_new", name_zh="井边", viewpoint_id=a.entry_viewpoint, yaw_deg=10, distance_m=5))
    b.season = "冬"
    d = compute_diff(a, b)
    assert "lm_new" in d["collections"]["landmarks"]
    assert d["fields"]["season"] == {"before": "夏", "after": "冬"}
