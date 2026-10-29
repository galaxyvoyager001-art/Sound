"""场景版本存储：每次修改都提交一个新版本，记录实体级差异用于撤销和研究分析。

撤销不是简单地回到上一个版本——因为在修改之后，后台生成任务（插画完成、全景完成）
也会提交版本。撤销时只把“那一次修改”涉及的实体恢复成修改前的样子，其他后续变化保留。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from ..db import Database
from .models import Anchor, SceneManifest, now_ts

COLLECTIONS = ("viewpoints", "landmarks", "objects", "ambience", "unresolved", "stories")
FIELDS = ("region_style", "era", "time_of_day", "weather", "season", "entry_viewpoint", "condition")
UNDOABLE_KINDS = ("edit", "narration")


def compute_diff(before: SceneManifest, after: SceneManifest) -> dict[str, Any]:
    b, a = before.model_dump(mode="json"), after.model_dump(mode="json")
    diff: dict[str, Any] = {"collections": {}, "fields": {}}
    for coll in COLLECTIONS:
        bm = {x["id"]: x for x in b[coll]}
        am = {x["id"]: x for x in a[coll]}
        changes = {}
        for i in set(bm) | set(am):
            if bm.get(i) != am.get(i):
                changes[i] = {"before": bm.get(i), "after": am.get(i)}
        if changes:
            diff["collections"][coll] = changes
    for f in FIELDS:
        if b[f] != a[f]:
            diff["fields"][f] = {"before": b[f], "after": a[f]}
    return diff


def diff_is_empty(diff: dict) -> bool:
    return not diff["collections"] and not diff["fields"]


def apply_inverse(current: SceneManifest, diff: dict[str, Any]) -> SceneManifest:
    data = current.model_dump(mode="json")
    for coll, changes in diff["collections"].items():
        items = data[coll]
        index = {x["id"]: n for n, x in enumerate(items)}
        for i, ch in changes.items():
            if ch["before"] is None:
                items[:] = [x for x in items if x["id"] != i]
                index = {x["id"]: n for n, x in enumerate(items)}
            elif i in index:
                items[index[i]] = ch["before"]
            else:
                items.append(ch["before"])
                index[i] = len(items) - 1
    for f, ch in diff["fields"].items():
        data[f] = ch["before"]
    _repair_refs(data)
    return SceneManifest.model_validate(data)


def _repair_refs(data: dict) -> None:
    """撤销后清理悬空引用（例如故事挂在一个被撤销掉的物件上）。"""
    vps = {v["id"] for v in data["viewpoints"]}
    lms = {l["id"]: l for l in data["landmarks"] if l["viewpoint_id"] in vps}
    data["landmarks"] = list(lms.values())
    for l in data["landmarks"]:
        if l.get("linked_viewpoint_id") and l["linked_viewpoint_id"] not in vps:
            l["linked_viewpoint_id"] = None
    for v in data["viewpoints"]:
        v["neighbors"] = [n for n in v["neighbors"] if n in vps]
    data["objects"] = [o for o in data["objects"] if o["landmark"] in lms]
    objs = {o["id"]: o for o in data["objects"]}
    if data.get("entry_viewpoint") and data["entry_viewpoint"] not in vps:
        data["entry_viewpoint"] = next(iter(vps), None)
    fallback_vp = data.get("entry_viewpoint")
    kept = []
    for s in data["stories"]:
        a = s["anchor"]
        ok = (a["kind"] == "object" and a["id"] in objs) or (a["kind"] == "landmark" and a["id"] in lms) \
            or (a["kind"] == "viewpoint" and a["id"] in vps)
        if not ok:
            if fallback_vp:
                s["anchor"] = Anchor(kind="viewpoint", id=fallback_vp).model_dump()
            else:
                continue
        kept.append(s)
    data["stories"] = kept


class SceneStore:
    def __init__(self, db: Database):
        self.db = db
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, scene_id: str) -> asyncio.Lock:
        """同一场景的修改串行执行（会谈中的指令和后台生成任务可能并发）。"""
        if scene_id not in self._locks:
            self._locks[scene_id] = asyncio.Lock()
        return self._locks[scene_id]

    def create(self, manifest: SceneManifest, cause: Optional[dict] = None) -> SceneManifest:
        manifest = manifest.model_copy(update={"version": 1, "updated_at": now_ts()})
        SceneManifest.model_validate(manifest.model_dump())
        self.db.execute(
            "INSERT INTO scene_versions (scene_id, version, manifest, cause, diff, created_at) VALUES (?,?,?,?,?,?)",
            (manifest.scene_id, 1, manifest.model_dump_json(), json.dumps(cause or {"kind": "create"}, ensure_ascii=False),
             None, now_ts()))
        return manifest

    def get(self, scene_id: str, version: Optional[int] = None) -> Optional[SceneManifest]:
        if version is None:
            row = self.db.one("SELECT manifest FROM scene_versions WHERE scene_id=? ORDER BY version DESC LIMIT 1",
                              (scene_id,))
        else:
            row = self.db.one("SELECT manifest FROM scene_versions WHERE scene_id=? AND version=?", (scene_id, version))
        return SceneManifest.model_validate_json(row["manifest"]) if row else None

    def commit(self, before: SceneManifest, after: SceneManifest, cause: dict) -> Optional[SceneManifest]:
        """提交新版本；没有任何变化时返回 None。after 会被重新校验。"""
        latest = self.get(before.scene_id)
        if latest is None:
            raise KeyError(before.scene_id)
        if latest.version != before.version:
            raise RuntimeError("场景版本冲突：请在 store.lock(scene_id) 内读取-修改-提交")
        validated = SceneManifest.model_validate(after.model_dump())
        diff = compute_diff(before, validated)
        if diff_is_empty(diff):
            return None
        validated = validated.model_copy(update={"version": before.version + 1, "updated_at": now_ts()})
        self.db.execute(
            "INSERT INTO scene_versions (scene_id, version, manifest, cause, diff, created_at) VALUES (?,?,?,?,?,?)",
            (validated.scene_id, validated.version, validated.model_dump_json(),
             json.dumps(cause, ensure_ascii=False), json.dumps(diff, ensure_ascii=False), now_ts()))
        return validated

    def undo_last(self, scene_id: str, kinds: tuple[str, ...] = UNDOABLE_KINDS) -> Optional[SceneManifest]:
        """撤销最近一次（尚未撤销的）用户修改。没有可撤销的修改时返回 None。"""
        rows = self.db.query(
            "SELECT version, cause, diff FROM scene_versions WHERE scene_id=? AND undone=0 AND diff IS NOT NULL "
            "ORDER BY version DESC", (scene_id,))
        target = None
        for r in rows:
            cause = json.loads(r["cause"])
            if cause.get("kind") in kinds:
                target = (r, cause)
                break
        if not target:
            return None
        row, cause = target
        current = self.get(scene_id)
        assert current is not None
        reverted = apply_inverse(current, json.loads(row["diff"]))
        new = self.commit(current, reverted, {"kind": "undo", "undoes_version": row["version"],
                                              "undoes": cause})
        self.db.execute("UPDATE scene_versions SET undone=1 WHERE scene_id=? AND version=?",
                        (scene_id, row["version"]))
        return new or current

    def history(self, scene_id: str) -> list[dict]:
        rows = self.db.query("SELECT version, cause, undone, created_at, diff FROM scene_versions "
                             "WHERE scene_id=? ORDER BY version", (scene_id,))
        out = []
        for r in rows:
            out.append({"version": r["version"], "cause": json.loads(r["cause"]), "undone": bool(r["undone"]),
                        "created_at": r["created_at"], "diff": json.loads(r["diff"]) if r["diff"] else None})
        return out

    def delete_scene(self, scene_id: str) -> None:
        self.db.execute("DELETE FROM scene_versions WHERE scene_id=?", (scene_id,))
