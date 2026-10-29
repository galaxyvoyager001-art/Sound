"""场景服务：加锁执行指令并提交版本、运行后台生成任务、组装前端素材表、创建各实验条件的场景。"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from ..assets.catalog import Catalog, SoundVocab
from ..assets.online import SoundFetcher
from ..assets.registry import AssetRegistry
from ..assets.retrieval import AssetResolverImpl
from ..config import Settings, to_abs
from ..db import Database
from ..generation.service import GenerationService, slug
from ..providers import Providers
from ..providers.depth import DepthMapSampler
from .commands import AddObject, CommandError, EditCommand, LocationRef
from .executor import ExecContext, ExecResult, Job, create_viewpoint, execute, refresh_auto_ambience
from .models import AddedBy, Condition, SceneManifest, new_id
from .placement import resolve_placement
from .store import SceneStore

log = logging.getLogger("scene")
Listener = Callable[[SceneManifest, list[str]], Awaitable[None]]


class SceneService:
    def __init__(self, s: Settings, store: SceneStore, registry: AssetRegistry, catalog: Catalog,
                 resolver: AssetResolverImpl, depth: DepthMapSampler, generation: GenerationService,
                 fetcher: SoundFetcher, providers: Providers, db: Database):
        self.s = s
        self.store = store
        self.registry = registry
        self.catalog = catalog
        self.resolver = resolver
        self.depth = depth
        self.generation = generation
        self.fetcher = fetcher
        self.providers = providers
        self.db = db
        self.listeners: dict[str, list[Listener]] = {}
        self.tasks: set[asyncio.Task] = set()

    # ---------------- 订阅 ----------------
    def subscribe(self, scene_id: str, fn: Listener) -> None:
        self.listeners.setdefault(scene_id, []).append(fn)

    def unsubscribe(self, scene_id: str, fn: Listener) -> None:
        if fn in self.listeners.get(scene_id, []):
            self.listeners[scene_id].remove(fn)

    async def notify(self, m: SceneManifest, highlight: list[str]) -> None:
        for fn in list(self.listeners.get(m.scene_id, [])):
            try:
                await fn(m, highlight)
            except Exception as e:
                log.warning("推送场景失败：%s", e)

    def load_depths(self, m: SceneManifest) -> None:
        for vp in m.viewpoints:
            if vp.depth_path and vp.id not in self.depth.maps:
                self.depth.load_png(vp.id, to_abs(vp.depth_path))

    # ---------------- 修改 ----------------
    async def apply(self, scene_id: str, cmds: list[EditCommand], cause: dict, current_vp: Optional[str],
                    added_by: AddedBy = "edit", session_id: Optional[str] = None
                    ) -> tuple[SceneManifest, list[tuple[EditCommand, ExecResult | None, str]]]:
        """执行一组指令（同一版本），返回新清单和每条指令的结果。"""
        results: list[tuple[EditCommand, ExecResult | None, str]] = []
        jobs: list[Job] = []
        highlight: list[str] = []
        async with self.store.lock(scene_id):
            before = self.store.get(scene_id)
            assert before is not None
            self.load_depths(before)
            work = copy.deepcopy(before)
            ctx = ExecContext(resolver=self.resolver, depth=self.depth, current_viewpoint=current_vp, added_by=added_by)
            for cmd in cmds:
                snapshot = copy.deepcopy(work)
                try:
                    r = await execute(work, cmd, ctx)
                    results.append((cmd, r, ""))
                    jobs += r.jobs
                    if r.applied:
                        highlight += r.changed_ids
                except (CommandError, ValueError) as e:
                    work = snapshot
                    results.append((cmd, None, str(e)))
            try:
                new = self.store.commit(before, work, {**cause, "ops": [c.op for c in cmds]})
            except Exception as e:  # 校验失败：整组不提交
                log.warning("指令执行后清单校验失败：%s", e)
                new = None
                results = [(c, None, f"校验失败：{e}") for c, _, _ in results]
                jobs = []
        m = new or before
        if new:
            await self.notify(m, highlight)
        for j in jobs:
            self.spawn(self.run_job(scene_id, j, session_id))
        return m, results

    async def system_update(self, scene_id: str, fn: Callable[[SceneManifest], Optional[list[str]]], cause: dict
                            ) -> Optional[SceneManifest]:
        async with self.store.lock(scene_id):
            before = self.store.get(scene_id)
            if before is None:
                return None
            work = copy.deepcopy(before)
            highlight = fn(work) or []
            try:
                new = self.store.commit(before, work, cause)
            except Exception as e:
                log.warning("系统更新失败：%s", e)
                return None
        if new:
            await self.notify(new, highlight)
        return new

    async def undo(self, scene_id: str) -> Optional[SceneManifest]:
        async with self.store.lock(scene_id):
            new = self.store.undo_last(scene_id)
        if new:
            await self.notify(new, [])
        return new

    # ---------------- 后台任务 ----------------
    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def run_job(self, scene_id: str, job: Job, session_id: Optional[str] = None) -> None:
        try:
            if job.kind == "object_asset":
                await self._job_object(scene_id, job.target_id)
            elif job.kind == "panorama":
                await self._job_panorama(scene_id, job.target_id, job.params.get("mode", "new"))
            elif job.kind == "fetch_sound":
                await self._job_fetch_sound(scene_id, job.target_id, job.params)
            self.db.log_event(session_id, "job_done", {"kind": job.kind, "target": job.target_id})
        except Exception as e:
            log.exception("后台任务失败 %s", job)
            self.db.log_event(session_id, "job_failed", {"kind": job.kind, "target": job.target_id, "error": str(e)[:300]})

    async def _job_object(self, scene_id: str, object_id: str) -> None:
        m = self.store.get(scene_id)
        obj = m.obj(object_id) if m else None
        if not obj:
            return
        if m.condition == "audio_only":
            return
        rec = await self.generation.object_asset(obj.type, m.region_style, m.era)
        model = await self.generation.object_model3d(rec) if rec else None

        def upd(w: SceneManifest):
            o = w.obj(object_id)
            if not o:
                return []
            if rec:
                o.asset_id = rec.id
                o.asset_status = "ready"
                if model:
                    o.model_asset_id = model.id
            else:
                o.asset_status = "failed"
            return [o.id]

        await self.system_update(scene_id, upd, {"kind": "system", "job": "object_asset", "object": object_id})

    async def _job_panorama(self, scene_id: str, vp_id: str, mode: str) -> None:
        m = self.store.get(scene_id)
        vp = m.viewpoint(vp_id) if m else None
        if not vp:
            return

        def mark(w: SceneManifest):
            v = w.viewpoint(vp_id)
            if v and v.status != "ready":
                v.status = "generating"
        await self.system_update(scene_id, mark, {"kind": "system", "job": "panorama_start", "viewpoint": vp_id})

        if m.condition == "audio_only":
            pano, depth = self.generation.audio_only_background()
            res = type("R", (), {"panorama_path": pano, "depth_path": depth, "asset_id": None, "prompt": ""})()
        else:
            res = await self.generation.generate_panorama(m, vp, mode)
        if res is None:  # 重绘失败：保持原画面
            await self.system_update(scene_id, lambda w: setattr(w.viewpoint(vp_id), "status", "ready") if w.viewpoint(vp_id) else None,
                                     {"kind": "system", "job": "panorama_failed", "viewpoint": vp_id})
            return
        self.depth.load_png(vp_id, to_abs(res.depth_path))

        def done(w: SceneManifest):
            v = w.viewpoint(vp_id)
            if not v:
                return []
            v.panorama_path, v.depth_path, v.panorama_asset_id = res.panorama_path, res.depth_path, res.asset_id
            v.generation_prompt = res.prompt
            v.status = "ready"
            v.generated_for = {"season": w.season, "weather": w.weather, "time_of_day": w.time_of_day}
            # 有了深度图，重新计算这个点位里物件的位置（避免穿墙）
            for o in w.objects:
                lm = w.landmark(o.landmark)
                if lm and lm.viewpoint_id == vp_id:
                    elev = o.placement.elevation_m if o.placement else 0.0
                    o.placement = resolve_placement(w, lm, o.offset, self.depth, elev, avoid_ids=(o.id,))
            return [vp_id]

        await self.system_update(scene_id, done, {"kind": "system", "job": "panorama", "viewpoint": vp_id})

    async def _job_fetch_sound(self, scene_id: str, unresolved_id: str, params: dict) -> None:
        """没有合适声音：在线检索/生成 → 入库 → 再检索并挂到物件或环境声上。"""
        query = params.get("query", "")
        from ..conversation.schemas import SoundKeywords
        from ..conversation.prompts import keywords_messages
        try:
            kw = await self.providers.llm.structured(keywords_messages(query), SoundKeywords,
                                                     model=self.providers.llm_fast_model)
        except Exception:
            kw = SoundKeywords(en=[], zh=[query], mode="event", is_sensitive=self.catalog.is_sensitive_text(query))

        def set_status(status: str, reason: str = ""):
            def f(w: SceneManifest):
                for u in w.unresolved:
                    if u.id == unresolved_id:
                        u.status = status  # type: ignore[assignment]
                        if reason:
                            u.reason = reason
            return f

        if kw.is_sensitive:
            await self.system_update(scene_id, set_status("dismissed", "敏感素材，不自动加入"), {"kind": "system"})
            return
        await self.system_update(scene_id, set_status("searching"), {"kind": "system", "job": "fetch_sound"})
        vocab = SoundVocab(key=query[:12], zh=kw.zh or [query], en=kw.en, mode=kw.mode)
        got = await self.fetcher.fetch_vocab(vocab, per_key=1) if (kw.en or kw.zh) else []
        if not got:
            await self.system_update(scene_id, set_status("open", "网上暂时没有找到合适的素材"), {"kind": "system"})
            return
        self.resolver.refresh()
        m = self.store.get(scene_id)
        match = await self.resolver.find_sound(query, m, "ambience" if not params.get("object_id") else "object")
        if not match:
            await self.system_update(scene_id, set_status("open", "下载的素材不够匹配，等待人工审核"), {"kind": "system"})
            return

        def attach(w: SceneManifest):
            from .executor import _sound_spec
            from .models import AmbienceLayer
            changed = []
            oid = params.get("object_id")
            if oid and w.obj(oid):
                w.obj(oid).sound = _sound_spec(match)
                changed.append(oid)
            elif not oid:
                w.ambience.append(AmbienceLayer(id=new_id("amb"), clip_id=match.clip_id, name_zh=match.name_zh, role="manual"))
            for u in w.unresolved:
                if u.id == unresolved_id:
                    u.status = "resolved"
            return changed

        await self.system_update(scene_id, attach, {"kind": "system", "job": "fetch_sound_attach"})

    # ---------------- 前端素材表 ----------------
    def url(self, rel_path: str) -> str:
        return "/files/" + rel_path

    def assets_for(self, m: SceneManifest) -> dict:
        table: dict = {}
        for vp in m.viewpoints:
            for p in (vp.panorama_path, vp.depth_path):
                if p:
                    table[f"file:{p}"] = {"urls": [self.url(p)]}
        for o in m.objects:
            if o.asset_id:
                rec = self.registry.get(o.asset_id)
                if rec:
                    w = h = None
                    try:
                        from PIL import Image
                        with Image.open(self.registry.path(rec)) as im:
                            w, h = im.size
                    except Exception:
                        pass
                    table[o.asset_id] = {"urls": [self.url(f) for f in rec.files], "width": w, "height": h,
                                         "name": rec.name_zh}
            if o.model_asset_id:
                rec = self.registry.get(o.model_asset_id)
                if rec:
                    table[o.model_asset_id] = {"urls": [self.url(f) for f in rec.files]}
            if o.sound:
                files = self.resolver.clip_files(o.sound.clip_id)
                table[o.sound.clip_id] = {"urls": [self.url(f) for f in files]}
        for a in m.ambience:
            files = self.resolver.clip_files(a.clip_id)
            table[a.clip_id] = {"urls": [self.url(f) for f in files]}
        for s in m.stories:
            if s.audio_path:
                table[f"file:{s.audio_path}"] = {"urls": [self.url(s.audio_path)]}
        for ir in self.registry.by_kind("ir"):
            if ir.tags:
                table[f"ir:{ir.tags[0]}"] = {"urls": [self.url(ir.files[0])]}
        return table

    # ---------------- 创建场景 ----------------
    def create_scene(self, participant_id: str, condition: Condition, region: str) -> SceneManifest:
        m = SceneManifest(scene_id=new_id("sc"), participant_id=participant_id, condition=condition,
                          region_style=region if region in self.catalog.style["regions"] else "其他")  # type: ignore[arg-type]
        return self.store.create(m)

    async def build_generic(self, scene_id: str, session_id: Optional[str] = None) -> SceneManifest:
        """按地域模板搭建通用场景（generic / audio_only 条件）。全景图按地域缓存共用。"""
        m = self.store.get(scene_id)
        assert m is not None
        templates = yaml.safe_load((self.s.config_dir / "generic_scenes.yaml").read_text("utf-8"))
        tpl = templates.get(m.region_style) or templates["其他"]
        cache_path = self.s.assets_dir / "generic_scenes" / f"{slug(m.region_style)}.json"
        cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}

        cmds_by_vp: list[tuple[str, list[EditCommand]]] = []

        def build(w: SceneManifest):
            w.era = tpl.get("era", w.era)
            prev = None
            for vt in tpl["viewpoints"]:
                vp = create_viewpoint(w, vt["name"], vt.get("description", ""), prev)
                cached = cache.get(vt["name"])
                if w.condition == "audio_only":
                    pano, depth = self.generation.audio_only_background()
                    vp.panorama_path, vp.depth_path, vp.status = pano, depth, "ready"
                elif cached and (to_abs(cached["panorama_path"])).exists():
                    vp.panorama_path, vp.depth_path, vp.status = cached["panorama_path"], cached["depth_path"], "ready"
                    self.depth.load_png(vp.id, to_abs(vp.depth_path))
                from .commands import AddObject as _A
                from .placement import DIRECTION_YAW, DISTANCE_M
                from .models import Landmark
                for lt in vt.get("landmarks", []):
                    w.landmarks.append(Landmark(id=new_id("lm"), name_zh=lt["name"], viewpoint_id=vp.id,
                                                yaw_deg=DIRECTION_YAW[lt.get("direction", "front")],
                                                distance_m=DISTANCE_M[lt.get("distance", "mid")]))
                cmds = [_A(object_type=o["type"], location=LocationRef(landmark_name=o["landmark"],
                                                                         relation=o.get("relation", "at"),
                                                                         viewpoint_id=vp.id)) for o in vt.get("objects", [])]
                cmds_by_vp.append((vp.id, cmds))
                prev = vp.id
            refresh_auto_ambience(w, self.resolver)

        await self.system_update(scene_id, build, {"kind": "system", "job": "generic_template"})
        for vp_id, cmds in cmds_by_vp:
            await self.apply(scene_id, cmds, {"kind": "system", "job": "generic_objects"}, vp_id, "generic", session_id)
        m = self.store.get(scene_id)
        # 缺全景的点位：生成（共享）并写入缓存
        for vp in m.viewpoints:
            if not vp.panorama_path:
                self.spawn(self._generic_panorama(scene_id, vp.id, cache_path))
        return m

    async def _generic_panorama(self, scene_id: str, vp_id: str, cache_path: Path) -> None:
        m = self.store.get(scene_id)
        vp = m.viewpoint(vp_id)
        res = await self.generation.generate_panorama(m, vp, "new", shared=True)
        if not res:
            return
        self.depth.load_png(vp_id, to_abs(res.depth_path))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
        if not res.placeholder:
            cache[vp.name_zh] = {"panorama_path": res.panorama_path, "depth_path": res.depth_path}
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), "utf-8")

        def done(w: SceneManifest):
            v = w.viewpoint(vp_id)
            if not v:
                return []
            v.panorama_path, v.depth_path, v.status = res.panorama_path, res.depth_path, "ready"
            for o in w.objects:
                lm = w.landmark(o.landmark)
                if lm and lm.viewpoint_id == vp_id:
                    o.placement = resolve_placement(w, lm, o.offset, self.depth,
                                                    o.placement.elevation_m if o.placement else 0.0, (o.id,))
            return [vp_id]
        await self.system_update(scene_id, done, {"kind": "system", "job": "panorama", "viewpoint": vp_id})
