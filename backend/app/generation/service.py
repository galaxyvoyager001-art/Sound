"""全景图 / 深度图 / 物件插画的生成、缓存与登记。所有失败都降级，不抛给会谈。"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from pypinyin import lazy_pinyin

from ..assets.catalog import Catalog
from ..assets.registry import AssetRecord, AssetRegistry, AssetSource
from ..config import Settings, to_rel
from ..providers import Providers
from ..providers.depth import DepthMapSampler, encode_depth_png
from ..providers.image import ProceduralPanorama
from ..scene.models import SceneManifest, Viewpoint
from .imaging import blurred_warm_background, finalize_panorama, matte_object, to_jpeg, to_png
from .style import object_prompt, panorama_prompt, style_key

log = logging.getLogger("generation")


def slug(text: str) -> str:
    s = "".join(lazy_pinyin(text, errors="ignore"))
    s = "".join(c for c in s.lower() if c.isalnum())
    return s[:24] or "x"


@dataclass
class PanoramaResult:
    panorama_path: str
    depth_path: str
    asset_id: str
    prompt: str
    placeholder: bool = False
    error: str = ""


class GenerationService:
    def __init__(self, settings: Settings, providers: Providers, registry: AssetRegistry, catalog: Catalog,
                 depth_sampler: DepthMapSampler):
        self.s = settings
        self.p = providers
        self.registry = registry
        self.catalog = catalog
        self.depth_sampler = depth_sampler
        self._obj_locks: dict[str, asyncio.Lock] = {}
        self.last_error = ""

    def rel(self, path: Path) -> str:
        return to_rel(path)

    # ---------------- 全景 ----------------
    async def generate_panorama(self, m: SceneManifest, vp: Viewpoint, mode: str = "new",
                                shared: bool = False) -> Optional[PanoramaResult]:
        """生成（或重绘）一个点位的全景图和深度图。mode=regenerate 失败时返回 None（保持当前画面）。"""
        prompt, negative = panorama_prompt(self.catalog, m, vp, restyle=(mode == "regenerate"))
        placeholder = False
        error = ""
        provider_depth = None
        try:
            raw, provider_depth = await self.p.panorama.generate_panorama(prompt, negative)
            native = self.p.panorama.native_equirect
        except Exception as e:
            log.warning("全景生成失败：%s", e)
            self.last_error = f"全景生成失败：{str(e)[:300]}"
            if mode == "regenerate" and vp.panorama_path:
                return None
            error = self.last_error
            if self.p.cloud:
                # 云端失败时用柔和的暖色背景兜底（比粗糙的程序化房子更不容易让人困惑）
                import io
                b = io.BytesIO()
                blurred_warm_background().save(b, "PNG")
                raw = b.getvalue()
            else:
                raw, _ = await ProceduralPanorama().generate_panorama(prompt, negative)
            native, placeholder = True, True

        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, finalize_panorama, raw, native)
        depth_kind = self.s.depth_provider
        use_provider_depth = provider_depth if depth_kind == "provider" or self.p.depth.name == "geometric" else None
        depth = await loop.run_in_executor(None, self.p.depth.estimate, img, use_provider_depth)

        base = (self.s.assets_dir / "images" / "panoramas") if shared else (self.s.data_dir / "scenes" / m.scene_id)
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d%H%M%S")
        name = f"{slug(m.region_style)}_{slug(vp.name_zh)}_{stamp}"
        pano_path = base / f"{name}.jpg"
        depth_path = base / f"{name}_depth.png"
        pano_path.write_bytes(to_jpeg(img))
        depth_path.write_bytes(encode_depth_png(depth))
        self.depth_sampler.set(vp.id, depth)

        asset_id = f"pano_{name}"
        service = "程序化占位" if placeholder else f"{self.p.panorama.name} / {getattr(self.p.image, 'model', '')}"
        rec = AssetRecord(
            id=asset_id, kind="panorama", name_zh=f"{m.region_style}·{vp.name_zh}", tags=[m.region_style, vp.name_zh],
            region=[m.region_style], era=[m.era], files=[self.rel(pano_path), self.rel(depth_path)],
            source=AssetSource(type="procedural" if placeholder else "ai", service=service, prompt=prompt,
                               notes=f"深度：{self.p.depth.name}"),
            license="AI 生成（服务条款允许使用）" if not placeholder else "程序生成", review_status="approved",
            style_key=style_key(self.catalog), participant_id=None if shared else m.participant_id)
        try:
            self.registry.register(rec, private=not shared)
        except Exception as e:
            log.warning("全景登记失败：%s", e)
        return PanoramaResult(self.rel(pano_path), self.rel(depth_path), asset_id, prompt, placeholder, error)

    def audio_only_background(self) -> tuple[str, str]:
        base = self.s.assets_dir / "images" / "panoramas"
        base.mkdir(parents=True, exist_ok=True)
        p = base / "audio_only_warm_blur.jpg"
        d = base / "audio_only_warm_blur_depth.png"
        if not p.exists():
            p.write_bytes(to_jpeg(blurred_warm_background(), quality=80))
            from ..providers.depth import geometric_prior
            d.write_bytes(encode_depth_png(geometric_prior(open_distance=40)))
        return self.rel(p), self.rel(d)

    # ---------------- 物件插画 ----------------
    def find_object_asset(self, object_type: str, region: str) -> Optional[AssetRecord]:
        sk = style_key(self.catalog)
        entry = self.catalog.find_object(object_type)
        canon = entry.type if entry else object_type
        cands = [a for a in self.registry.by_kind("object_image") if a.style_key == sk and canon in a.tags]
        if not cands:
            return None
        cands.sort(key=lambda a: (region not in a.region, a.created_at))
        return cands[0]

    async def object_asset(self, object_type: str, region: str, era: str) -> Optional[AssetRecord]:
        """同一风格下同一种物件只生成一次。失败返回 None（前端显示柔和的占位光点）。"""
        entry = self.catalog.find_object(object_type)
        canon = entry.type if entry else object_type.strip()
        lock = self._obj_locks.setdefault(canon, asyncio.Lock())
        async with lock:
            hit = self.find_object_asset(canon, region)
            if hit:
                return hit
            prompt, negative = object_prompt(self.catalog, canon, entry, region, era)
            try:
                raw = await self.p.image.generate(prompt if self.p.cloud else f"{canon}|{prompt}", negative, 1280, 1280)
                service = f"{self.p.image.name} / {getattr(self.p.image, 'model', '')}"
                src_type = "ai" if self.p.cloud else "procedural"
            except Exception as e:
                log.warning("物件插画生成失败 %s：%s", canon, e)
                self.last_error = f"物件图“{canon}”生成失败：{str(e)[:300]}"
                return None
            loop = asyncio.get_running_loop()
            img: Image.Image = await loop.run_in_executor(None, matte_object, raw)
            base = self.s.assets_dir / "images" / "objects"
            base.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha1(prompt.encode()).hexdigest()[:6]
            path = base / f"{slug(canon)}_{h}.png"
            path.write_bytes(to_png(img))
            rec = AssetRecord(
                id=f"obj_{slug(canon)}_{h}", kind="object_image", name_zh=canon,
                tags=[canon] + (entry.synonyms if entry else []), region=[region], era=[era],
                files=[self.rel(path)],
                source=AssetSource(type=src_type, service=service, prompt=prompt,
                                   notes=f"白底生成后本地抠图；宽高 {img.width}×{img.height}"),
                license="AI 生成（服务条款允许使用）" if src_type == "ai" else "程序生成",
                review_status="approved" if src_type == "ai" else "pending", style_key=style_key(self.catalog))
            self.registry.register(rec)
            return rec

    async def object_model3d(self, image_rec: AssetRecord) -> Optional[AssetRecord]:
        if not self.p.model3d.available:
            return None
        existing = next((a for a in self.registry.by_kind("model3d") if image_rec.id in a.tags), None)
        if existing:
            return existing
        try:
            glb = await self.p.model3d.image_to_glb(self.registry.path(image_rec).read_bytes())
        except Exception as e:
            log.warning("3D 生成失败：%s", e)
            return None
        base = self.s.assets_dir / "models3d"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{image_rec.id}.glb"
        path.write_bytes(glb)
        rec = AssetRecord(id=f"m3d_{image_rec.id}", kind="model3d", name_zh=image_rec.name_zh,
                          tags=[image_rec.id] + image_rec.tags, files=[self.rel(path)],
                          source=AssetSource(type="ai", service=self.p.model3d.name, notes=f"由 {image_rec.id} 图生 3D"),
                          license="AI 生成（服务条款允许使用）", review_status="approved",
                          style_key=image_rec.style_key)
        self.registry.register(rec)
        return rec
