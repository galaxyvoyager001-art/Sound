"""服务容器：把配置、数据库、素材、Provider、生成服务组装在一起，供 API 和会谈共用。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .assets.catalog import Catalog, get_catalog
from .assets.online import SoundFetcher
from .assets.registry import AssetRegistry
from .assets.retrieval import AssetResolverImpl
from .config import Settings, get_settings
from .db import Database
from .generation.service import GenerationService
from .providers import Providers, build_providers
from .providers.depth import DepthMapSampler
from .scene.service import SceneService
from .scene.store import SceneStore


@dataclass
class AppContext:
    settings: Settings
    db: Database
    store: SceneStore
    registry: AssetRegistry
    catalog: Catalog
    providers: Providers
    resolver: AssetResolverImpl
    depth: DepthMapSampler
    generation: GenerationService
    fetcher: SoundFetcher
    scenes: SceneService


_ctx: Optional[AppContext] = None


def build_context(settings: Optional[Settings] = None, mock_llm_handler=None) -> AppContext:
    s = settings or get_settings()
    db = Database(s.db_path)
    store = SceneStore(db)
    registry = AssetRegistry(s.root, s.assets_dir / "manifest.json", s.data_dir / "generated_manifest.json")
    catalog = get_catalog()
    providers = build_providers(s, mock_llm_handler)
    resolver = AssetResolverImpl(registry, catalog, providers.embedder)
    depth = DepthMapSampler()
    generation = GenerationService(s, providers, registry, catalog, depth)
    fetcher = SoundFetcher(s.root, registry, catalog, s.freesound_api_key, providers.sfx)
    scenes = SceneService(s, store, registry, catalog, resolver, depth, generation, fetcher, providers, db)
    return AppContext(s, db, store, registry, catalog, providers, resolver, depth, generation, fetcher, scenes)


def get_context() -> AppContext:
    global _ctx
    if _ctx is None:
        _ctx = build_context()
    return _ctx


def set_context(ctx: Optional[AppContext]) -> None:
    global _ctx
    _ctx = ctx
