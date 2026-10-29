"""脚本公共部分：把 backend 加入导入路径，构造素材登记表。"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def registry():
    from app.assets.registry import AssetRegistry
    from app.config import get_settings
    s = get_settings()
    return AssetRegistry(ROOT, ROOT / "assets" / "manifest.json", s.data_dir / "generated_manifest.json")
