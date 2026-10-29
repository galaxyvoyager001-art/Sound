import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DEMO_MODE", "true")


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    # 素材目录也放到临时目录，测试不会污染仓库里的 assets/manifest.json
    assets = tmp_path / "assets"
    assets.mkdir()
    real = Path(__file__).resolve().parents[2] / "assets" / "manifest.json"
    if real.exists():
        shutil.copy(real, assets / "manifest.json")
    monkeypatch.setenv("ASSETS_DIR", str(assets))
    monkeypatch.setenv("DEMO_MODE", "true")
    from app.config import reset_settings_cache
    reset_settings_cache()
    yield tmp_path / "data"
    reset_settings_cache()
