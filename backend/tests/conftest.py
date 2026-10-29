import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DEMO_MODE", "true")


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DEMO_MODE", "true")
    from app.config import reset_settings_cache
    reset_settings_cache()
    yield tmp_path / "data"
    reset_settings_cache()
