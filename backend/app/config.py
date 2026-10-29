"""全局配置：从 .env 读取，所有路径相对项目根目录。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "data"))))
    assets_dir: Path = field(default_factory=lambda: Path(_env("ASSETS_DIR", str(ROOT / "assets"))))
    config_dir: Path = ROOT / "config"
    frontend_dist: Path = ROOT / "frontend" / "dist"

    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY"))
    dashscope_host: str = field(default_factory=lambda: _env("DASHSCOPE_HOST", "dashscope.aliyuncs.com"))

    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3.7-plus"))
    llm_fast_model: str = field(default_factory=lambda: _env("LLM_FAST_MODEL", "qwen3.8-flash"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"))
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "fun-asr-realtime"))
    asr_vocabulary_id: str = field(default_factory=lambda: _env("ASR_VOCABULARY_ID"))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "cosyvoice-v3-flash"))
    tts_voice: str = field(default_factory=lambda: _env("TTS_VOICE", "longanrou_v3"))
    tts_rate: float = field(default_factory=lambda: float(_env("TTS_RATE", "0.85") or 0.85))
    image_model: str = field(default_factory=lambda: _env("IMAGE_MODEL", "wan2.6-t2i"))

    panorama_provider: str = field(default_factory=lambda: _env("PANORAMA_PROVIDER", "wanx"))
    blockade_api_key: str = field(default_factory=lambda: _env("BLOCKADE_API_KEY"))
    blockade_style_id: str = field(default_factory=lambda: _env("BLOCKADE_STYLE_ID"))

    depth_provider: str = field(default_factory=lambda: _env("DEPTH_PROVIDER", "local_onnx"))
    depth_onnx_path: str = field(default_factory=lambda: _env("DEPTH_ONNX_PATH", "models/depth_anything_v2_small.onnx"))

    freesound_api_key: str = field(default_factory=lambda: _env("FREESOUND_API_KEY"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))

    model3d_provider: str = field(default_factory=lambda: _env("MODEL3D_PROVIDER", "off"))
    tripo_api_key: str = field(default_factory=lambda: _env("TRIPO_API_KEY"))
    meshy_api_key: str = field(default_factory=lambda: _env("MESHY_API_KEY"))

    demo_mode: bool = field(default_factory=lambda: _env_bool("DEMO_MODE", False))

    # 会谈参数
    reply_silence_s: float = field(default_factory=lambda: float(_env("REPLY_SILENCE_S", "4.0")))
    default_session_minutes: int = 30
    confidence_threshold: float = 0.7

    @property
    def use_cloud(self) -> bool:
        return bool(self.dashscope_api_key) and not self.demo_mode

    @property
    def compatible_base(self) -> str:
        return f"https://{self.dashscope_host}/compatible-mode/v1"

    @property
    def api_base(self) -> str:
        return f"https://{self.dashscope_host}/api/v1"

    @property
    def ws_inference(self) -> str:
        return f"wss://{self.dashscope_host}/api-ws/v1/inference"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.sqlite3"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.data_dir / "sessions", self.data_dir / "exports",
                  self.data_dir / "cache", self.data_dir / "reports"):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def to_rel(path: Path) -> str:
    """绝对路径 → 清单里保存的相对路径：素材目录下为 assets/…，数据目录下为 data/…。"""
    s = get_settings()
    p = Path(path).resolve()
    for prefix, base in (("assets", s.assets_dir), ("data", s.data_dir)):
        try:
            return f"{prefix}/" + p.relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            continue
    return p.relative_to(s.root.resolve()).as_posix()


def to_abs(rel: str) -> Path:
    s = get_settings()
    if rel.startswith("assets/"):
        return Path(s.assets_dir) / rel[len("assets/"):]
    if rel.startswith("data/"):
        return Path(s.data_dir) / rel[len("data/"):]
    return s.root / rel
