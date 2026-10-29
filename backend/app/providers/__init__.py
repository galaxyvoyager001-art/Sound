"""Provider 注册表：根据配置组装各项服务；缺 Key 或 DEMO_MODE 时使用本地实现。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from ..config import Settings
from .asr import ASRSession, DashScopeASRSession, NullASRSession
from .depth import DepthEstimator, make_depth_estimator
from .embedding import DashScopeEmbedder, Embedder, HashEmbedder
from .image import (ImageProvider, PanoramaProvider, ProceduralImage, ProceduralPanorama,
                    SkyboxPanorama, WanxImage, WanxPanorama)
from .llm import LLM, DashScopeLLM, MockLLM
from .model3d import MeshyImageTo3D, Model3DGenerator
from .sfx import ElevenLabsSFX, SoundEffectGenerator
from .tts import TTS, CosyVoiceTTS, NullTTS

log = logging.getLogger("providers")


@dataclass
class Providers:
    llm: LLM
    llm_model: Optional[str]
    llm_fast_model: Optional[str]
    asr_factory: Callable[[], ASRSession]
    tts: TTS
    image: ImageProvider
    panorama: PanoramaProvider
    depth: DepthEstimator
    embedder: Embedder
    sfx: SoundEffectGenerator
    model3d: Model3DGenerator
    cloud: bool

    def describe(self) -> dict:
        return {"cloud": self.cloud, "llm": self.llm_model, "llm_fast": self.llm_fast_model,
                "tts": type(self.tts).__name__, "image": self.image.name, "panorama": self.panorama.name,
                "depth": self.depth.name, "embedder": self.embedder.name, "sfx": self.sfx.name,
                "model3d": self.model3d.name}


def build_providers(s: Settings, mock_llm_handler=None) -> Providers:
    cloud = s.use_cloud
    if cloud:
        llm: LLM = DashScopeLLM(s.dashscope_api_key, s.compatible_base, s.llm_model)
        asr_factory = lambda: DashScopeASRSession(s.dashscope_api_key, s.ws_inference, s.asr_model)  # noqa: E731
        tts: TTS = CosyVoiceTTS(s.dashscope_api_key, s.ws_inference, s.tts_model, s.tts_voice, s.tts_rate,
                                s.data_dir / "cache" / "tts")
        image: ImageProvider = WanxImage(s.dashscope_api_key, s.api_base, s.image_model)
        embedder: Embedder = DashScopeEmbedder(s.dashscope_api_key, s.compatible_base, s.embedding_model,
                                               s.data_dir / "cache" / "embeddings.json")
    else:
        if mock_llm_handler is None:
            from ..conversation.demo_brain import demo_llm_handler
            mock_llm_handler = demo_llm_handler
        llm = MockLLM(mock_llm_handler)
        asr_factory = NullASRSession
        tts = NullTTS()
        image = ProceduralImage()
        embedder = HashEmbedder()

    if s.panorama_provider == "skybox" and s.blockade_api_key and not s.demo_mode:
        panorama: PanoramaProvider = SkyboxPanorama(s.blockade_api_key, s.blockade_style_id)
    elif cloud and isinstance(image, WanxImage):
        panorama = WanxPanorama(image)
    else:
        panorama = ProceduralPanorama()

    depth_kind = s.depth_provider
    depth = make_depth_estimator(depth_kind if depth_kind != "provider" else "geometric", s.root / s.depth_onnx_path)

    sfx: SoundEffectGenerator = ElevenLabsSFX(s.elevenlabs_api_key) if s.elevenlabs_api_key and not s.demo_mode \
        else SoundEffectGenerator()
    model3d: Model3DGenerator = MeshyImageTo3D(s.meshy_api_key) if (s.model3d_provider == "meshy" and s.meshy_api_key) \
        else Model3DGenerator()

    p = Providers(llm=llm, llm_model=s.llm_model if cloud else None,
                  llm_fast_model=s.llm_fast_model if cloud else None, asr_factory=asr_factory, tts=tts,
                  image=image, panorama=panorama, depth=depth, embedder=embedder, sfx=sfx, model3d=model3d,
                  cloud=cloud)
    log.info("服务配置：%s", p.describe())
    return p
