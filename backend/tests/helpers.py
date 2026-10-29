from __future__ import annotations

from typing import Optional

from app.scene.executor import ObjectMeta, SoundMatch, create_viewpoint
from app.scene.models import AmbienceLayer, SceneManifest

SOUNDS = {
    "馄饨": SoundMatch("snd_bangzi", "木梆子", "event", (10, 30), "clean", -3, variants=3),
    "梆子": SoundMatch("snd_bangzi", "木梆子", "event", (10, 30), "clean", -3, variants=3),
    "自行车": SoundMatch("snd_bike_bell", "自行车铃", "event", (8, 20), "clean", -4, variants=2),
    "收音机": SoundMatch("snd_radio_opera", "越剧", "loop", (8, 25), "old_radio", -6),
    "越剧": SoundMatch("snd_radio_opera", "越剧", "loop", (8, 25), "old_radio", -6),
    "鸡": SoundMatch("snd_chicken", "鸡叫", "event", (15, 40), "clean", -8),
    "雨": SoundMatch("snd_rain_tiles", "雨打瓦片", "loop", (8, 25), "clean", -10),
    "蝉": SoundMatch("snd_cicada", "蝉鸣", "loop", (8, 25), "clean", -12),
    "狗": SoundMatch("snd_dog", "狗叫", "event", (20, 60), "clean", -8),
}


class FakeResolver:
    async def find_sound(self, query: str, manifest: SceneManifest, purpose: str = "object",
                         exclude: tuple[str, ...] = ()) -> Optional[SoundMatch]:
        for k, v in SOUNDS.items():
            if k in query and v.clip_id not in exclude:
                return v
        if "狗" in query and "snd_dog" in exclude:
            return SoundMatch("snd_dog2", "狗叫2", "event")
        return None

    def object_meta(self, object_type: str) -> ObjectMeta:
        if "收音机" in object_type:
            return ObjectMeta(0.4, 1.0, "收音机 越剧", True)
        if "馄饨" in object_type:
            return ObjectMeta(1.5, 0.0, "馄饨 梆子", True)
        if "自行车" in object_type:
            return ObjectMeta(1.0, 0.0, "自行车", True)
        if "鸡" in object_type:
            return ObjectMeta(0.8, 0.0, "鸡", True)
        return ObjectMeta(1.2, 0.0, None, False)

    def sound_variants(self, clip_id: str) -> int:
        return next((v.variants for v in SOUNDS.values() if v.clip_id == clip_id), 1)

    def auto_ambience(self, manifest: SceneManifest) -> list[AmbienceLayer]:
        out = [AmbienceLayer(id="amb_auto_base", clip_id="snd_base", name_zh="弄堂底噪", role="base", auto=True)]
        if manifest.weather in ("小雨", "大雨"):
            out.append(AmbienceLayer(id="amb_auto_rain", clip_id="snd_rain_tiles", name_zh="雨", role="weather", auto=True))
        if manifest.season == "夏" and manifest.time_of_day not in ("夜晚",):
            out.append(AmbienceLayer(id="amb_auto_cicada", clip_id="snd_cicada", name_zh="蝉鸣", role="season", auto=True))
        return out


def make_scene() -> SceneManifest:
    m = SceneManifest(scene_id="sc_test", participant_id="P001")
    vp = create_viewpoint(m, "家门口", "石库门门口")
    m.entry_viewpoint = vp.id
    return m
