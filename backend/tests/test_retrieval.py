"""声音检索：同义词、拼音容错、敏感素材排除、无匹配时不硬塞。"""
import pytest

from app.assets.catalog import load_catalog
from app.assets.online import normalize_license
from app.assets.registry import AssetRecord, AssetRegistry, AssetSource, SoundProps
from app.assets.retrieval import AssetResolverImpl, pinyin_sim
from app.config import ROOT
from app.scene.models import SceneManifest


def rec(i, key, *, sensitive=False, amb=False, kind="sound", status="pending"):
    return AssetRecord(id=f"snd_{i}", kind=kind, name_zh=key, tags=[key], files=[f"assets/x/{i}.mp3"],
                       source=AssetSource(type="procedural"), license="CC0-1.0", sensitive=sensitive,
                       review_status=status, sound=SoundProps(mode="loop" if amb else "event", is_ambience=amb))


@pytest.fixture
def resolver(tmp_path):
    reg = AssetRegistry(tmp_path, tmp_path / "m.json", tmp_path / "p.json")
    for i, (k, kw) in enumerate([("木梆子", {}), ("木梆子", {}), ("自行车铃", {}), ("雨打瓦片", {"amb": True}),
                                 ("蝉鸣", {"amb": True}), ("鸡叫", {}), ("防空警报", {"sensitive": True}),
                                 ("远处人声", {"amb": True}), ("鸟叫", {"amb": True})]):
        reg.register(rec(i, k, **kw))
    return AssetResolverImpl(reg, load_catalog(ROOT / "config"))


def scene():
    return SceneManifest(scene_id="s", participant_id="p")


async def test_object_synonym_maps_to_sound(resolver):
    m = await resolver.find_sound("卖馄饨的", scene())
    assert m and m.clip_id == "grp:木梆子" and m.variants == 2
    m = await resolver.find_sound("脚踏车", scene())
    assert m and m.clip_id == "grp:自行车铃"


async def test_pinyin_tolerates_asr_typos(resolver):
    assert pinyin_sim("混沌", "馄饨") >= 0.75
    assert pinyin_sim("木梆子", "木帮子") == 1.0
    assert pinyin_sim("自行车", "收音机") < 0.2
    m = await resolver.find_sound("木帮子", scene())  # 识别错字
    assert m and m.clip_id == "grp:木梆子"


async def test_no_forced_match(resolver):
    assert await resolver.find_sound("爆米花机砰的一声", scene()) is None
    assert await resolver.find_sound("飞机", scene()) is None


async def test_sensitive_never_returned(resolver):
    assert await resolver.find_sound("防空警报", scene()) is None
    assert await resolver.find_sound("警报声", scene()) is None


async def test_exclude_and_variants(resolver):
    assert resolver.sound_variants("grp:木梆子") == 2
    m = await resolver.find_sound("梆子", scene(), exclude=("grp:木梆子",))
    assert m is None or m.clip_id != "grp:木梆子"


def test_auto_ambience_rules(resolver):
    m = scene()
    m.weather, m.time_of_day, m.season = "小雨", "上午", "夏"
    clips = {a.clip_id for a in resolver.auto_ambience(m)}
    assert "grp:雨打瓦片" in clips and "grp:远处人声" in clips
    assert "grp:蝉鸣" not in clips  # 下雨时没有蝉鸣
    m.weather = "晴"
    clips = {a.clip_id for a in resolver.auto_ambience(m)}
    assert "grp:蝉鸣" in clips and "grp:鸟叫" in clips


def test_license_normalization():
    assert normalize_license("CC0") == "CC0-1.0"
    assert normalize_license("CC BY-SA 4.0") == "CC-BY-SA-4.0"
    assert normalize_license("CC BY 3.0") == "CC-BY-3.0"
    assert normalize_license("Public domain") == "Public-Domain"
    assert normalize_license("CC BY-NC 4.0") is None
    assert normalize_license("Attribution NonCommercial") is None
    assert normalize_license("") is None
    assert normalize_license("GFDL") is None


def test_registry_requires_license(tmp_path):
    reg = AssetRegistry(tmp_path, tmp_path / "m.json", tmp_path / "p.json")
    bad = rec(1, "x").model_copy(update={"source": AssetSource(type="web", url="http://x"), "license": "unknown"})
    with pytest.raises(ValueError):
        reg.register(bad)
    bad = rec(2, "x").model_copy(update={"license": "CC-BY-4.0", "attribution": None})
    with pytest.raises(ValueError):
        reg.register(bad)
    reg.register(rec(3, "x"))
    assert (tmp_path / "assets" / "REFERENCES.md").exists()
