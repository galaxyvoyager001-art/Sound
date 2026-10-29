"""登记本地素材：自己修复的老录音、冲激响应、自己的录音。

目录约定：
  assets/sounds/my_restorations/   自己修复过的老歌、戏曲录音
  assets/sounds/own/               自己录制的声音
  assets/irs/                      设备冲激响应，文件名以 old_radio / loudspeaker_horn / gramophone 开头

每个音频文件旁边放一个同名的 .yaml 说明文件（第一次运行会自动生成模板，填好后再运行一次）：

  name_zh: 茉莉花（越剧唱段）
  tags: [老歌, 越剧]           # 第一个标签是“逻辑名称”，同名的录音会组成一组轮换播放
  era: [1950年代]
  region: [江南水乡]
  original_source: 1957 年中国唱片社 78 转唱片（个人收藏）
  copyright_status: copyrighted   # public_domain / cc / copyrighted / unknown
  license: 仅限本项目内部怀旧活动使用
  attribution: 演唱者：XXX
  acoustic_profile: gramophone    # clean / old_radio / loudspeaker_horn / gramophone
  mode: loop                      # loop / event
  gain_db: -8
  closing_song: true              # 可作为会谈收尾的欢快老歌
  sensitive: false
  notes: 自行降噪修复

copyright_status 不是 public_domain / cc 的录音只在本地会谈中播放，不会被导出到“老家漫游”网页。
用法：python scripts/register_local_assets.py
"""
import hashlib
import shutil
import subprocess
from pathlib import Path

import yaml

from _common import ROOT, registry

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
TEMPLATE = {
    "name_zh": "", "tags": ["老歌"], "era": ["any"], "region": ["any"], "original_source": "",
    "copyright_status": "unknown", "license": "", "attribution": "", "acoustic_profile": "gramophone",
    "mode": "loop", "gain_db": -8, "closing_song": False, "sensitive": False, "notes": "",
}


def ensure_mp3(src: Path) -> Path:
    """浏览器兼容：非 mp3 的转成 mp3（原文件保留）。"""
    if src.suffix.lower() == ".mp3":
        return src
    dst = src.with_suffix(".web.mp3")
    if not dst.exists() and shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ar", "44100", "-b:a", "192k", str(dst)], check=True)
    return dst if dst.exists() else src


def main():
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from app.assets.online import ffprobe_duration
    from app.assets.registry import AssetRecord, AssetSource, SoundProps

    reg = registry()
    n_new, n_tpl = 0, 0
    for folder, kind, src_type in ((ROOT / "assets" / "sounds" / "my_restorations", "music", "restoration"),
                                   (ROOT / "assets" / "sounds" / "own", "sound", "own_recording"),
                                   (ROOT / "assets" / "irs", "ir", "own_recording")):
        folder.mkdir(parents=True, exist_ok=True)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() not in AUDIO_EXT or f.name.endswith(".web.mp3"):
                continue
            meta_path = f.with_suffix(".yaml")
            if kind == "ir":
                profile = next((p for p in ("old_radio", "loudspeaker_horn", "gramophone") if f.stem.startswith(p)), None)
                if not profile:
                    print(f"跳过 {f.name}：冲激响应文件名需以 old_radio / loudspeaker_horn / gramophone 开头")
                    continue
                meta = yaml.safe_load(meta_path.read_text("utf-8")) if meta_path.exists() else {}
                rec = AssetRecord(id=f"ir_{profile}_{hashlib.sha1(f.name.encode()).hexdigest()[:6]}", kind="ir",
                                  name_zh=f"{profile} 冲激响应", tags=[profile], files=[ensure_mp3(f).relative_to(ROOT).as_posix()],
                                  source=AssetSource(type="own_recording", notes=meta.get("notes", ""),
                                                     url=meta.get("url")),
                                  license=meta.get("license", "自有"), attribution=meta.get("attribution"),
                                  review_status="approved")
                reg.register(rec)
                n_new += 1
                continue
            if not meta_path.exists():
                tpl = dict(TEMPLATE, name_zh=f.stem, tags=["老歌" if kind == "music" else f.stem])
                if kind == "sound":
                    tpl.update(copyright_status="not_applicable", license="自有录音", acoustic_profile="clean", mode="event")
                meta_path.write_text(yaml.safe_dump(tpl, allow_unicode=True, sort_keys=False), "utf-8")
                print(f"已生成说明模板，请填写后重新运行：{meta_path.relative_to(ROOT)}")
                n_tpl += 1
                continue
            meta = yaml.safe_load(meta_path.read_text("utf-8")) or {}
            if not meta.get("name_zh") or not meta.get("license"):
                print(f"跳过 {f.name}：说明文件缺少 name_zh 或 license")
                continue
            web = ensure_mp3(f)
            aid = f"{'mus' if kind == 'music' else 'own'}_{hashlib.sha1(f.name.encode()).hexdigest()[:8]}"
            tags = list(meta.get("tags") or [meta["name_zh"]])
            if meta.get("closing_song"):
                tags.append("收尾老歌")
            rec = AssetRecord(
                id=aid, kind=kind, name_zh=meta["name_zh"], tags=tags, region=meta.get("region") or ["any"],
                era=meta.get("era") or ["any"], files=[web.relative_to(ROOT).as_posix()],
                source=AssetSource(type=src_type, notes=meta.get("notes", ""), original_title=meta.get("original_source")),
                license=str(meta["license"]), attribution=meta.get("attribution") or None,
                sensitive=bool(meta.get("sensitive", False)),
                copyright_status=meta.get("copyright_status", "unknown"), review_status="approved",
                sound=SoundProps(mode=meta.get("mode", "loop"), gain_db=float(meta.get("gain_db", -8)),
                                 acoustic_profile=meta.get("acoustic_profile", "clean"),
                                 duration_s=round(ffprobe_duration(web), 2)))
            reg.register(rec)
            n_new += 1
            print("登记", rec.id, rec.name_zh, rec.copyright_status)
    print(f"完成：登记/更新 {n_new} 个，新生成模板 {n_tpl} 个。")


if __name__ == "__main__":
    main()
