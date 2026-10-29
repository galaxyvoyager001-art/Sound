"""按 config/catalog.yaml 的声音词表在线检索、下载、登记声音素材。

用法：
  python scripts/fetch_sounds.py                    # 全部词条，每条最多 2 个
  python scripts/fetch_sounds.py --keys 木梆子 自行车铃 --per-key 3
  python scripts/fetch_sounds.py --dry-run          # 只列出候选，不下载
  python scripts/fetch_sounds.py --query 木梆子 "wooden fish percussion"   # 给某个词条追加检索词

来源：Wikimedia Commons（无需 Key）；配置了 FREESOUND_API_KEY 时同时检索 Freesound（只取 CC0 / CC-BY）；
配置了 ELEVENLABS_API_KEY 时，网上找不到的词条用文生音效生成。授权不明确或不允许使用的素材不会下载。
下载的素材 review_status=pending，请试听后在设置页或 manifest.json 中改为 approved / rejected。
"""
import argparse
import asyncio

from _common import registry


async def main():
    from app.assets.catalog import get_catalog
    from app.assets.online import SoundFetcher
    from app.config import get_settings
    from app.providers.sfx import ElevenLabsSFX

    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="*", help="只处理这些词条（逻辑名称）")
    ap.add_argument("--per-key", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--query", nargs="+", metavar=("KEY", "QUERY"), help="给一个词条追加检索词")
    args = ap.parse_args()

    s = get_settings()
    cat = get_catalog()
    reg = registry()
    sfx = ElevenLabsSFX(s.elevenlabs_api_key) if s.elevenlabs_api_key else None
    f = SoundFetcher(s.root, reg, cat, s.freesound_api_key, sfx)
    vocabs = cat.sounds
    extra: list[str] = []
    if args.query:
        vocabs = [v for v in vocabs if v.key == args.query[0]]
        extra = args.query[1:]
    elif args.keys:
        vocabs = [v for v in vocabs if v.key in args.keys]
    missing = []
    for v in vocabs:
        if args.dry_run:
            print(f"\n== {v.key} ==")
            for c in (await f.candidates(v, extra))[:8]:
                print(f"  {c.score:.2f}  [{c.license}] {c.title}  ({c.site}, {c.duration_s or '?'} s)\n        {c.page_url}")
            continue
        got = await f.fetch_vocab(v, args.per_key, extra)
        have = [a for a in reg.all() if v.key in a.tags and a.kind in ("sound", "music")]
        print(f"{v.key}: 新增 {len(got)}，共 {len(have)}")
        if not have:
            missing.append(v.key)
    if missing:
        print("\n仍缺少素材的词条（可配置 FREESOUND_API_KEY / ELEVENLABS_API_KEY 后重试，或手动添加）：")
        print("  " + "、".join(missing))


if __name__ == "__main__":
    asyncio.run(main())
