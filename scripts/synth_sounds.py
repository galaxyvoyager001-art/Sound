"""程序合成一批基础音效（物理建模 / 滤波噪声），保证在没有网络素材时系统也有声音可用。

这些声音登记为 source.type=procedural、授权 CC0（本项目自制），review_status=pending。
网上检索到更真实的录音后，检索会优先使用已审核（approved）的录音；也可以直接把这些占位素材标为 rejected。

用法：python scripts/synth_sounds.py [--force]
"""
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from _common import ROOT, registry

SR = 44100
rng = np.random.default_rng(20261029)
OUT = ROOT / "assets" / "sounds" / "library"


def env_exp(n, tau_s):
    return np.exp(-np.arange(n) / (tau_s * SR))


def bandnoise(n, lo, hi):
    """频域塑形的带限噪声。"""
    spec = np.fft.rfft(rng.normal(0, 1, n))
    f = np.fft.rfftfreq(n, 1 / SR)
    mask = ((f >= lo) & (f <= hi)).astype(float)
    # 边缘平滑
    mask = np.convolve(mask, np.hanning(31) / np.hanning(31).sum(), mode="same")
    x = np.fft.irfft(spec * mask, n)
    return x / (np.abs(x).max() + 1e-9)


def modal(n, freqs, amps, taus, t0=0.0):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a, tau in zip(freqs, amps, taus):
        y += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    return y


def place(buf, clip, at_s, gain=1.0):
    i = int(at_s * SR)
    j = min(len(buf), i + len(clip))
    buf[i:j] += clip[: j - i] * gain


def woodblock_knock(pitch=1.0):
    n = int(0.25 * SR)
    hit = modal(n, [820 * pitch, 1630 * pitch, 2470 * pitch, 3900 * pitch], [1, 0.45, 0.25, 0.1], [0.045, 0.03, 0.02, 0.01])
    click = bandnoise(n, 1500, 9000) * env_exp(n, 0.004) * 0.6
    return hit + click


def gen_bangzi(variant):
    # 卖馄饨的梆子：“梆、梆梆——”的节奏，远处略带回声
    pattern = [[0, 0.42, 0.62], [0, 0.35, 0.7, 0.9], [0, 0.5, 0.68, 0.86]][variant % 3]
    buf = np.zeros(int(2.2 * SR))
    for k, t in enumerate(pattern):
        place(buf, woodblock_knock(1.0 + (0.04 if k % 2 else 0) + rng.normal(0, 0.01)), t, 0.8 + rng.uniform(-0.1, 0.1))
    return buf


def gen_bike_bell(variant):
    # 转铃：两个略失谐的金属碗 + 快速连续敲击
    buf = np.zeros(int(1.6 * SR))
    strikes = [0, 0.11, 0.22] if variant == 0 else [0, 0.09, 0.18, 0.27, 0.36]
    for t in strikes:
        n = int(1.2 * SR)
        b = modal(n, [2350, 2390, 5150, 6920, 8800], [1, 0.8, 0.5, 0.3, 0.15], [0.55, 0.5, 0.25, 0.15, 0.08])
        b += bandnoise(n, 3000, 12000) * env_exp(n, 0.003) * 0.4
        place(buf, b, t, 0.5)
    return buf


def gen_hand_bell():
    buf = np.zeros(int(4.0 * SR))
    for k in range(12):
        n = int(1.0 * SR)
        b = modal(n, [1180, 2950, 4120, 5900], [1, 0.6, 0.35, 0.2], [0.6, 0.35, 0.2, 0.1])
        place(buf, b, 0.08 + k * 0.28 + rng.normal(0, 0.015), 0.4 + 0.2 * (k % 2))
    return buf


def gen_clock(seconds=12.0):
    buf = np.zeros(int(seconds * SR))
    for k in range(int(seconds)):
        n = int(0.06 * SR)
        tick = modal(n, [3200, 5100, 7300] if k % 2 else [2900, 4700, 6900], [1, 0.5, 0.3], [0.012, 0.008, 0.005])
        tick += bandnoise(n, 2000, 10000) * env_exp(n, 0.002) * 0.5
        place(buf, tick, k + 0.02, 0.35)
    return buf


def gen_abacus():
    buf = np.zeros(int(2.5 * SR))
    t = 0.05
    for _ in range(9):
        n = int(0.05 * SR)
        c = modal(n, [2600 + rng.uniform(-200, 200), 4800], [1, 0.4], [0.008, 0.005]) + bandnoise(n, 2500, 9000) * env_exp(n, 0.002) * 0.6
        place(buf, c, t, 0.4)
        t += rng.uniform(0.08, 0.3)
    return buf


def gen_enamel_basin():
    n = int(2.0 * SR)
    b = modal(n, [410, 1043, 1912, 2980, 4210], [1, 0.7, 0.45, 0.3, 0.15], [0.9, 0.6, 0.4, 0.25, 0.15])
    b += bandnoise(n, 800, 7000) * env_exp(n, 0.01) * 0.5
    return b * 0.6


def gen_rain(seconds=30.0, heavy=False):
    n = int(seconds * SR)
    bed = bandnoise(n, 300 if heavy else 600, 9000) * (0.35 if heavy else 0.18)
    # 雨点打在瓦片上：大量短促的随机滴答
    drops = np.zeros(n)
    rate = 900 if heavy else 260
    for _ in range(int(rate * seconds)):
        m = int(0.02 * SR)
        f = rng.uniform(1800, 6000)
        d = np.sin(2 * np.pi * f * np.arange(m) / SR) * env_exp(m, rng.uniform(0.002, 0.006))
        place(drops, d, rng.uniform(0, seconds - 0.03), rng.uniform(0.02, 0.12))
    # 屋檐滴水：较大的水滴
    for _ in range(int(seconds * (3 if heavy else 1.2))):
        m = int(0.12 * SR)
        f0 = rng.uniform(500, 900)
        sweep = np.sin(2 * np.pi * np.cumsum(np.linspace(f0, f0 * 1.8, m)) / SR) * env_exp(m, 0.025)
        place(drops, sweep, rng.uniform(0, seconds - 0.2), rng.uniform(0.08, 0.2))
    stereo = np.stack([bed + drops, np.roll(bed, 1200) + np.roll(drops, 700)], 1)
    return stereo


def gen_wind(seconds=30.0, leaves=False):
    n = int(seconds * SR)
    base = bandnoise(n, 80, 1200 if not leaves else 6000)
    t = np.arange(n) / SR
    gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.07 * t + 1) * np.sin(2 * np.pi * 0.023 * t)
    gust = np.convolve(np.abs(gust), np.ones(4000) / 4000, mode="same")
    y = base * gust * (0.5 if leaves else 0.7)
    return np.stack([y, np.roll(y, 3000)], 1)


def gen_crickets(seconds=30.0):
    n = int(seconds * SR)
    out = np.zeros((n, 2))
    for k in range(5):  # 几只不同远近的蟋蟀
        f = rng.uniform(4300, 5200)
        rate = rng.uniform(2.5, 4.0)
        t = np.arange(n) / SR
        chirp_env = (np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)) > 0.55).astype(float)
        pulse = (np.sin(2 * np.pi * 30 * t) > 0).astype(float)
        sig = np.sin(2 * np.pi * f * t) * np.convolve(chirp_env * pulse, np.ones(60) / 60, mode="same")
        pan = rng.uniform(0.2, 0.8)
        g = rng.uniform(0.05, 0.15)
        out[:, 0] += sig * g * (1 - pan)
        out[:, 1] += sig * g * pan
    return out


def gen_cicadas(seconds=30.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for k in range(3):
        buzz = bandnoise(n, 4000, 7500) * (0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(80, 120) * t))
        swell = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.03, 0.08) * t + rng.uniform(0, 6))
        pan = rng.uniform(0.2, 0.8)
        out[:, 0] += buzz * swell * 0.12 * (1 - pan)
        out[:, 1] += buzz * swell * 0.12 * pan
    return out


def gen_radio_static(seconds=6.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = bandnoise(n, 300, 5000) * 0.25
    # 调台时的啸叫扫频
    sweep = np.sin(2 * np.pi * np.cumsum(800 + 600 * np.sin(2 * np.pi * 0.3 * t)) / SR) * 0.08
    burst = (np.sin(2 * np.pi * 0.5 * t) > 0.6) * bandnoise(n, 500, 3000) * 0.2
    return noise + sweep * (t > 1.5) * (t < 4) + burst


def gen_steam_whistle(seconds=4.0, f0=440.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    env = np.clip(t / 0.25, 0, 1) * np.clip((seconds - t) / 0.6, 0, 1)
    chord = sum(np.sin(2 * np.pi * f * t * (1 + 0.003 * np.sin(2 * np.pi * 5 * t))) * a
                for f, a in ((f0, 1), (f0 * 1.26, 0.8), (f0 * 1.5, 0.6), (f0 * 2, 0.3)))
    breath = bandnoise(n, 500, 4000) * 0.3
    return (chord * 0.25 + breath) * env


def gen_bellows(seconds=10.0):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    puff = np.clip(np.sin(2 * np.pi * 0.55 * t), 0, 1) ** 2
    y = bandnoise(n, 100, 1500) * puff * 0.5
    creak = np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.55 * t) > 0.95) * 0.05
    return y + creak


def gen_stove(seconds=20.0):
    n = int(seconds * SR)
    y = bandnoise(n, 60, 800) * 0.12
    for _ in range(int(seconds * 4)):
        m = int(0.03 * SR)
        place(y, bandnoise(m, 1500, 8000) * env_exp(m, 0.004), rng.uniform(0, seconds - 0.05), rng.uniform(0.05, 0.3))
    return y


def gen_sewing(seconds=8.0):
    n = int(seconds * SR)
    y = np.zeros(n)
    t = 0.0
    rate = 11.0
    while t < seconds - 0.1:
        m = int(0.03 * SR)
        place(y, modal(m, [1900, 3700], [1, 0.4], [0.006, 0.004]) + bandnoise(m, 1000, 6000) * env_exp(m, 0.003) * 0.5, t, 0.5)
        place(y, bandnoise(int(0.06 * SR), 120, 600) * env_exp(int(0.06 * SR), 0.02) * 0.3, t + 0.03)
        t += 1 / rate
    return y


SPECS = [
    # (逻辑名称, 文件名, 生成函数, 立体声?, 说明)
    ("木梆子", "muBangzi_synth_a", lambda: gen_bangzi(0), False, "木梆子三击（模态合成）"),
    ("木梆子", "muBangzi_synth_b", lambda: gen_bangzi(1), False, "木梆子四击（模态合成）"),
    ("自行车铃", "bikeBell_synth_a", lambda: gen_bike_bell(0), False, "转铃（模态合成）"),
    ("自行车铃", "bikeBell_synth_b", lambda: gen_bike_bell(1), False, "转铃连响（模态合成）"),
    ("学校铃声", "schoolBell_synth", gen_hand_bell, False, "手摇铃（模态合成）"),
    ("挂钟", "clock_synth", gen_clock, False, "摆钟滴答（模态合成）"),
    ("算盘", "abacus_synth", gen_abacus, False, "算盘珠（模态合成）"),
    ("搪瓷盆", "enamel_synth", gen_enamel_basin, False, "搪瓷盆碰撞（模态合成）"),
    ("雨打瓦片", "rainTiles_synth", lambda: gen_rain(30, False), True, "小雨打瓦片（滤波噪声 + 随机雨滴）"),
    ("大雨", "heavyRain_synth", lambda: gen_rain(30, True), True, "大雨（滤波噪声 + 随机雨滴）"),
    ("风声", "wind_synth", lambda: gen_wind(30, False), True, "风声（调制滤波噪声）"),
    ("风吹树叶", "leaves_synth", lambda: gen_wind(30, True), True, "风吹树叶（调制滤波噪声）"),
    ("蟋蟀", "crickets_synth", gen_crickets, True, "夜晚蟋蟀（调幅正弦）"),
    ("蝉鸣", "cicadas_synth", gen_cicadas, True, "夏日蝉鸣（调幅噪声）"),
    ("收音机调台", "radioStatic_synth", gen_radio_static, False, "收音机调台杂音"),
    ("火车汽笛", "trainWhistle_synth", lambda: gen_steam_whistle(4.0, 392), False, "蒸汽汽笛（和弦 + 气流噪声）"),
    ("工厂汽笛", "factoryWhistle_synth", lambda: gen_steam_whistle(6.0, 262), False, "工厂汽笛（和弦 + 气流噪声）"),
    ("风箱", "bellows_synth", gen_bellows, False, "拉风箱（调制噪声）"),
    ("炉子", "stove_synth", gen_stove, False, "炉火噼啪（滤波噪声）"),
    ("缝纫机", "sewing_synth", gen_sewing, False, "脚踏缝纫机（模态合成）"),
]


def write_wav(path: Path, x: np.ndarray):
    x = np.asarray(x, np.float64)
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    ch = 2 if x.ndim == 2 else 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def main():
    sys.path.insert(0, str(ROOT / "backend"))
    from app.assets.catalog import get_catalog
    from app.assets.online import transcode, ffprobe_duration, vocab_props
    from app.assets.registry import AssetRecord, AssetSource

    force = "--force" in sys.argv
    reg = registry()
    cat = get_catalog()
    OUT.mkdir(parents=True, exist_ok=True)
    for key, fname, fn, stereo, desc in SPECS:
        aid = f"snd_synth_{fname}"
        dst = OUT / f"{fname}.mp3"
        if reg.get(aid) and dst.exists() and not force:
            continue
        vocab = cat.sound_vocab(key)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "x.wav"
            write_wav(wav, fn())
            transcode(wav, dst, not stereo, -26.0 if stereo else -20.0, 60.0)
        rec = AssetRecord(
            id=aid, kind="sound", name_zh=key, tags=[key] + (vocab.zh if vocab else []),
            synonyms=vocab.en if vocab else [], files=[dst.relative_to(ROOT).as_posix()],
            source=AssetSource(type="procedural", service="scripts/synth_sounds.py",
                               notes=f"程序合成占位：{desc}。有真实录音后建议替换。"),
            license="CC0-1.0（本项目自制）", copyright_status="public_domain", review_status="pending",
            sound=vocab_props(vocab, ffprobe_duration(dst)) if vocab else None)
        reg.register(rec)
        print("合成", key, "→", dst.name)


if __name__ == "__main__":
    main()
