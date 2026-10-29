// 空间音频引擎：只由场景清单驱动。
// 物件声源链：source → 时代效果链 → 遮挡低通 → 空气吸收低通 → 增益 → Panner(HRTF/equalpower) → 干声总线
//                                                                    └→ 混响发送（距离越远越多）
// 总线：干声 + 混响 + 环境底层 → 用户音量 → 闪避(陪聊说话时) → 安静开关 → 限幅器 → 音量上限 → 输出
import { DepthMap } from "../shared/depth";
import { AcousticProfile, AssetTable, SceneManifest, SceneObject, clamp, dbToGain, placementToVec } from "../shared/types";
import { EffectChain, makeProfileChain, outdoorReverbIR } from "./effects";
import { loadOverrides } from "./overrides";

type PanMode = "HRTF" | "equalpower";

export class BufferStore {
  private cache = new Map<string, Promise<AudioBuffer | null>>();
  constructor(private ctx: BaseAudioContext) {}
  get(url: string): Promise<AudioBuffer | null> {
    if (!this.cache.has(url)) {
      this.cache.set(url, fetch(url)
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
        .then((b) => this.ctx.decodeAudioData(b))
        .catch(() => null));
    }
    return this.cache.get(url)!;
  }
}

/** 无缝循环：相邻两次播放交叠并交叉淡化，避免录音首尾不接的“咔哒”声 */
class Looper {
  private timer: number | null = null;
  private live: { src: AudioBufferSourceNode; g: GainNode }[] = [];
  private stopped = false;
  constructor(private ctx: AudioContext, private buffer: AudioBuffer, private out: AudioNode, private xfade = 1.5) {}
  start() {
    this.schedule(this.ctx.currentTime + 0.05, Math.random() * Math.max(0, this.buffer.duration - this.xfade * 2));
  }
  private schedule(when: number, offset: number) {
    if (this.stopped) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    const g = this.ctx.createGain();
    const x = Math.min(this.xfade, this.buffer.duration / 4);
    const playDur = this.buffer.duration - offset;
    g.gain.setValueAtTime(0, when);
    g.gain.linearRampToValueAtTime(1, when + x);
    g.gain.setValueAtTime(1, when + playDur - x);
    g.gain.linearRampToValueAtTime(0, when + playDur);
    src.connect(g).connect(this.out);
    src.start(when, offset);
    src.stop(when + playDur + 0.05);
    this.live.push({ src, g });
    src.onended = () => { this.live = this.live.filter((l) => l.src !== src); g.disconnect(); };
    const next = when + playDur - x;
    const delayMs = Math.max(0, (next - this.ctx.currentTime - 0.5) * 1000);
    this.timer = window.setTimeout(() => this.schedule(next, 0), delayMs);
  }
  stop(fade = 0.6) {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    const t = this.ctx.currentTime;
    for (const l of this.live) {
      l.g.gain.cancelScheduledValues(t);
      l.g.gain.setValueAtTime(l.g.gain.value, t);
      l.g.gain.linearRampToValueAtTime(0, t + fade);
      try { l.src.stop(t + fade + 0.05); } catch { /* 已停止 */ }
    }
  }
}

class ObjectVoice {
  readonly chain: EffectChain;
  readonly occLP: BiquadFilterNode;
  readonly airLP: BiquadFilterNode;
  readonly gain: GainNode;
  readonly panner: PannerNode;
  readonly send: GainNode;
  private looper: Looper | null = null;
  private eventTimer: number | null = null;
  private lastVariant = -1;
  private disposed = false;
  key: string;

  constructor(private eng: AudioEngine, public obj: SceneObject, private buffers: AudioBuffer[], ir: AudioBuffer | null) {
    const ctx = eng.ctx;
    const s = obj.sound!;
    this.key = ObjectVoice.keyOf(obj);
    const ov = s.acoustic_profile === "clean" ? undefined : loadOverrides()[s.acoustic_profile];
    this.chain = makeProfileChain(ctx, s.acoustic_profile, ov, ir);
    this.occLP = ctx.createBiquadFilter();
    this.occLP.type = "lowpass";
    this.occLP.frequency.value = 20000;
    this.airLP = ctx.createBiquadFilter();
    this.airLP.type = "lowpass";
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;
    this.panner = new PannerNode(ctx, {
      panningModel: eng.panMode, distanceModel: "inverse", refDistance: s.ref_distance_m,
      rolloffFactor: s.rolloff, maxDistance: s.max_distance_m, coneInnerAngle: 360,
    });
    this.send = ctx.createGain();
    this.chain.output.connect(this.occLP).connect(this.airLP).connect(this.gain).connect(this.panner).connect(eng.dryBus);
    this.gain.connect(this.send).connect(eng.reverbBus);
    this.updatePlacement(obj);
    this.start();
  }

  static keyOf(o: SceneObject): string {
    const s = o.sound!;
    return [s.clip_id, s.variant_index, s.mode, s.acoustic_profile, s.interval_range_s.join("-")].join("|");
  }

  updatePlacement(obj: SceneObject) {
    this.obj = obj;
    const ctx = this.eng.ctx;
    const t = ctx.currentTime;
    const pl = obj.placement!;
    const [x, y, z] = placementToVec(pl, obj.height_m * 0.6);
    this.panner.positionX.setTargetAtTime(x, t, 0.3);
    this.panner.positionY.setTargetAtTime(y, t, 0.3);
    this.panner.positionZ.setTargetAtTime(z, t, 0.3);
    // 距离感：远处高频衰减、混响比例增加
    const d = pl.distance_m;
    this.airLP.frequency.setTargetAtTime(clamp(20000 * Math.exp(-d / 38), 2600, 20000), t, 0.4);
    this.send.gain.setTargetAtTime(clamp(0.06 + d / 70, 0.06, 0.55), t, 0.4);
    // 遮挡：深度图判断声源是否在墙或房子后面
    const occ = this.eng.depth ? this.eng.depth.occlusion(pl.yaw_deg, d, pl.elevation_m, obj.height_m) : pl.occluded;
    this.occLP.frequency.setTargetAtTime(20000 * Math.pow(900 / 20000, occ), t, 0.5);
    const target = dbToGain(obj.sound!.gain_db) * (1 - 0.45 * occ);
    this.gain.gain.setTargetAtTime(target, t, 0.6);
  }

  private pickBuffer(): AudioBuffer {
    const s = this.obj.sound!;
    if (this.buffers.length === 1) return this.buffers[0];
    if (s.mode === "loop") return this.buffers[s.variant_index % this.buffers.length];
    // 事件模式：从指定变体开始，之后随机轮换，不连续重复
    let i = this.lastVariant < 0 ? s.variant_index % this.buffers.length : Math.floor(Math.random() * this.buffers.length);
    if (i === this.lastVariant) i = (i + 1) % this.buffers.length;
    this.lastVariant = i;
    return this.buffers[i];
  }

  private start() {
    const s = this.obj.sound!;
    if (s.mode === "loop") {
      this.looper = new Looper(this.eng.ctx, this.pickBuffer(), this.chain.input);
      this.looper.start();
    } else {
      // 第一次事件稍早一点出现，让老人很快听到改动
      this.scheduleEvent(1 + Math.random() * 3);
    }
  }

  private scheduleEvent(afterS: number) {
    if (this.disposed) return;
    this.eventTimer = window.setTimeout(() => {
      if (this.disposed) return;
      const ctx = this.eng.ctx;
      const src = ctx.createBufferSource();
      src.buffer = this.pickBuffer();
      src.playbackRate.value = 0.97 + Math.random() * 0.06;
      const g = ctx.createGain();
      g.gain.value = dbToGain((Math.random() - 0.5) * 3);
      src.connect(g).connect(this.chain.input);
      src.start();
      src.onended = () => g.disconnect();
      const [lo, hi] = this.obj.sound!.interval_range_s;
      this.scheduleEvent(lo + Math.random() * (hi - lo));
    }, afterS * 1000);
  }

  setPanMode(m: PanMode) {
    this.panner.panningModel = m;
  }

  dispose(fade = 0.8) {
    this.disposed = true;
    if (this.eventTimer) clearTimeout(this.eventTimer);
    this.looper?.stop(fade);
    const t = this.eng.ctx.currentTime;
    this.gain.gain.setTargetAtTime(0, t, fade / 3);
    window.setTimeout(() => {
      this.chain.dispose();
      this.gain.disconnect();
      this.panner.disconnect();
      this.send.disconnect();
    }, (fade + 0.3) * 1000);
  }
}

class AmbienceVoice {
  private looper: Looper;
  readonly gain: GainNode;
  constructor(private eng: AudioEngine, public clipId: string, buffer: AudioBuffer, public gainDb: number) {
    this.gain = eng.ctx.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(eng.ambienceBus);
    this.looper = new Looper(eng.ctx, buffer, this.gain, 3);
    this.looper.start();
    this.setGain(gainDb, 3);
  }
  setGain(db: number, fadeS = 2) {
    this.gainDb = db;
    this.gain.gain.setTargetAtTime(dbToGain(db), this.eng.ctx.currentTime, fadeS / 3);
  }
  dispose(fadeS = 3) {
    this.looper.stop(fadeS);
    this.gain.gain.setTargetAtTime(0, this.eng.ctx.currentTime, fadeS / 3);
    window.setTimeout(() => this.gain.disconnect(), (fadeS + 0.5) * 1000);
  }
}

export interface EngineOptions {
  maxVolume: number; // 0~1，主输出上限
  panMode: PanMode;
}

export class AudioEngine {
  readonly ctx: AudioContext;
  readonly buffers: BufferStore;
  readonly dryBus: GainNode;
  readonly reverbBus: GainNode;
  readonly ambienceBus: GainNode;
  private sceneBus: GainNode;
  private userVolume: GainNode;
  private duck: GainNode;
  private quietGain: GainNode;
  private limiter: DynamicsCompressorNode;
  private cap: GainNode;
  private voiceBus: GainNode;
  private voices = new Map<string, ObjectVoice>();
  private ambience = new Map<string, AmbienceVoice>();
  private irs = new Map<string, AudioBuffer | null>();
  private currentSpeech: AudioBufferSourceNode | null = null;
  private musicChain: { src: AudioBufferSourceNode; chain: EffectChain; g: GainNode } | null = null;
  private assets: AssetTable = {};
  private syncToken = 0;
  depth: DepthMap | null = null;
  panMode: PanMode;
  quiet = false;

  constructor(opts: Partial<EngineOptions> = {}) {
    this.ctx = new AudioContext({ latencyHint: "playback" });
    this.buffers = new BufferStore(this.ctx);
    this.panMode = opts.panMode ?? "equalpower";
    const c = this.ctx;
    this.dryBus = c.createGain();
    this.reverbBus = c.createGain();
    this.ambienceBus = c.createGain();
    this.sceneBus = c.createGain();
    const reverb = c.createConvolver();
    reverb.buffer = outdoorReverbIR(c);
    const reverbOut = c.createGain();
    reverbOut.gain.value = 0.8;
    this.dryBus.connect(this.sceneBus);
    this.reverbBus.connect(reverb).connect(reverbOut).connect(this.sceneBus);
    this.ambienceBus.connect(this.sceneBus);
    this.userVolume = c.createGain();
    this.duck = c.createGain();
    this.quietGain = c.createGain();
    this.limiter = new DynamicsCompressorNode(c, { threshold: -10, knee: 6, ratio: 20, attack: 0.003, release: 0.25 });
    this.cap = c.createGain();
    this.cap.gain.value = opts.maxVolume ?? 0.8;
    this.sceneBus.connect(this.userVolume).connect(this.duck).connect(this.quietGain).connect(this.limiter);
    // 陪聊者的声音不受“闪避”影响，但同样经过限幅和音量上限
    this.voiceBus = c.createGain();
    this.voiceBus.connect(this.limiter);
    this.limiter.connect(this.cap).connect(c.destination);
  }

  async resume() {
    if (this.ctx.state !== "running") await this.ctx.resume();
  }

  setMaxVolume(v: number) {
    this.cap.gain.setTargetAtTime(clamp(v, 0, 1), this.ctx.currentTime, 0.1);
  }

  setUserVolumeDb(db: number) {
    this.userVolume.gain.setTargetAtTime(dbToGain(clamp(db, -30, 6)), this.ctx.currentTime, 0.3);
  }

  setPanMode(m: PanMode) {
    this.panMode = m;
    for (const v of this.voices.values()) v.setPanMode(m);
  }

  /** 每帧由相机朝向更新听者方向（听者在原点） */
  updateListener(yawDeg: number, pitchDeg: number) {
    const l = this.ctx.listener;
    const y = (yawDeg * Math.PI) / 180;
    const p = (pitchDeg * Math.PI) / 180;
    const fx = Math.cos(p) * Math.sin(y), fy = Math.sin(p), fz = -Math.cos(p) * Math.cos(y);
    const t = this.ctx.currentTime;
    if (l.forwardX) {
      l.forwardX.setTargetAtTime(fx, t, 0.05);
      l.forwardY.setTargetAtTime(fy, t, 0.05);
      l.forwardZ.setTargetAtTime(fz, t, 0.05);
      l.upX.value = 0; l.upY.value = 1; l.upZ.value = 0;
      l.positionX.value = 0; l.positionY.value = 0; l.positionZ.value = 0;
    } else {
      l.setOrientation(fx, fy, fz, 0, 1, 0);
    }
  }

  private async loadIR(profile: AcousticProfile): Promise<AudioBuffer | null> {
    if (profile === "clean") return null;
    if (!this.irs.has(profile)) {
      const url = this.assets[`ir:${profile}`]?.urls[0];
      this.irs.set(profile, url ? await this.buffers.get(url) : null);
    }
    return this.irs.get(profile) ?? null;
  }

  /** 场景变化 → 增量更新声源（只动有变化的） */
  async setScene(m: SceneManifest, assets: AssetTable, viewpointId: string | null, depth: DepthMap | null) {
    const token = ++this.syncToken;
    this.assets = { ...this.assets, ...assets };
    this.depth = depth;
    const active = m.objects.filter((o) => o.sound && o.placement && o.placement.viewpoint_id === viewpointId);
    const keep = new Set<string>();
    for (const o of active) {
      keep.add(o.id);
      const existing = this.voices.get(o.id);
      if (existing && existing.key === ObjectVoice.keyOf(o)) {
        existing.updatePlacement(o);
        continue;
      }
      const urls = this.assets[o.sound!.clip_id]?.urls ?? [];
      const bufs = (await Promise.all(urls.map((u) => this.buffers.get(u)))).filter((b): b is AudioBuffer => !!b);
      if (token !== this.syncToken) return; // 期间又有新场景
      if (!bufs.length) continue;
      const ir = await this.loadIR(o.sound!.acoustic_profile);
      existing?.dispose();
      this.voices.set(o.id, new ObjectVoice(this, o, bufs, ir));
    }
    for (const [id, v] of this.voices) {
      if (!keep.has(id)) {
        v.dispose();
        this.voices.delete(id);
      }
    }
    // 环境底层：按 clip 交叉淡入淡出
    const wanted = new Map(m.ambience.map((a) => [a.clip_id, a.gain_db]));
    for (const [clip, v] of this.ambience) {
      if (!wanted.has(clip)) {
        v.dispose();
        this.ambience.delete(clip);
      } else if (v.gainDb !== wanted.get(clip)) {
        v.setGain(wanted.get(clip)!);
      }
    }
    for (const [clip, db] of wanted) {
      if (this.ambience.has(clip)) continue;
      const url = this.assets[clip]?.urls[0];
      if (!url) continue;
      const buf = await this.buffers.get(url);
      if (token !== this.syncToken) return;
      if (buf && !this.ambience.has(clip)) this.ambience.set(clip, new AmbienceVoice(this, clip, buf, db));
    }
  }

  /** “太吵了 / 停一下”：0.4 秒内全部淡出（陪聊声音也停） */
  hush() {
    this.quiet = true;
    const t = this.ctx.currentTime;
    this.quietGain.gain.cancelScheduledValues(t);
    this.quietGain.gain.setValueAtTime(this.quietGain.gain.value, t);
    this.quietGain.gain.linearRampToValueAtTime(0, t + 0.4);
    this.stopSpeech(0.3);
    this.stopMusic(0.4);
  }

  unhush(fadeS = 3) {
    this.quiet = false;
    const t = this.ctx.currentTime;
    this.quietGain.gain.cancelScheduledValues(t);
    this.quietGain.gain.setValueAtTime(this.quietGain.gain.value, t);
    this.quietGain.gain.linearRampToValueAtTime(1, t + fadeS);
  }

  /** 陪聊者说话时场景声音降低 */
  setDucked(on: boolean) {
    this.duck.gain.setTargetAtTime(on ? dbToGain(-9) : 1, this.ctx.currentTime, on ? 0.15 : 0.8);
  }

  /** 播放陪聊者的语音；返回播放结束的 Promise */
  async playSpeech(data: ArrayBuffer): Promise<void> {
    this.stopSpeech(0.05);
    let buf: AudioBuffer;
    try {
      buf = await this.ctx.decodeAudioData(data.slice(0));
    } catch {
      return;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const g = this.ctx.createGain();
    g.gain.value = 1.0;
    src.connect(g).connect(this.voiceBus);
    this.currentSpeech = src;
    this.setDucked(true);
    return new Promise((res) => {
      src.onended = () => {
        if (this.currentSpeech === src) {
          this.currentSpeech = null;
          this.setDucked(false);
        }
        g.disconnect();
        res();
      };
      src.start();
    });
  }

  get speaking(): boolean {
    return this.currentSpeech !== null;
  }

  stopSpeech(fade = 0.15) {
    const s = this.currentSpeech;
    if (!s) return;
    try { s.stop(this.ctx.currentTime + fade); } catch { /* 已停止 */ }
  }

  /** 收尾的老歌 / 试听：经过对应的时代效果链，不做空间化 */
  async playMusic(url: string, profile: AcousticProfile = "gramophone", gainDb = -6): Promise<void> {
    this.stopMusic(0.5);
    const buf = await this.buffers.get(url);
    if (!buf) return;
    const ir = await this.loadIR(profile);
    const chain = makeProfileChain(this.ctx, profile, profile === "clean" ? undefined : loadOverrides()[profile], ir);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const g = this.ctx.createGain();
    g.gain.value = 0;
    g.gain.setTargetAtTime(dbToGain(gainDb), this.ctx.currentTime, 1);
    src.connect(chain.input);
    chain.output.connect(g).connect(this.voiceBus);
    src.start();
    this.musicChain = { src, chain, g };
    return new Promise((res) => (src.onended = () => res()));
  }

  stopMusic(fade = 1) {
    const m = this.musicChain;
    if (!m) return;
    this.musicChain = null;
    m.g.gain.setTargetAtTime(0, this.ctx.currentTime, fade / 3);
    window.setTimeout(() => { try { m.src.stop(); } catch { /* */ } m.chain.dispose(); m.g.disconnect(); }, fade * 1000 + 300);
  }

  stopAll(fade = 1) {
    for (const v of this.voices.values()) v.dispose(fade);
    this.voices.clear();
    for (const a of this.ambience.values()) a.dispose(fade);
    this.ambience.clear();
    this.stopMusic(fade);
  }
}
