// 时代声学效果链：老收音机 / 大喇叭（有线广播）/ 留声机。
// assets/irs/ 中有对应设备的冲激响应时用 ConvolverNode 卷积，否则用合成近似。
import { AcousticProfile } from "../shared/types";

export interface ProfileParams {
  lowHz: number;
  highHz: number;
  drive: number; // 软削波强度
  noiseDb: number; // 底噪电平
  wobbleDepth: number; // 信号起伏 / 转速抖动深度
  wobbleHz: number;
  echoDelayS: number;
  echoFeedback: number;
  echoMix: number;
  crackle: number; // 唱针爆豆声密度
  irMix: number; // 冲激响应混合比例（有 IR 时）
}

export const PROFILE_DEFAULTS: Record<Exclude<AcousticProfile, "clean">, ProfileParams> = {
  old_radio: { lowHz: 200, highHz: 5000, drive: 2.2, noiseDb: -42, wobbleDepth: 0.18, wobbleHz: 0.13, echoDelayS: 0, echoFeedback: 0, echoMix: 0, crackle: 0, irMix: 0.7 },
  loudspeaker_horn: { lowHz: 300, highHz: 4000, drive: 6, noiseDb: -48, wobbleDepth: 0.04, wobbleHz: 0.2, echoDelayS: 0.42, echoFeedback: 0.32, echoMix: 0.38, crackle: 0, irMix: 0.8 },
  gramophone: { lowHz: 160, highHz: 5500, drive: 1.6, noiseDb: -38, wobbleDepth: 0.0018, wobbleHz: 0.55, echoDelayS: 0, echoFeedback: 0, echoMix: 0, crackle: 0.6, irMix: 0.6 },
};

export interface EffectChain {
  input: AudioNode;
  output: AudioNode;
  dispose(): void;
}

function softClipCurve(k: number): Float32Array {
  const n = 2048;
  const curve = new Float32Array(n);
  const norm = Math.tanh(k);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    curve[i] = Math.tanh(k * x) / norm;
  }
  return curve;
}

const noiseCache = new WeakMap<BaseAudioContext, AudioBuffer>();
function pinkNoise(ctx: BaseAudioContext): AudioBuffer {
  let b = noiseCache.get(ctx);
  if (b) return b;
  const len = ctx.sampleRate * 4;
  b = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = b.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < len; i++) {
    const w = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
    d[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
    b6 = w * 0.115926;
  }
  noiseCache.set(ctx, b);
  return b;
}

/** 唱针爆豆声：稀疏的随机脉冲 + 轻微的 50Hz 电源哼声 */
function crackleBuffer(ctx: BaseAudioContext, density: number): AudioBuffer {
  const len = ctx.sampleRate * 6;
  const b = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = b.getChannelData(0);
  const perSecond = 6 + density * 40;
  for (let i = 0; i < len; i++) {
    if (Math.random() < perSecond / ctx.sampleRate) {
      const amp = (Math.random() ** 3) * 0.9 * (Math.random() < 0.5 ? -1 : 1);
      const dur = 4 + Math.floor(Math.random() * 30);
      for (let k = 0; k < dur && i + k < len; k++) d[i + k] += amp * Math.exp(-k / (dur / 3)) * (Math.random() * 0.6 + 0.4);
    }
    d[i] += Math.sin((2 * Math.PI * 50 * i) / ctx.sampleRate) * 0.004;
  }
  return b;
}

/** 合成近似的设备响应（没有真实 IR 时）：短小的箱体共鸣 */
export function syntheticDeviceIR(ctx: BaseAudioContext, profile: AcousticProfile): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * (profile === "loudspeaker_horn" ? 0.09 : 0.05));
  const b = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = b.getChannelData(0);
  d[0] = 1;
  const f = profile === "gramophone" ? 900 : profile === "loudspeaker_horn" ? 1400 : 650;
  for (let i = 1; i < len; i++) {
    const t = i / ctx.sampleRate;
    d[i] = (Math.random() * 2 - 1) * 0.25 * Math.exp(-t * 70) + Math.sin(2 * Math.PI * f * t) * 0.35 * Math.exp(-t * 55);
  }
  return b;
}

export function makeProfileChain(
  ctx: BaseAudioContext, profile: AcousticProfile, params?: Partial<ProfileParams>, ir?: AudioBuffer | null,
): EffectChain {
  const input = ctx.createGain();
  const output = ctx.createGain();
  if (profile === "clean") {
    input.connect(output);
    return { input, output, dispose: () => input.disconnect() };
  }
  const p = { ...PROFILE_DEFAULTS[profile], ...params };
  const nodes: AudioNode[] = [input];
  const sources: AudioScheduledSourceNode[] = [];

  // 1) 带通（两级高通 + 两级低通，坡度更陡）
  const hp1 = ctx.createBiquadFilter(); hp1.type = "highpass"; hp1.frequency.value = p.lowHz; hp1.Q.value = 0.7;
  const hp2 = ctx.createBiquadFilter(); hp2.type = "highpass"; hp2.frequency.value = p.lowHz; hp2.Q.value = 0.7;
  const lp1 = ctx.createBiquadFilter(); lp1.type = "lowpass"; lp1.frequency.value = p.highHz; lp1.Q.value = 0.7;
  const lp2 = ctx.createBiquadFilter(); lp2.type = "lowpass"; lp2.frequency.value = p.highHz; lp2.Q.value = 0.9;
  // 喇叭/收音机特有的中频突出
  const peak = ctx.createBiquadFilter(); peak.type = "peaking";
  peak.frequency.value = profile === "loudspeaker_horn" ? 1600 : profile === "old_radio" ? 1100 : 2200;
  peak.Q.value = 1.1; peak.gain.value = profile === "loudspeaker_horn" ? 7 : 4;
  // 2) 软削波
  const pre = ctx.createGain(); pre.gain.value = 1.0;
  const shaper = ctx.createWaveShaper(); shaper.curve = softClipCurve(p.drive); shaper.oversample = "2x";
  const post = ctx.createGain(); post.gain.value = 0.8;

  // 3) 转速抖动（留声机）：调制短延迟；其他设备几乎不需要
  let afterWobble: AudioNode = post;
  if (profile === "gramophone") {
    const dl = ctx.createDelay(0.05); dl.delayTime.value = 0.008;
    const lfo = ctx.createOscillator(); lfo.frequency.value = p.wobbleHz;
    const lfoGain = ctx.createGain(); lfoGain.gain.value = p.wobbleDepth;
    const flutter = ctx.createOscillator(); flutter.frequency.value = 6.5;
    const flutterGain = ctx.createGain(); flutterGain.gain.value = p.wobbleDepth * 0.15;
    lfo.connect(lfoGain).connect(dl.delayTime);
    flutter.connect(flutterGain).connect(dl.delayTime);
    lfo.start(); flutter.start();
    sources.push(lfo, flutter);
    post.connect(dl);
    afterWobble = dl;
    nodes.push(dl, lfoGain, flutterGain);
  }

  // 4) 信号起伏（收音机）：缓慢的增益调制
  const wob = ctx.createGain(); wob.gain.value = 1;
  if (profile !== "gramophone" && p.wobbleDepth > 0) {
    const lfo = ctx.createOscillator(); lfo.frequency.value = p.wobbleHz;
    const g = ctx.createGain(); g.gain.value = p.wobbleDepth;
    lfo.connect(g).connect(wob.gain);
    lfo.start();
    sources.push(lfo);
    nodes.push(g);
  }
  afterWobble.connect(wob);

  // 5) 设备响应：真实 IR 卷积 或 合成近似
  const conv = ctx.createConvolver();
  conv.normalize = true;
  conv.buffer = ir ?? syntheticDeviceIR(ctx, profile);
  const convWet = ctx.createGain(); convWet.gain.value = ir ? p.irMix : p.irMix * 0.35;
  const convDry = ctx.createGain(); convDry.gain.value = 1 - convWet.gain.value * 0.5;
  wob.connect(conv).connect(convWet).connect(output);
  wob.connect(convDry).connect(output);

  // 6) 大喇叭：远距离长延迟回声（楼宇间反射）
  if (p.echoDelayS > 0 && p.echoMix > 0) {
    const d1 = ctx.createDelay(2); d1.delayTime.value = p.echoDelayS;
    const fb = ctx.createGain(); fb.gain.value = p.echoFeedback;
    const echoLp = ctx.createBiquadFilter(); echoLp.type = "lowpass"; echoLp.frequency.value = 2200;
    const mix = ctx.createGain(); mix.gain.value = p.echoMix;
    wob.connect(d1).connect(echoLp).connect(fb).connect(d1);
    echoLp.connect(mix).connect(output);
    const d2 = ctx.createDelay(3); d2.delayTime.value = p.echoDelayS * 2.3;
    const mix2 = ctx.createGain(); mix2.gain.value = p.echoMix * 0.45;
    wob.connect(d2).connect(mix2).connect(output);
    nodes.push(d1, fb, echoLp, mix, d2, mix2);
  }

  // 7) 底噪（电流声 / 唱针沙沙声）
  const noise = ctx.createBufferSource(); noise.buffer = pinkNoise(ctx); noise.loop = true;
  const nbp = ctx.createBiquadFilter(); nbp.type = "bandpass"; nbp.frequency.value = (p.lowHz + p.highHz) / 2; nbp.Q.value = 0.4;
  const ng = ctx.createGain(); ng.gain.value = Math.pow(10, p.noiseDb / 20);
  noise.connect(nbp).connect(ng).connect(output);
  noise.start();
  sources.push(noise);
  if (p.crackle > 0) {
    const cr = ctx.createBufferSource(); cr.buffer = crackleBuffer(ctx, p.crackle); cr.loop = true;
    const cg = ctx.createGain(); cg.gain.value = 0.35 * p.crackle;
    const chp = ctx.createBiquadFilter(); chp.type = "highpass"; chp.frequency.value = 700;
    cr.connect(chp).connect(cg).connect(output);
    cr.start();
    sources.push(cr);
    nodes.push(cg, chp);
  }

  input.connect(hp1).connect(hp2).connect(peak).connect(pre).connect(shaper).connect(lp1).connect(lp2).connect(post);
  nodes.push(hp1, hp2, lp1, lp2, peak, pre, shaper, post, wob, conv, convWet, convDry, nbp, ng);

  return {
    input, output,
    dispose() {
      for (const s of sources) { try { s.stop(); } catch { /* 已停止 */ } }
      for (const n of nodes) n.disconnect();
      output.disconnect();
    },
  };
}

/** 合成的室外混响冲激响应（用于远处声音的空间感） */
export function outdoorReverbIR(ctx: BaseAudioContext, seconds = 1.8): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * seconds);
  const b = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = b.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      const t = i / ctx.sampleRate;
      // 早期反射（墙面）+ 指数衰减的扩散尾巴，高频衰减更快
      const early = (i % Math.floor(ctx.sampleRate * (0.011 + ch * 0.003)) === 0 && t < 0.08) ? 0.5 * Math.exp(-t * 30) : 0;
      d[i] = early + (Math.random() * 2 - 1) * Math.exp(-t * 3.2) * 0.35;
    }
    // 简单的一阶低通，让尾巴更暗
    let y = 0;
    for (let i = 0; i < len; i++) { y = y * 0.6 + d[i] * 0.4; d[i] = y; }
  }
  return b;
}
