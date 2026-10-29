// 调参试听页：选择声音 → 在“干净版”和各种时代效果之间 A/B 对比；调参数；试听距离与遮挡效果。
// 调好的参数保存在本机浏览器（localStorage），同一台电脑上的会谈页会直接使用。
import { EffectChain, PROFILE_DEFAULTS, ProfileParams, makeProfileChain, outdoorReverbIR } from "../audio/effects";
import { loadOverrides, saveOverrides } from "../audio/overrides";
import { AcousticProfile } from "../shared/types";
import "./tuner.css";

type Prof = Exclude<AcousticProfile, "clean">;
interface AssetItem { id: string; kind: string; name_zh: string; urls: string[]; tags: string[]; source: { type: string; site?: string }; license: string; review_status: string }

const ctx = new AudioContext();
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const clipSel = $("clip") as HTMLSelectElement;
const profSel = $("profile") as HTMLSelectElement;
const irSel = $("ir") as HTMLSelectElement;
const params = $("params");
const ab = $("ab");
const out = $("json");
const statusEl = $("status");

let assets: AssetItem[] = [];
let irs: AssetItem[] = [];
let overrides = loadOverrides();
let src: AudioBufferSourceNode | null = null;
let chainA: EffectChain | null = null; // 干净
let chainB: EffectChain | null = null; // 效果
const gA = ctx.createGain();
const gB = ctx.createGain();
const dist = ctx.createGain();
const occ = ctx.createBiquadFilter();
occ.type = "lowpass";
const air = ctx.createBiquadFilter();
air.type = "lowpass";
const panner = new PannerNode(ctx, { panningModel: "HRTF", distanceModel: "inverse", refDistance: 3, rolloffFactor: 1 });
const send = ctx.createGain();
const reverb = ctx.createConvolver();
reverb.buffer = outdoorReverbIR(ctx);
const master = ctx.createGain();
master.gain.value = 0.8;
const limiter = new DynamicsCompressorNode(ctx, { threshold: -10, ratio: 20, attack: 0.003, release: 0.25 });
gA.connect(dist);
gB.connect(dist);
dist.connect(occ).connect(air).connect(panner).connect(master);
air.connect(send).connect(reverb).connect(master);
master.connect(limiter).connect(ctx.destination);
let useB = true;
const bufCache = new Map<string, AudioBuffer>();

async function buffer(url: string): Promise<AudioBuffer> {
  if (!bufCache.has(url)) bufCache.set(url, await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer()));
  return bufCache.get(url)!;
}

function currentParams(): ProfileParams {
  const p = profSel.value as Prof;
  return { ...PROFILE_DEFAULTS[p], ...(overrides[p] ?? {}) };
}

function renderParams() {
  const p = profSel.value as Prof;
  const cur = currentParams();
  const ranges: Record<keyof ProfileParams, [number, number, number, string]> = {
    lowHz: [50, 800, 10, "低切 Hz"], highHz: [1500, 12000, 100, "高切 Hz"], drive: [0.5, 12, 0.1, "失真"],
    noiseDb: [-70, -20, 1, "底噪 dB"], wobbleDepth: [0, p === "gramophone" ? 0.006 : 0.5, p === "gramophone" ? 0.0001 : 0.01, "起伏/抖动深度"],
    wobbleHz: [0.02, 8, 0.01, "起伏/抖动频率 Hz"], echoDelayS: [0, 1.2, 0.01, "回声延迟 s"], echoFeedback: [0, 0.8, 0.01, "回声反馈"],
    echoMix: [0, 1, 0.01, "回声比例"], crackle: [0, 1, 0.01, "爆豆声"], irMix: [0, 1, 0.01, "冲激响应比例"],
  };
  params.innerHTML = "";
  for (const key of Object.keys(ranges) as (keyof ProfileParams)[]) {
    const [lo, hi, step, label] = ranges[key];
    const row = document.createElement("label");
    row.className = "param";
    row.innerHTML = `<span>${label}</span><input type="range" min="${lo}" max="${hi}" step="${step}" value="${cur[key]}"><output>${cur[key]}</output>`;
    const input = row.querySelector("input")!;
    input.addEventListener("input", () => {
      overrides[p] = { ...(overrides[p] ?? {}), [key]: Number(input.value) };
      row.querySelector("output")!.textContent = input.value;
      showJson();
    });
    input.addEventListener("change", () => rebuild());
    params.appendChild(row);
  }
  showJson();
}

function showJson() {
  out.textContent = JSON.stringify(overrides, null, 2);
}

async function irBuffer(): Promise<AudioBuffer | null> {
  const id = irSel.value;
  if (!id) return null;
  const a = irs.find((x) => x.id === id);
  return a ? buffer(a.urls[0]) : null;
}

async function rebuild() {
  chainA?.dispose();
  chainB?.dispose();
  chainA = makeProfileChain(ctx, "clean");
  chainB = makeProfileChain(ctx, profSel.value as Prof, currentParams(), await irBuffer());
  chainA.output.connect(gA);
  chainB.output.connect(gB);
  if (src) {
    src.disconnect();
    src.connect(chainA.input);
    src.connect(chainB.input);
  }
  setAB(useB);
}

function setAB(b: boolean) {
  useB = b;
  const t = ctx.currentTime;
  gA.gain.setTargetAtTime(b ? 0 : 1, t, 0.02);
  gB.gain.setTargetAtTime(b ? 1 : 0, t, 0.02);
  ab.textContent = b ? `正在听：B（${profSel.selectedOptions[0].text}）` : "正在听：A（干净版）";
  ab.classList.toggle("b", b);
}

async function play() {
  await ctx.resume();
  stop();
  const a = assets.find((x) => x.id === clipSel.value);
  if (!a) return;
  statusEl.textContent = "加载中…";
  try {
    const buf = await buffer(a.urls[0]);
    await rebuild();
    src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    src.connect(chainA!.input);
    src.connect(chainB!.input);
    src.start();
    statusEl.textContent = `${a.name_zh}｜${a.license}｜${a.source.site ?? a.source.type}`;
  } catch (e) {
    statusEl.textContent = "无法播放：" + e;
  }
}

function stop() {
  try { src?.stop(); } catch { /* */ }
  src = null;
}

function updateSpace() {
  const d = Number(($("distance") as HTMLInputElement).value);
  const o = Number(($("occlusion") as HTMLInputElement).value);
  const yaw = (Number(($("azimuth") as HTMLInputElement).value) * Math.PI) / 180;
  $("distance-v").textContent = `${d} 米`;
  $("occlusion-v").textContent = `${Math.round(o * 100)}%`;
  $("azimuth-v").textContent = `${Math.round((yaw * 180) / Math.PI)}°`;
  const t = ctx.currentTime;
  panner.positionX.setTargetAtTime(d * Math.sin(yaw), t, 0.05);
  panner.positionZ.setTargetAtTime(-d * Math.cos(yaw), t, 0.05);
  air.frequency.setTargetAtTime(Math.max(2600, 20000 * Math.exp(-d / 38)), t, 0.05);
  send.gain.setTargetAtTime(Math.min(0.55, 0.06 + d / 70), t, 0.05);
  occ.frequency.setTargetAtTime(20000 * Math.pow(900 / 20000, o), t, 0.05);
  dist.gain.setTargetAtTime(1 - 0.45 * o, t, 0.05);
}

async function init() {
  const all: AssetItem[] = await (await fetch("/api/assets")).json();
  assets = all.filter((a) => (a.kind === "sound" || a.kind === "music") && a.review_status !== "rejected");
  irs = all.filter((a) => a.kind === "ir");
  assets.sort((a, b) => a.name_zh.localeCompare(b.name_zh, "zh"));
  clipSel.innerHTML = assets.map((a) => `<option value="${a.id}">${a.name_zh}（${a.source.site ?? a.source.type}）</option>`).join("");
  irSel.innerHTML = `<option value="">合成近似（没有冲激响应）</option>` +
    irs.map((a) => `<option value="${a.id}">${a.name_zh}</option>`).join("");
  renderParams();
  updateSpace();
}

$("play").addEventListener("click", play);
$("stop").addEventListener("click", stop);
ab.addEventListener("click", () => setAB(!useB));
document.addEventListener("keydown", (e) => { if (e.code === "Space") { e.preventDefault(); setAB(!useB); } });
profSel.addEventListener("change", () => { renderParams(); rebuild(); });
irSel.addEventListener("change", () => rebuild());
clipSel.addEventListener("change", () => { if (src) play(); });
for (const id of ["distance", "occlusion", "azimuth"]) $(id).addEventListener("input", updateSpace);
$("save").addEventListener("click", () => { saveOverrides(overrides); statusEl.textContent = "已保存：本机的会谈页会使用这些参数"; });
$("reset").addEventListener("click", () => {
  delete overrides[profSel.value as Prof];
  saveOverrides(overrides);
  renderParams();
  rebuild();
});
init();
