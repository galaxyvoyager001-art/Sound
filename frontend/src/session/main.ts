// 会谈页：老人面前的大屏幕。不显示任何按钮和菜单；工作人员面板按 S 键（或连点左上角三次）呼出。
import { AudioEngine } from "../audio/engine";
import { ScenePayload } from "../shared/types";
import { PanoramaViewer } from "../viewer/viewer";
import { MicCapture } from "../voice/mic";
import "./session.css";

const params = new URLSearchParams(location.search);
const sessionId = params.get("session") ?? "";
const app = document.getElementById("app")!;
const startOverlay = document.getElementById("start")!;
const staff = document.getElementById("staff")!;
const log = document.getElementById("staff-log")!;
const statusEl = document.getElementById("staff-status")!;

const viewer = new PanoramaViewer(app);
viewer.look.opts.idleBeforeAutoS = 12; // 拖动或语音转向后，12 秒没有操作再恢复自动环顾
const audio = new AudioEngine({ maxVolume: Number(localStorage.getItem("maxVolume") ?? 0.8),
  panMode: (localStorage.getItem("panMode") as "HRTF" | "equalpower") ?? "equalpower" });
const mic = new MicCapture();
let ws: WebSocket | null = null;
let lastScene: ScenePayload | null = null;
let speechQueue: Promise<void> = Promise.resolve();
let ended = false;

function staffLog(text: string, cls = "") {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  log.prepend(div);
  while (log.childElementCount > 200) log.lastChild?.remove();
}

viewer.onFrame = () => audio.updateListener(viewer.look.yaw, viewer.look.pitch);
// 便于工作人员在浏览器控制台排查问题
(window as unknown as Record<string, unknown>).__tj = { viewer, audio, mic };

async function applyScene(p: ScenePayload) {
  lastScene = p;
  await viewer.setScene(p.manifest, p.assets, p.highlight ?? []);
  await audio.setScene(p.manifest, p.assets, viewer.currentViewpoint, viewer.depthFor());
}

function send(obj: unknown) {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function b64ToBuf(b64: string): ArrayBuffer {
  const s = atob(b64);
  const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return u.buffer;
}

/** 浏览器自带中文语音兜底（演示模式或语音合成服务不可用时） */
function speakWithBrowser(text: string): Promise<void> {
  return new Promise<void>((res) => {
    if (!("speechSynthesis" in window)) return res();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "zh-CN";
    u.rate = 0.82;
    u.pitch = 1.0;
    u.onend = () => res();
    u.onerror = () => res();
    audio.setDucked(true);
    speechSynthesis.speak(u);
  }).then(() => audio.setDucked(false));
}

function say(id: string, text: string, audioB64: string | null) {
  speechQueue = speechQueue.then(async () => {
    if (audio.quiet || ended && id !== "goodbye") return;
    mic.systemSpeaking = true;
    send({ type: "tts_state", id, playing: true, t: mic.timeS });
    try {
      if (audioB64) await audio.playSpeech(b64ToBuf(audioB64));
      else await speakWithBrowser(text);
    } finally {
      // 回声尾巴：再等 300 ms 才恢复正常门限
      await new Promise((r) => setTimeout(r, 300));
      mic.systemSpeaking = false;
      send({ type: "tts_state", id, playing: false, t: mic.timeS });
    }
  });
}

async function handleMessage(msg: any) {
  switch (msg.type) {
    case "scene":
      await applyScene(msg as ScenePayload);
      break;
    case "say":
      say(msg.id, msg.text, msg.audio_b64 ?? null);
      break;
    case "navigate":
      if (msg.viewpoint_id && msg.viewpoint_id !== viewer.currentViewpoint) {
        await viewer.goToViewpoint(msg.viewpoint_id, msg.yaw ?? undefined);
        if (lastScene) await audio.setScene(lastScene.manifest, lastScene.assets, viewer.currentViewpoint, viewer.depthFor());
        send({ type: "viewpoint_changed", viewpoint_id: viewer.currentViewpoint });
      } else if (typeof msg.yaw === "number") {
        viewer.look.lookAt(msg.yaw);
      } else if (typeof msg.turn === "number") {
        viewer.look.turnBy(msg.turn);
      }
      break;
    case "quiet":
      audio.hush();
      speechSynthesis?.cancel?.();
      staffLog("已安静");
      break;
    case "unquiet":
      audio.unhush();
      staffLog("恢复声音");
      break;
    case "music":
      if (msg.url) audio.playMusic(msg.url, msg.profile ?? "gramophone", msg.gain_db ?? -8);
      break;
    case "end":
      ended = true;
      staffLog("会谈结束");
      await speechQueue;
      if (msg.music_url) await audio.playMusic(msg.music_url, msg.profile ?? "gramophone", -8);
      audio.stopAll(4);
      break;
    case "transcript":
      staffLog(`${msg.speaker === "elder" ? "老人" : "陪聊"}${msg.final ? "" : "（识别中）"}：${msg.text}`, msg.speaker);
      break;
    case "status":
      statusEl.textContent = msg.text;
      break;
    case "staff_notice":
      staffLog("⚠ " + msg.text, "error");
      break;
    case "stop_speaking":
      audio.stopSpeech(0.15);
      speechSynthesis?.cancel?.();
      break;
  }
}

function connect(attempt = 0) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/session/${encodeURIComponent(sessionId)}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    attempt = 0;
    send({ type: "hello", t: mic.timeS, viewpoint_id: viewer.currentViewpoint });
    staffLog("已连接");
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") handleMessage(JSON.parse(e.data)).catch((err) => console.warn(err));
  };
  ws.onclose = () => {
    if (ended) return;
    staffLog("连接断开，正在重连…");
    // 断线期间画面和声音保持不变
    setTimeout(() => connect(attempt + 1), Math.min(10000, 1000 * 2 ** attempt));
  };
}

mic.onFrame = (pcm) => {
  if (ws?.readyState === WebSocket.OPEN) ws.send(pcm);
};
mic.onGate = (open) => send({ type: "gate", open, t: mic.timeS });
mic.onVad = (e) => {
  send({ type: "vad", speaking: e.speaking, t: e.t });
  if (e.speaking && mic.systemSpeaking) {
    // 老人插话：陪聊者立刻停下
    audio.stopSpeech(0.2);
    speechSynthesis?.cancel?.();
    send({ type: "barge_in", t: e.t });
  }
};
const levelEl = document.getElementById("staff-level")!;
mic.onLevel = (db, floor) => (levelEl.textContent = `麦克风 ${db.toFixed(0)} dB / 噪声底 ${floor.toFixed(0)} dB`);

startOverlay.addEventListener("click", async () => {
  startOverlay.classList.add("hidden");
  await audio.resume();
  try {
    await document.documentElement.requestFullscreen?.();
  } catch { /* 全屏失败不影响 */ }
  try {
    await mic.start();
  } catch (e) {
    staffLog("麦克风无法开启：" + e, "error");
  }
  connect();
});

// ---------------- 鼠标 / 触摸拖动环顾（工作人员测试用；老人不需要操作） ----------------
{
  const el = viewer.renderer.domElement;
  let last: { x: number; y: number } | null = null;
  el.addEventListener("pointerdown", (e) => {
    last = { x: e.clientX, y: e.clientY };
    el.setPointerCapture(e.pointerId);
    document.body.style.cursor = "grabbing";
  });
  el.addEventListener("pointermove", (e) => {
    if (!last) {
      document.body.style.cursor = "grab";
      return;
    }
    const k = 90 / el.clientWidth;
    viewer.look.drag(-(e.clientX - last.x) * k, (e.clientY - last.y) * k);
    last = { x: e.clientX, y: e.clientY };
  });
  const end = () => { last = null; document.body.style.cursor = ""; };
  el.addEventListener("pointerup", end);
  el.addEventListener("pointercancel", end);
  // 鼠标停止移动 3 秒后隐藏光标（屏幕上不显示多余的东西）
  let hideTimer = 0;
  window.addEventListener("pointermove", () => {
    clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => (document.body.style.cursor = "none"), 3000);
  });
}

// ---------------- 工作人员面板（老人看不到） ----------------
let corner = 0;
document.addEventListener("keydown", (e) => {
  if (e.key === "s" || e.key === "S") staff.classList.toggle("open");
});
document.getElementById("corner")!.addEventListener("click", () => {
  corner++;
  setTimeout(() => (corner = 0), 1200);
  if (corner >= 3) staff.classList.toggle("open");
});
const textInput = document.getElementById("staff-text") as HTMLInputElement;
document.getElementById("staff-send")!.addEventListener("click", () => {
  if (textInput.value.trim()) {
    send({ type: "staff_text", text: textInput.value.trim(), t: mic.timeS });
    textInput.value = "";
  }
});
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("staff-send")!.click();
});
document.getElementById("staff-end")!.addEventListener("click", () => send({ type: "staff", action: "end" }));
document.getElementById("staff-quiet")!.addEventListener("click", () => send({ type: "staff", action: "quiet" }));
document.getElementById("staff-unquiet")!.addEventListener("click", () => send({ type: "staff", action: "unquiet" }));
const vol = document.getElementById("staff-volume") as HTMLInputElement;
vol.value = String(Number(localStorage.getItem("maxVolume") ?? 0.8) * 100);
vol.addEventListener("input", () => {
  audio.setMaxVolume(Number(vol.value) / 100);
  localStorage.setItem("maxVolume", String(Number(vol.value) / 100));
});
const autoLook = document.getElementById("staff-autolook") as HTMLInputElement;
autoLook.checked = localStorage.getItem("autoLook") !== "0";
viewer.look.autoEnabled = autoLook.checked;
autoLook.addEventListener("change", () => {
  viewer.look.autoEnabled = autoLook.checked;
  localStorage.setItem("autoLook", autoLook.checked ? "1" : "0");
});
const pan = document.getElementById("staff-pan") as HTMLSelectElement;
pan.value = audio.panMode;
pan.addEventListener("change", () => {
  audio.setPanMode(pan.value as "HRTF" | "equalpower");
  localStorage.setItem("panMode", pan.value);
});
