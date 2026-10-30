// “老家漫游”：导出给家属的离线网页。数据和素材全部以 data URI 内嵌在 window.ROAM_DATA 中。
import { AudioEngine } from "../audio/engine";
import { AssetTable, SceneManifest, Story } from "../shared/types";
import { PanoramaViewer } from "../viewer/viewer";
import "./roam.css";

interface RoamData {
  manifest: SceneManifest;
  assets: AssetTable;
  title: string;
  credits: { name: string; source: string; license: string; attribution: string }[];
}
const data = (window as unknown as { ROAM_DATA?: RoamData }).ROAM_DATA;
const $ = (id: string) => document.getElementById(id)!;

if (!data) {
  $("sub").textContent = "（没有找到场景数据）";
} else {
  $("sub").textContent = data.title;
  const viewer = new PanoramaViewer($("app"), { showStoryLabels: true, onStoryClick: openStory, pixelRatioCap: 1.5 });
  const audio = new AudioEngine({ maxVolume: 0.8, panMode: "HRTF" });
  viewer.onFrame = () => audio.updateListener(viewer.look.yaw, viewer.look.pitch);
  viewer.look.opts.idleBeforeAutoS = 8;

  const m = data.manifest;
  viewer.setScene(m, data.assets).then(() => audio.setScene(m, data.assets, viewer.currentViewpoint, viewer.depthFor()));

  // 点位切换（家属用的网页可以有简单的按钮）
  const places = $("places");
  const renderPlaces = () => {
    places.innerHTML = "";
    for (const vp of m.viewpoints.filter((v) => v.panorama_path)) {
      const b = document.createElement("button");
      b.textContent = vp.name_zh;
      b.className = vp.id === viewer.currentViewpoint ? "on" : "";
      b.onclick = async () => {
        await viewer.goToViewpoint(vp.id);
        await audio.setScene(m, data.assets, viewer.currentViewpoint, viewer.depthFor());
        renderPlaces();
      };
      places.appendChild(b);
    }
    places.style.display = places.childElementCount > 1 ? "flex" : "none";
  };
  renderPlaces();

  $("go").addEventListener("click", async () => {
    await audio.resume();
    $("enter").classList.add("hidden");
  });

  // 手指 / 鼠标拖动转头
  const el = viewer.renderer.domElement;
  let last: { x: number; y: number } | null = null;
  let moved = 0;
  el.addEventListener("pointerdown", (e) => { last = { x: e.clientX, y: e.clientY }; moved = 0; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", (e) => {
    if (!last) return;
    const k = 90 / el.clientWidth;
    viewer.look.drag(-(e.clientX - last.x) * k, (e.clientY - last.y) * k);
    moved += Math.abs(e.clientX - last.x) + Math.abs(e.clientY - last.y);
    last = { x: e.clientX, y: e.clientY };
  });
  el.addEventListener("pointerup", () => { last = null; });
  // 拖动后不触发故事点击
  el.addEventListener("click", (e) => { if (moved > 8) e.stopImmediatePropagation(); }, true);

  function openStory(s: Story) {
    $("story-title").textContent = s.title_zh || "故事";
    $("story-text").textContent = s.transcript;
    const a = $("story-audio") as HTMLAudioElement;
    const url = s.audio_path ? data!.assets[`file:${s.audio_path}`]?.urls[0] : undefined;
    a.style.display = url ? "block" : "none";
    if (url) {
      a.src = url;
      audio.setDucked(true);
      a.play().catch(() => undefined);
      a.onended = a.onpause = () => audio.setDucked(false);
    }
    $("story").classList.add("open");
  }
  $("story-close").addEventListener("click", () => {
    ($("story-audio") as HTMLAudioElement).pause();
    $("story").classList.remove("open");
  });

  $("credits-body").innerHTML = "<h3>素材来源与授权</h3>" + data.credits.map((c) =>
    `<p><b>${c.name}</b>：${c.source}，${c.license}${c.attribution ? "，" + c.attribution : ""}</p>`).join("") +
    "<p>画面由 AI 根据讲述生成，不是真实拍摄的照片，不代表真实的人物和地点。</p>";
  $("credits-btn").addEventListener("click", () => $("credits").classList.add("open"));
  $("credits-close").addEventListener("click", () => $("credits").classList.remove("open"));
}
