// 设置页（会谈前后使用，会谈中不出现）：参与者、知情同意、实验条件、会谈时长、数据查看/导出/删除、“老家漫游”导出。
import "./setup.css";

const COND = { personal: "个人共建场景", generic: "通用场景", audio_only: "只听" } as const;
const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

async function api(path: string, opts: RequestInit = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail ?? msg; } catch { /* */ }
    throw new Error(msg);
  }
  return r.json();
}

function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

function fmtTime(ts: number | null): string {
  return ts ? new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false }) : "";
}

let regions: string[] = [];

async function loadStatus() {
  const s = await api("/api/status");
  regions = s.regions;
  const sel = $<HTMLSelectElement>("#new-p select[name=region_style]");
  sel.innerHTML = regions.map((r) => `<option>${esc(r)}</option>`).join("");
  $("#service").innerHTML = s.demo_mode
    ? `<b>演示模式</b>：没有配置百炼 API Key（或 DEMO_MODE=true），使用本地规则和占位画面，可以在会谈页的工作人员面板（按 S 键）里打字模拟老人说话。`
    : `<b>云端服务已连接</b>：大语言模型 ${esc(s.providers.llm)}；全景 ${esc(s.providers.panorama)}；深度 ${esc(s.providers.depth)}。`;
  $("#service").innerHTML += `　声音素材 ${s.sound_assets} 个（${s.sound_groups} 组）。`;
}

async function loadParticipants() {
  const ps = await api("/api/participants");
  const box = $("#plist");
  if (!ps.length) {
    box.innerHTML = `<p class="hint">还没有参与者。</p>`;
    return;
  }
  box.innerHTML = ps.map((p: any) => `
    <div class="participant" data-id="${esc(p.id)}">
      <div class="row head">
        <b class="pid">${esc(p.id)}</b>
        <select class="region">${regions.map((r) => `<option ${r === p.region_style ? "selected" : ""}>${esc(r)}</option>`).join("")}</select>
        <label class="consent"><input type="checkbox" class="consent-box" ${p.consent ? "checked" : ""}/>
          已获得本人和家属的知情同意（含录音和使用云端服务）</label>
      </div>
      <div class="row">
        <select class="cond">${Object.entries(COND).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>
        <label>时长 <input class="dur" type="number" min="5" max="90" value="30" /> 分钟</label>
        <button class="start" ${p.consent ? "" : "disabled title='请先勾选知情同意'"}>开始会谈</button>
        <button class="roam">导出“老家漫游”</button>
        <a class="btn" href="/api/participants/${encodeURIComponent(p.id)}/data.zip">导出全部数据</a>
        <button class="danger delete">删除全部数据</button>
      </div>
      <table class="sessions">
        <tr><th>会谈</th><th>条件</th><th>开始</th><th>状态</th><th></th></tr>
        ${(p.sessions as any[]).map((s) => `<tr><td><code>${esc(s.id)}</code></td><td>${esc(COND[s.condition as keyof typeof COND])}</td>
          <td>${esc(fmtTime(s.started_at))}</td><td>${esc({ created: "未开始", running: "进行中", ended: "已结束" }[s.status as string] ?? s.status)}</td>
          <td>${s.status !== "ended" ? `<a href="/?session=${esc(s.id)}" target="_blank">打开会谈页</a>` : ""}
              <a href="/api/sessions/${esc(s.id)}/report" target="_blank">报告</a>
              <a href="/api/sessions/${esc(s.id)}/report.csv">CSV</a>
              <a href="#" class="analyze" data-sid="${esc(s.id)}">重新分析</a></td></tr>`).join("")}
      </table>
    </div>`).join("");

  box.querySelectorAll<HTMLElement>(".participant").forEach((el) => {
    const pid = el.dataset.id!;
    el.querySelector<HTMLInputElement>(".consent-box")!.addEventListener("change", async (e) => {
      await api(`/api/participants/${pid}`, { method: "PATCH", body: JSON.stringify({ consent: (e.target as HTMLInputElement).checked }) });
      loadParticipants();
    });
    el.querySelector<HTMLSelectElement>(".region")!.addEventListener("change", async (e) => {
      await api(`/api/participants/${pid}`, { method: "PATCH", body: JSON.stringify({ region_style: (e.target as HTMLSelectElement).value }) });
    });
    el.querySelector(".start")!.addEventListener("click", async () => {
      try {
        const r = await api("/api/sessions", { method: "POST", body: JSON.stringify({
          participant_id: pid, condition: el.querySelector<HTMLSelectElement>(".cond")!.value,
          duration_min: Number(el.querySelector<HTMLInputElement>(".dur")!.value) }) });
        window.open(r.url, "_blank");
        loadParticipants();
      } catch (e) { alert(String(e)); }
    });
    el.querySelector(".delete")!.addEventListener("click", async () => {
      const typed = prompt(`将永久删除 ${pid} 的全部录音、文字、场景和日志，无法恢复。\n请输入编号 ${pid} 确认：`);
      if (typed !== pid) return;
      await api(`/api/participants/${pid}`, { method: "DELETE" });
      loadParticipants();
    });
    el.querySelector(".roam")!.addEventListener("click", () => openRoam(pid));
    el.querySelectorAll<HTMLAnchorElement>(".analyze").forEach((a) => a.addEventListener("click", async (e) => {
      e.preventDefault();
      a.textContent = "分析中…";
      try { await api(`/api/sessions/${a.dataset.sid}/analyze`, { method: "POST" }); a.textContent = "已更新"; }
      catch (err) { a.textContent = "失败：" + err; }
    }));
  });
}

async function openRoam(pid: string) {
  const ps = await api("/api/participants");
  const p = ps.find((x: any) => x.id === pid);
  const dlg = $<HTMLDialogElement>("#roam-dlg");
  const box = $("#roam-stories");
  $("#roam-result").textContent = "";
  if (!p?.scene_id) {
    box.innerHTML = `<p>这位参与者还没有个人共建的场景。</p>`;
    dlg.showModal();
    return;
  }
  const { manifest } = await api(`/api/scenes/${p.scene_id}`);
  box.innerHTML = manifest.stories.length ? manifest.stories.map((s: any) => `
    <label class="story"><input type="checkbox" value="${esc(s.id)}" ${s.include_in_export ? "checked" : ""}/>
      <b>${esc(s.title_zh)}</b><span>${esc(s.transcript)}</span></label>`).join("") : `<p class="hint">还没有故事，导出的网页里只有场景和声音。</p>`;
  dlg.showModal();
  $("#roam-go").onclick = async (e) => {
    e.preventDefault();
    const ids = [...box.querySelectorAll<HTMLInputElement>("input:checked")].map((i) => i.value);
    for (const s of manifest.stories) {
      await api(`/api/scenes/${p.scene_id}/stories/${s.id}`, { method: "PATCH", body: JSON.stringify({ include_in_export: ids.includes(s.id) }) });
    }
    $("#roam-result").textContent = "正在打包…";
    try {
      const r = await api(`/api/participants/${pid}/roam`, { method: "POST", body: JSON.stringify({ story_ids: ids }) });
      $("#roam-result").innerHTML = `完成（${r.size_mb} MB）：<a href="${esc(r.url)}" download>下载“老家漫游”网页</a>`;
    } catch (err) {
      $("#roam-result").textContent = "导出失败：" + err;
    }
  };
}

async function loadAssets() {
  const list = (await api("/api/assets")).filter((a: any) => a.kind === "sound" || a.kind === "music" || a.kind === "ir");
  list.sort((a: any, b: any) => (a.review_status === "pending" ? 0 : 1) - (b.review_status === "pending" ? 0 : 1) || a.name_zh.localeCompare(b.name_zh, "zh"));
  $("#alist").innerHTML = `<table class="assets"><tr><th>名称</th><th>来源</th><th>授权</th><th>试听</th><th>状态</th></tr>${list.map((a: any) => `
    <tr data-id="${esc(a.id)}" class="${a.sensitive ? "sensitive" : ""}"><td>${esc(a.name_zh)}${a.sensitive ? "（敏感，不会自动加入）" : ""}<br><small>${esc(a.id)}</small></td>
    <td>${a.source.url ? `<a href="${esc(a.source.url)}" target="_blank">${esc(a.source.site)}</a>` : esc(a.source.type)}<br><small>${esc(a.source.author ?? "")}</small></td>
    <td>${esc(a.license)}<br><small>${esc(a.attribution ?? "")}</small></td>
    <td><audio controls preload="none" src="${esc(a.urls[0])}"></audio></td>
    <td><select class="rs">${["pending", "approved", "rejected"].map((s) => `<option value="${s}" ${s === a.review_status ? "selected" : ""}>${{ pending: "待审核", approved: "通过", rejected: "不用" }[s]}</option>`).join("")}</select></td></tr>`).join("")}</table>`;
  $("#alist").querySelectorAll<HTMLSelectElement>(".rs").forEach((sel) => sel.addEventListener("change", async () => {
    const id = sel.closest("tr")!.getAttribute("data-id");
    await api(`/api/assets/${id}`, { method: "PATCH", body: JSON.stringify({ review_status: sel.value }) });
  }));
}

$("#new-p").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target as HTMLFormElement);
  try {
    await api("/api/participants", { method: "POST", body: JSON.stringify(Object.fromEntries(f)) });
    (e.target as HTMLFormElement).reset();
    loadParticipants();
  } catch (err) { alert(String(err)); }
});
$("#show-assets").addEventListener("click", (e) => {
  e.preventDefault();
  const sec = $("#assets");
  sec.hidden = !sec.hidden;
  if (!sec.hidden) loadAssets();
});

loadStatus().then(loadParticipants);
