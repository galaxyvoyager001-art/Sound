// 2.5D 全景 + 分层物件的渲染器。只读场景清单；所有外部资源加载失败都静默降级。
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { DepthMap } from "../shared/depth";
import {
  AssetTable, Landmark, SceneManifest, SceneObject, Story, Viewpoint, dirToVec, normYaw, placementToVec,
} from "../shared/types";
import { Grade, computeGrade, lerpGrade } from "./grade";
import { LookController } from "./look";
import {
  BILLBOARD_FRAG, BILLBOARD_VERT, GLOW_FRAG, PANO_FRAG, PANO_VERT, PRECIP_VERT, RAIN_FRAG, SHADOW_FRAG,
  SNOW_FRAG, VIGNETTE_FRAG,
} from "./shaders";

const PAPER = new THREE.Color(0.93, 0.88, 0.8);

function gradeUniforms() {
  return {
    uExposure: { value: 1 }, uTint: { value: new THREE.Vector3(1, 1, 1) }, uSaturation: { value: 1 },
    uContrast: { value: 1 }, uFog: { value: 0 }, uFogColor: { value: new THREE.Vector3(0.9, 0.88, 0.8) },
    uSkyTint: { value: new THREE.Vector3(1, 1, 1) }, uFade: { value: 1 },
    uFadeColor: { value: new THREE.Vector3(PAPER.r, PAPER.g, PAPER.b) },
  };
}

function solidTexture(r: number, g: number, b: number): THREE.DataTexture {
  const t = new THREE.DataTexture(new Uint8Array([r, g, b, 255]), 1, 1);
  t.needsUpdate = true;
  return t;
}

function depthTextureFromMap(map: DepthMap): THREE.DataTexture {
  const { width, height, data } = map;
  const arr = new Uint8Array(width * height * 4);
  const k = 255 / Math.log(200 / 0.5);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = Math.round(Math.log(data[y * width + x] / 0.5) * k);
      const i = ((height - 1 - y) * width + x) * 4; // DataTexture 不翻转，手动翻转行
      arr[i] = arr[i + 1] = arr[i + 2] = v;
      arr[i + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(arr, width, height);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.wrapS = THREE.RepeatWrapping;
  t.needsUpdate = true;
  return t;
}

interface ObjectNode {
  id: string;
  group: THREE.Group;
  material?: THREE.ShaderMaterial;
  shadow?: THREE.Mesh;
  glow?: THREE.Mesh;
  key: string; // 用于判断是否需要重建
  highlightUntil: number;
  opacity: number;
  targetOpacity: number;
}

export interface ViewerOptions {
  showStoryLabels?: boolean;
  onStoryClick?: (story: Story) => void;
  pixelRatioCap?: number;
}

export class PanoramaViewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly look = new LookController();
  private sphere: THREE.Mesh;
  private sphereMat: THREE.ShaderMaterial;
  private objectsRoot = new THREE.Group();
  private storiesRoot = new THREE.Group();
  private nodes = new Map<string, ObjectNode>();
  private storyNodes = new Map<string, THREE.Object3D>();
  private texLoader = new THREE.TextureLoader();
  private gltfLoader = new GLTFLoader();
  private texCache = new Map<string, Promise<THREE.Texture>>();
  private depthMaps = new Map<string, DepthMap>();
  private currentDepth: DepthMap = DepthMap.geometric();
  private currentDepthTex: THREE.Texture;
  private rain: THREE.Points;
  private snow: THREE.Points;
  private vignette: THREE.Mesh;
  private gradeNow: Grade = computeGrade("上午", "晴", "夏");
  private gradeFrom: Grade = this.gradeNow;
  private gradeTo: Grade = this.gradeNow;
  private gradeT = 1;
  private fade = 1; // 开场从纸色淡入
  private fadeTarget = 0;
  private clock = new THREE.Clock();
  private manifest: SceneManifest | null = null;
  private assets: AssetTable = {};
  currentViewpoint: string | null = null;
  private loadedPanorama: string | null = null;
  private switching = false;
  private raycaster = new THREE.Raycaster();
  private opts: ViewerOptions;
  onFrame?: (dt: number) => void;

  constructor(private container: HTMLElement, opts: ViewerOptions = {}) {
    this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, opts.pixelRatioCap ?? 2));
    this.renderer.setClearColor(PAPER);
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(68, 1, 0.1, 1000);
    this.camera.rotation.order = "YXZ";

    const placeholder = solidTexture(236, 224, 204);
    this.currentDepthTex = depthTextureFromMap(this.currentDepth);
    this.sphereMat = new THREE.ShaderMaterial({
      uniforms: {
        ...gradeUniforms(), uTexA: { value: placeholder }, uTexB: { value: placeholder },
        uDepthA: { value: this.currentDepthTex }, uDepthB: { value: this.currentDepthTex }, uMix: { value: 0 },
      },
      vertexShader: PANO_VERT, fragmentShader: PANO_FRAG, side: THREE.BackSide, depthWrite: false,
    });
    this.sphere = new THREE.Mesh(new THREE.SphereGeometry(500, 96, 64), this.sphereMat);
    this.sphere.renderOrder = -10;
    this.scene.add(this.sphere, this.objectsRoot, this.storiesRoot);

    this.rain = this.makePrecip(5000, RAIN_FRAG, 14, 1.6);
    this.snow = this.makePrecip(2500, SNOW_FRAG, 1.1, 2.2);
    this.scene.add(this.rain, this.snow);

    const vg = new THREE.ShaderMaterial({
      uniforms: { uVignette: { value: 0.35 }, uGrain: { value: 0.035 }, uTime: { value: 0 } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: VIGNETTE_FRAG, transparent: true, depthTest: false, depthWrite: false,
    });
    this.vignette = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), vg);
    this.vignette.frustumCulled = false;
    this.vignette.renderOrder = 100;
    this.scene.add(this.vignette);

    window.addEventListener("resize", () => this.resize());
    this.resize();
    if (opts.onStoryClick) this.renderer.domElement.addEventListener("click", (e) => this.handleClick(e));
    this.renderer.setAnimationLoop(() => this.frame());
  }

  private makePrecip(n: number, frag: string, speed: number, size: number): THREE.Points {
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 40;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 24;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 40;
      seed[i] = Math.random();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));
    const m = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uSpeed: { value: speed }, uSize: { value: size }, uOpacity: { value: 0 } },
      vertexShader: PRECIP_VERT, fragmentShader: frag, transparent: true, depthWrite: false,
    });
    const p = new THREE.Points(g, m);
    p.frustumCulled = false;
    p.visible = false;
    return p;
  }

  resize() {
    const w = this.container.clientWidth || window.innerWidth;
    const h = this.container.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    // 竖屏手机上加大视角，避免只看到很窄的一条
    this.camera.fov = w / h < 1 ? 80 : 68;
    this.camera.updateProjectionMatrix();
  }

  // ---------------- 资源 ----------------
  private loadTexture(url: string, color = true): Promise<THREE.Texture> {
    if (!this.texCache.has(url)) {
      this.texCache.set(url, new Promise((resolve, reject) => {
        this.texLoader.load(url, (t) => {
          if (color) t.colorSpace = THREE.SRGBColorSpace;
          t.generateMipmaps = false; // 全景接缝处 mipmap 会出现竖线
          t.minFilter = THREE.LinearFilter;
          t.wrapS = THREE.RepeatWrapping;
          t.anisotropy = 4;
          resolve(t);
        }, undefined, reject);
      }));
    }
    return this.texCache.get(url)!;
  }

  private async loadDepth(vp: Viewpoint): Promise<DepthMap> {
    const url = vp.depth_path ? this.assets[`file:${vp.depth_path}`]?.urls[0] : undefined;
    if (!url) return DepthMap.geometric();
    if (!this.depthMaps.has(url)) {
      try {
        this.depthMaps.set(url, await DepthMap.load(url));
      } catch {
        return DepthMap.geometric();
      }
    }
    return this.depthMaps.get(url)!;
  }

  depthFor(): DepthMap {
    return this.currentDepth;
  }

  // ---------------- 场景更新 ----------------
  async setScene(manifest: SceneManifest, assets: AssetTable, highlight: string[] = []) {
    const prevVp = this.currentViewpoint;
    this.manifest = manifest;
    this.assets = { ...this.assets, ...assets };
    if (!this.currentViewpoint || !manifest.viewpoints.find((v) => v.id === this.currentViewpoint)) {
      this.currentViewpoint = manifest.entry_viewpoint ?? manifest.viewpoints[0]?.id ?? null;
    }
    this.setGrade(manifest.time_of_day, manifest.weather, manifest.season);
    const vp = manifest.viewpoints.find((v) => v.id === this.currentViewpoint);
    if (vp && vp.panorama_path && vp.panorama_path !== this.loadedPanorama && !this.switching) {
      // 同一点位换了新全景（重绘完成）→ 交叉淡入；首次 → 从纸色淡入
      await this.crossfadeTo(vp, prevVp === vp.id && this.loadedPanorama !== null);
    }
    this.syncObjects(highlight);
    this.syncStories();
    this.updateInterest();
  }

  private async crossfadeTo(vp: Viewpoint, crossfade: boolean) {
    const url = this.assets[`file:${vp.panorama_path}`]?.urls[0];
    if (!url) return;
    let tex: THREE.Texture;
    try {
      tex = await this.loadTexture(url);
    } catch {
      return; // 加载失败：保持当前画面
    }
    const depth = await this.loadDepth(vp);
    const depthTex = depthTextureFromMap(depth);
    const u = this.sphereMat.uniforms;
    this.loadedPanorama = vp.panorama_path;
    if (crossfade) {
      u.uTexB.value = tex;
      u.uDepthB.value = depthTex;
      await this.animate(2.5, (t) => (u.uMix.value = t * t * (3 - 2 * t)));
      u.uTexA.value = tex;
      u.uDepthA.value = depthTex;
      u.uMix.value = 0;
    } else {
      u.uTexA.value = tex;
      u.uDepthA.value = depthTex;
      u.uMix.value = 0;
      this.fadeTarget = 0;
    }
    this.currentDepth = depth;
    this.currentDepthTex = depthTex;
    for (const n of this.nodes.values()) if (n.material) n.material.uniforms.uDepth.value = depthTex;
  }

  /** 切换点位：淡出到纸色 → 换图和物件 → 淡入（不做连续行走） */
  async goToViewpoint(viewpointId: string, lookYaw?: number) {
    if (!this.manifest || this.switching || viewpointId === this.currentViewpoint) {
      if (lookYaw !== undefined) this.look.lookAt(lookYaw);
      return;
    }
    const vp = this.manifest.viewpoints.find((v) => v.id === viewpointId);
    if (!vp) return;
    this.switching = true;
    this.fadeTarget = 1;
    await this.waitFade();
    this.currentViewpoint = viewpointId;
    this.loadedPanorama = null;
    if (vp.panorama_path) await this.crossfadeTo(vp, false);
    this.look.jumpTo(lookYaw ?? 0);
    this.syncObjects([]);
    this.syncStories();
    this.updateInterest();
    this.fadeTarget = 0;
    this.switching = false;
  }

  private waitFade(): Promise<void> {
    return new Promise((res) => {
      const check = () => (Math.abs(this.fade - this.fadeTarget) < 0.01 ? res() : requestAnimationFrame(check));
      check();
    });
  }

  private animate(seconds: number, step: (t: number) => void): Promise<void> {
    return new Promise((res) => {
      const t0 = performance.now();
      const tick = () => {
        const t = Math.min(1, (performance.now() - t0) / (seconds * 1000));
        step(t);
        if (t < 1) requestAnimationFrame(tick);
        else res();
      };
      tick();
    });
  }

  setGrade(time: string, weather: string, season: string) {
    const g = computeGrade(time, weather, season);
    if (JSON.stringify(g) === JSON.stringify(this.gradeTo)) return;
    this.gradeFrom = this.gradeNow;
    this.gradeTo = g;
    this.gradeT = 0;
  }

  private objectKey(o: SceneObject): string {
    const p = o.placement;
    return [o.asset_id, o.model_asset_id, o.asset_status, p?.yaw_deg, p?.distance_m, p?.elevation_m, o.height_m].join("|");
  }

  private syncObjects(highlight: string[]) {
    if (!this.manifest) return;
    const visible = this.manifest.condition === "audio_only" ? [] :
      this.manifest.objects.filter((o) => o.placement && o.placement.viewpoint_id === this.currentViewpoint);
    const keep = new Set(visible.map((o) => o.id));
    for (const [id, n] of this.nodes) {
      if (!keep.has(id)) {
        n.targetOpacity = 0; // 淡出后移除
        setTimeout(() => {
          if (this.nodes.get(id) === n && n.targetOpacity === 0) {
            this.objectsRoot.remove(n.group);
            this.nodes.delete(id);
          }
        }, 1500);
      }
    }
    const now = performance.now() / 1000;
    for (const o of visible) {
      const key = this.objectKey(o);
      let node = this.nodes.get(o.id);
      if (!node || node.key !== key) {
        const old = node;
        node = this.buildObject(o, key);
        if (old) {
          node.opacity = old.opacity * 0.3;
          this.objectsRoot.remove(old.group);
        }
        this.nodes.set(o.id, node);
        this.objectsRoot.add(node.group);
      }
      node.targetOpacity = 1;
      if (highlight.includes(o.id)) node.highlightUntil = now + 4;
    }
  }

  private buildObject(o: SceneObject, key: string): ObjectNode {
    const group = new THREE.Group();
    const node: ObjectNode = { id: o.id, group, key, highlightUntil: 0, opacity: 0, targetOpacity: 1 };
    const pl = o.placement!;
    const [x, y, z] = placementToVec(pl);
    group.position.set(x, y, z);
    group.lookAt(0, y, 0); // 竖直 billboard，始终朝向观看者（相机只转不动，所以只需设置一次）

    // 地面接触阴影
    if (pl.elevation_m < 0.3) {
      const sm = new THREE.ShaderMaterial({ uniforms: { uOpacity: { value: 0 } }, vertexShader:
        "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
        fragmentShader: SHADOW_FRAG, transparent: true, depthWrite: false });
      const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), sm);
      shadow.rotation.x = -Math.PI / 2;
      shadow.position.y = 0.01;
      shadow.scale.set(o.height_m * 0.9, o.height_m * 0.35, 1);
      group.add(shadow);
      node.shadow = shadow;
    }

    const glb = o.model_asset_id ? this.assets[o.model_asset_id] : undefined;
    const img = o.asset_id ? this.assets[o.asset_id] : undefined;
    if (glb?.urls[0]) {
      this.gltfLoader.load(glb.urls[0], (gltf) => {
        const m = gltf.scene;
        const box = new THREE.Box3().setFromObject(m);
        const s = o.height_m / Math.max(0.01, box.max.y - box.min.y);
        m.scale.setScalar(s);
        m.position.y = -box.min.y * s;
        group.add(m);
      }, undefined, () => this.addBillboard(node, o, img));
      const amb = new THREE.HemisphereLight(0xfff2dd, 0x806a55, 1.2);
      group.add(amb);
    } else if (img?.urls[0] && o.asset_status === "ready") {
      this.addBillboard(node, o, img);
    } else {
      this.addGlow(node, o, new THREE.Color(1.0, 0.86, 0.62), 0.45); // 生成中：柔和光点
    }
    return node;
  }

  private addBillboard(node: ObjectNode, o: SceneObject, img?: { urls: string[]; width?: number; height?: number }) {
    if (!img?.urls[0]) return;
    const aspect = img.width && img.height ? img.width / img.height : 1;
    const h = o.height_m;
    const geo = new THREE.PlaneGeometry(h * aspect, h);
    geo.translate(0, h / 2, 0);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        ...gradeUniforms(), uMap: { value: solidTexture(0, 0, 0) }, uDepth: { value: this.currentDepthTex },
        uOpacity: { value: 0 }, uHighlight: { value: 0 }, uTime: { value: 0 }, uUseDepth: { value: 1 },
      },
      vertexShader: BILLBOARD_VERT, fragmentShader: BILLBOARD_FRAG, transparent: true, depthWrite: false,
    });
    this.loadTexture(img.urls[0]).then((t) => {
      t.generateMipmaps = true;
      t.minFilter = THREE.LinearMipmapLinearFilter;
      t.wrapS = THREE.ClampToEdgeWrapping;
      t.needsUpdate = true;
      mat.uniforms.uMap.value = t;
    }).catch(() => undefined);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = Math.round(-(o.placement?.distance_m ?? 10) * 10); // 远的先画
    node.group.add(mesh);
    node.material = mat;
  }

  private addGlow(node: ObjectNode, o: SceneObject, color: THREE.Color, size: number) {
    const m = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: color }, uOpacity: { value: 0 }, uTime: { value: 0 }, uPhase: { value: Math.random() * 6 } },
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: GLOW_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const s = size * Math.max(1, (o.placement?.distance_m ?? 5) / 8);
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(s, s), m);
    mesh.position.y = Math.min(o.height_m * 0.6, 1.2);
    node.group.add(mesh);
    node.glow = mesh;
  }

  // ---------------- 故事标记 ----------------
  private anchorPosition(story: Story): THREE.Vector3 | null {
    const m = this.manifest!;
    if (story.anchor.kind === "object") {
      const o = m.objects.find((x) => x.id === story.anchor.id);
      if (!o?.placement || o.placement.viewpoint_id !== this.currentViewpoint) return null;
      const [x, y, z] = placementToVec(o.placement, o.height_m + 0.35);
      return new THREE.Vector3(x, y, z);
    }
    if (story.anchor.kind === "landmark") {
      const l = m.landmarks.find((x: Landmark) => x.id === story.anchor.id);
      if (!l || l.viewpoint_id !== this.currentViewpoint) return null;
      const [x, y, z] = dirToVec(l.yaw_deg, 2, Math.min(l.distance_m, 12));
      return new THREE.Vector3(x, y, z);
    }
    if (story.anchor.id !== this.currentViewpoint) return null;
    const idx = m.stories.filter((s) => s.anchor.kind === "viewpoint" && s.anchor.id === story.anchor.id).indexOf(story);
    const [x, y, z] = dirToVec(-30 + idx * 25, 6, 8);
    return new THREE.Vector3(x, y, z);
  }

  private syncStories() {
    if (!this.manifest) return;
    for (const n of this.storyNodes.values()) this.storiesRoot.remove(n);
    this.storyNodes.clear();
    for (const s of this.manifest.stories) {
      const pos = this.anchorPosition(s);
      if (!pos) continue;
      const g = new THREE.Group();
      g.position.copy(pos);
      g.lookAt(0, pos.y, 0);
      const size = 0.35 * Math.max(1, pos.length() / 6);
      const m = new THREE.ShaderMaterial({
        uniforms: { uColor: { value: new THREE.Color(1.0, 0.78, 0.45) }, uOpacity: { value: 0.9 }, uTime: { value: 0 },
          uPhase: { value: Math.random() * 6 } },
        vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
        fragmentShader: GLOW_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), m);
      mesh.userData.story = s;
      g.add(mesh);
      if (this.opts.showStoryLabels && s.title_zh) g.add(this.makeLabel(s.title_zh, size));
      this.storiesRoot.add(g);
      this.storyNodes.set(s.id, g);
    }
  }

  private makeLabel(text: string, size: number): THREE.Mesh {
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d")!;
    const fs = 44;
    ctx.font = `${fs}px "PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif`;
    const w = Math.ceil(ctx.measureText(text).width) + 40;
    c.width = w;
    c.height = fs + 30;
    ctx.font = `${fs}px "PingFang SC","Noto Sans SC","Microsoft YaHei",sans-serif`;
    ctx.fillStyle = "rgba(60,40,20,0.55)";
    ctx.beginPath();
    ctx.roundRect(0, 0, w, c.height, 18);
    ctx.fill();
    ctx.fillStyle = "#fff6e6";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 20, c.height / 2 + 2);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    const h = size * 0.55;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry((h * w) / c.height, h),
      new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false }));
    mesh.position.y = size * 0.8;
    mesh.userData.story = undefined;
    return mesh;
  }

  private handleClick(e: MouseEvent) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const p = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(p, this.camera);
    const hits = this.raycaster.intersectObjects(this.storiesRoot.children, true);
    for (const h of hits) {
      let o: THREE.Object3D | null = h.object;
      while (o && !o.userData.story) o = o.parent;
      const s = o?.userData.story as Story | undefined;
      if (s) {
        this.opts.onStoryClick?.(s);
        return;
      }
    }
  }

  private updateInterest() {
    if (!this.manifest) return;
    const yaws = this.manifest.objects
      .filter((o) => o.placement?.viewpoint_id === this.currentViewpoint && (o.sound || this.manifest!.stories.some((s) => s.anchor.id === o.id)))
      .map((o) => o.placement!.yaw_deg);
    this.look.setInterest(yaws);
  }

  /** 语音导航：“看看巷口” → 解析到方位角 */
  yawOfLandmark(landmarkId: string): number | null {
    const l = this.manifest?.landmarks.find((x) => x.id === landmarkId);
    return l ? normYaw(l.yaw_deg) : null;
  }

  // ---------------- 渲染循环 ----------------
  private frame() {
    const dt = this.clock.getDelta();
    const now = performance.now() / 1000;
    this.look.update(dt, now);
    this.camera.rotation.set((this.look.pitch * Math.PI) / 180, (-this.look.yaw * Math.PI) / 180, 0);

    // 淡入淡出
    const fadeSpeed = this.fadeTarget > this.fade ? 1.6 : 0.9;
    this.fade += Math.sign(this.fadeTarget - this.fade) * Math.min(Math.abs(this.fadeTarget - this.fade), fadeSpeed * dt);
    if (this.gradeT < 1) {
      this.gradeT = Math.min(1, this.gradeT + dt / 3);
      this.gradeNow = lerpGrade(this.gradeFrom, this.gradeTo, this.gradeT);
    }
    const g = this.gradeNow;
    const apply = (u: Record<string, THREE.IUniform>) => {
      u.uExposure.value = g.exposure;
      (u.uTint.value as THREE.Vector3).set(...g.tint);
      u.uSaturation.value = g.saturation;
      u.uContrast.value = g.contrast;
      u.uFog.value = g.fog;
      (u.uFogColor.value as THREE.Vector3).set(...g.fogColor);
      (u.uSkyTint.value as THREE.Vector3).set(...g.skyTint);
      u.uFade.value = this.fade;
    };
    apply(this.sphereMat.uniforms);
    for (const n of this.nodes.values()) {
      n.opacity += (n.targetOpacity - n.opacity) * Math.min(1, dt * 1.5);
      const op = n.opacity * (1 - this.fade);
      if (n.material) {
        apply(n.material.uniforms);
        n.material.uniforms.uFade.value = 0;
        n.material.uniforms.uOpacity.value = op;
        n.material.uniforms.uTime.value = now;
        const hl = Math.max(0, Math.min(1, n.highlightUntil - now));
        n.material.uniforms.uHighlight.value = hl;
      }
      if (n.shadow) (n.shadow.material as THREE.ShaderMaterial).uniforms.uOpacity.value = op * (n.material ? 1 : 0);
      if (n.glow) {
        const u = (n.glow.material as THREE.ShaderMaterial).uniforms;
        u.uOpacity.value = op * 0.8;
        u.uTime.value = now;
      }
    }
    for (const s of this.storyNodes.values()) {
      s.traverse((o) => {
        const m = (o as THREE.Mesh).material as THREE.ShaderMaterial | undefined;
        if (m?.uniforms?.uTime) {
          m.uniforms.uTime.value = now;
          m.uniforms.uOpacity.value = (0.55 + g.lights * 0.4) * (1 - this.fade);
        }
      });
    }
    const rm = this.rain.material as THREE.ShaderMaterial;
    rm.uniforms.uTime.value = now;
    rm.uniforms.uOpacity.value = g.rain * (1 - this.fade);
    this.rain.visible = g.rain > 0.01;
    const sm = this.snow.material as THREE.ShaderMaterial;
    sm.uniforms.uTime.value = now;
    sm.uniforms.uOpacity.value = g.snow * (1 - this.fade);
    this.snow.visible = g.snow > 0.01;
    const vu = (this.vignette.material as THREE.ShaderMaterial).uniforms;
    vu.uVignette.value = g.vignette;
    vu.uTime.value = now % 100;

    this.onFrame?.(dt);
    this.renderer.render(this.scene, this.camera);
  }
}
