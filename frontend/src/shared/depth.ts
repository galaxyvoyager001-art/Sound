// 深度图：8 位灰度 PNG，对数编码 v = 255·ln(d/D_MIN)/ln(D_MAX/D_MIN)（与后端 providers/depth.py 一致）
import { EYE_HEIGHT_M } from "./types";

export const D_MIN = 0.5;
export const D_MAX = 200;

export class DepthMap {
  constructor(public width: number, public height: number, public data: Float32Array) {}

  static geometric(width = 256, height = 128): DepthMap {
    const data = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      const pitch = 90 - ((y + 0.5) / height) * 180;
      const d = pitch < -0.5 ? Math.min(30, EYE_HEIGHT_M / Math.tan((-pitch * Math.PI) / 180)) : 15;
      data.fill(d, y * width, (y + 1) * width);
    }
    return new DepthMap(width, height, data);
  }

  static async load(url: string): Promise<DepthMap> {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = url;
    await img.decode();
    const w = Math.min(img.naturalWidth, 1024);
    const h = Math.round((w * img.naturalHeight) / img.naturalWidth);
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(img, 0, 0, w, h);
    const px = ctx.getImageData(0, 0, w, h).data;
    const data = new Float32Array(w * h);
    const k = Math.log(D_MAX / D_MIN) / 255;
    for (let i = 0; i < w * h; i++) data[i] = D_MIN * Math.exp(px[i * 4] * k);
    return new DepthMap(w, h, data);
  }

  /** 某方向上场景表面的距离（米） */
  at(yawDeg: number, pitchDeg: number): number {
    const u = ((yawDeg / 360 + 0.5) % 1 + 1) % 1;
    const v = (90 - pitchDeg) / 180;
    const x = Math.min(this.width - 1, Math.max(0, Math.floor(u * this.width)));
    const y = Math.min(this.height - 1, Math.max(0, Math.floor(v * this.height)));
    return this.data[y * this.width + x];
  }

  /**
   * 声源遮挡系数 0~1：在声源（水平距离 dist、底部离地 elev、高 h）上取若干点，
   * 看视线方向上的场景表面是否比声源近。
   */
  occlusion(yawDeg: number, dist: number, elev: number, h: number): number {
    let blocked = 0;
    let n = 0;
    for (const fy of [0.25, 0.5, 0.8]) {
      for (const dy of [-2, 0, 2]) {
        const yRel = -EYE_HEIGHT_M + elev + h * fy;
        const pitch = (Math.atan2(yRel, dist) * 180) / Math.PI;
        const r = Math.hypot(dist, yRel);
        const surf = this.at(yawDeg + dy, pitch);
        if (surf < r * 0.82 - 0.4) blocked++;
        n++;
      }
    }
    return blocked / n;
  }
}
