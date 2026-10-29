// 时段 / 天气 / 季节 → 调色与雾、雨雪参数（轻量变化全部在这里实现，不重新生成全景）

export interface Grade {
  exposure: number;
  tint: [number, number, number];
  saturation: number;
  contrast: number;
  fog: number; // 0~1
  fogColor: [number, number, number];
  skyTint: [number, number, number]; // 只作用于地平线以上
  rain: number; // 0~1
  snow: number; // 0~1
  vignette: number;
  lights: number; // 窗口/故事标记的发光强度（夜晚更亮）
}

const BASE: Grade = {
  exposure: 1, tint: [1, 1, 1], saturation: 0.95, contrast: 1, fog: 0.05, fogColor: [0.92, 0.88, 0.8],
  skyTint: [1, 1, 1], rain: 0, snow: 0, vignette: 0.35, lights: 0.4,
};

const TIME: Record<string, Partial<Grade>> = {
  清晨: { exposure: 0.92, tint: [0.96, 0.98, 1.05], saturation: 0.8, fog: 0.22, fogColor: [0.85, 0.87, 0.9], skyTint: [1.0, 0.98, 1.05] },
  上午: {},
  中午: { exposure: 1.05, contrast: 1.04 },
  下午: { tint: [1.06, 1.0, 0.9], skyTint: [1.05, 1.0, 0.92] },
  傍晚: { exposure: 0.82, tint: [1.12, 0.93, 0.78], saturation: 0.95, fogColor: [0.95, 0.75, 0.6], fog: 0.12, skyTint: [1.25, 0.9, 0.7], lights: 0.8 },
  夜晚: { exposure: 0.5, tint: [0.72, 0.84, 1.15], saturation: 0.55, fog: 0.15, fogColor: [0.2, 0.24, 0.35], skyTint: [0.7, 0.8, 1.1], vignette: 0.55, lights: 1.0 },
};

const WEATHER: Record<string, Partial<Grade>> = {
  晴: {},
  阴: { saturation: -0.25, exposure: -0.08, contrast: -0.08, fog: 0.12, skyTint: [0.92, 0.94, 0.98] },
  小雨: { saturation: -0.3, exposure: -0.15, contrast: -0.1, fog: 0.25, fogColor: [0.7, 0.74, 0.78], rain: 0.45, skyTint: [0.85, 0.88, 0.94] },
  大雨: { saturation: -0.4, exposure: -0.25, contrast: -0.15, fog: 0.45, fogColor: [0.62, 0.66, 0.72], rain: 1.0, skyTint: [0.78, 0.82, 0.9] },
  雪: { saturation: -0.35, exposure: 0.04, fog: 0.3, fogColor: [0.9, 0.92, 0.96], snow: 0.8, tint: [0.97, 0.99, 1.05] },
  雾: { saturation: -0.3, fog: 0.65, fogColor: [0.88, 0.88, 0.86], contrast: -0.15 },
};

const SEASON: Record<string, Partial<Grade>> = {
  春: { tint: [1.0, 1.03, 0.98] },
  夏: { tint: [1.03, 1.01, 0.96] },
  秋: { tint: [1.08, 0.99, 0.86], saturation: 0.05 },
  冬: { tint: [0.96, 0.98, 1.04], saturation: -0.15 },
};

function mul3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

export function computeGrade(time: string, weather: string, season: string): Grade {
  const g: Grade = JSON.parse(JSON.stringify(BASE));
  const t = TIME[time] ?? {};
  Object.assign(g, t);
  const w = WEATHER[weather] ?? {};
  // 天气用“增量”叠加在时段上
  g.saturation += w.saturation ?? 0;
  g.exposure += w.exposure ?? 0;
  g.contrast += w.contrast ?? 0;
  g.fog = Math.max(g.fog, w.fog ?? 0);
  if (w.fogColor) g.fogColor = time === "夜晚" ? mul3(w.fogColor, [0.3, 0.33, 0.42]) : w.fogColor;
  if (w.skyTint) g.skyTint = mul3(g.skyTint, w.skyTint);
  if (w.tint) g.tint = mul3(g.tint, w.tint);
  g.rain = w.rain ?? 0;
  g.snow = w.snow ?? 0;
  const s = SEASON[season] ?? {};
  if (s.tint) g.tint = mul3(g.tint, s.tint);
  g.saturation = Math.max(0.2, g.saturation + (s.saturation ?? 0));
  return g;
}

export function lerpGrade(a: Grade, b: Grade, t: number): Grade {
  const l = (x: number, y: number) => x + (y - x) * t;
  const l3 = (x: [number, number, number], y: [number, number, number]): [number, number, number] => [l(x[0], y[0]), l(x[1], y[1]), l(x[2], y[2])];
  return {
    exposure: l(a.exposure, b.exposure), tint: l3(a.tint, b.tint), saturation: l(a.saturation, b.saturation),
    contrast: l(a.contrast, b.contrast), fog: l(a.fog, b.fog), fogColor: l3(a.fogColor, b.fogColor),
    skyTint: l3(a.skyTint, b.skyTint), rain: l(a.rain, b.rain), snow: l(a.snow, b.snow),
    vignette: l(a.vignette, b.vignette), lights: l(a.lights, b.lights),
  };
}
