// 全景球、物件 billboard 共用的调色函数；物件按深度图逐像素遮挡

export const GRADE_UNIFORMS_GLSL = /* glsl */ `
uniform float uExposure;
uniform vec3 uTint;
uniform float uSaturation;
uniform float uContrast;
uniform float uFog;
uniform vec3 uFogColor;
uniform vec3 uSkyTint;
uniform float uFade;       // 点位切换时淡出到纸色（0 = 正常，1 = 全纸色）
uniform vec3 uFadeColor;

const float D_MIN = 0.5;
const float D_MAX = 200.0;
const float PI = 3.14159265359;

vec2 dirToEquirect(vec3 d) {
  float u = 0.5 + atan(d.x, -d.z) / (2.0 * PI);
  float v = 0.5 + asin(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2(u, v);
}

float decodeDepth(float g) {
  return D_MIN * exp(g * log(D_MAX / D_MIN));
}

vec3 grade(vec3 c, float dist, float upness) {
  c *= uTint;
  c = mix(c, c * uSkyTint, smoothstep(0.0, 0.25, upness));
  float l = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(vec3(l), c, uSaturation);
  c = (c - 0.5) * uContrast + 0.5;
  c *= uExposure;
  float fogAmt = uFog * (1.0 - exp(-dist / 18.0));
  c = mix(c, uFogColor * uExposure, clamp(fogAmt, 0.0, 0.92));
  c = mix(c, uFadeColor, uFade);
  return c;
}
`;

export const PANO_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize(position);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const PANO_FRAG = /* glsl */ `
${GRADE_UNIFORMS_GLSL}
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform sampler2D uDepthA;
uniform sampler2D uDepthB;
uniform float uMix;        // A→B 交叉淡入
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  vec2 uv = dirToEquirect(d);
  // 接缝处避免 mipmap 跳变：用 textureGrad 的简化替代——直接关闭 mipmap
  vec3 a = texture2D(uTexA, uv).rgb;
  vec3 b = texture2D(uTexB, uv).rgb;
  float da = decodeDepth(texture2D(uDepthA, uv).r);
  float db = decodeDepth(texture2D(uDepthB, uv).r);
  vec3 c = mix(a, b, uMix);
  float dist = mix(da, db, uMix);
  gl_FragColor = vec4(grade(c, dist, d.y), 1.0);
  #include <colorspace_fragment>
}
`;

export const BILLBOARD_VERT = /* glsl */ `
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vUv = uv;
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

export const BILLBOARD_FRAG = /* glsl */ `
${GRADE_UNIFORMS_GLSL}
uniform sampler2D uMap;
uniform sampler2D uDepth;
uniform float uOpacity;
uniform float uHighlight;
uniform float uTime;
uniform float uUseDepth;
varying vec2 vUv;
varying vec3 vWorld;
void main() {
  vec4 t = texture2D(uMap, vUv);
  if (t.a < 0.02) discard;
  float dist = length(vWorld);
  vec3 dir = vWorld / dist;
  float scene = decodeDepth(texture2D(uDepth, dirToEquirect(dir)).r);
  // 场景表面比物件这一点更近 → 被墙/房子挡住（留出容差，避免地面接触处闪烁）
  float margin = 0.6 + dist * 0.12;
  float vis = mix(1.0, smoothstep(dist - margin - 0.8, dist - margin, scene), uUseDepth);
  vec3 c = t.rgb;
  // 修改后的轻轻高亮：暖色微光脉动
  float pulse = 0.5 + 0.5 * sin(uTime * 3.0);
  c = mix(c, c * vec3(1.25, 1.12, 0.9) + vec3(0.08, 0.05, 0.0), uHighlight * (0.4 + 0.4 * pulse));
  gl_FragColor = vec4(grade(c, dist, 0.0), t.a * uOpacity * vis);
  #include <colorspace_fragment>
}
`;

// 地面接触阴影（柔和椭圆）
export const SHADOW_FRAG = /* glsl */ `
uniform float uOpacity;
varying vec2 vUv;
void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float a = smoothstep(1.0, 0.2, length(p)) * 0.28 * uOpacity;
  gl_FragColor = vec4(0.12, 0.09, 0.06, a);
}
`;

// 故事标记 / 生成中占位：柔和的暖色光点
export const GLOW_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uTime;
uniform float uPhase;
varying vec2 vUv;
void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float r = length(p);
  float pulse = 0.85 + 0.15 * sin(uTime * 1.6 + uPhase);
  float core = smoothstep(0.32, 0.0, r);
  float halo = smoothstep(1.0, 0.0, r) * 0.45;
  float a = (core + halo) * pulse * uOpacity;
  gl_FragColor = vec4(uColor * (0.9 + core * 0.4), a);
}
`;

// 雨丝 / 雪花粒子（围绕相机的立方体内循环下落）
export const PRECIP_VERT = /* glsl */ `
uniform float uTime;
uniform float uSpeed;
uniform float uSize;
attribute float aSeed;
varying float vAlpha;
void main() {
  vec3 p = position;
  float range = 24.0;
  p.y = mod(p.y - uTime * uSpeed * (0.8 + aSeed * 0.4), range) - range * 0.5;
  p.x += sin(uTime * 0.6 + aSeed * 30.0) * (uSpeed < 5.0 ? 0.6 : 0.05);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  float dist = -mv.z;
  gl_PointSize = uSize * (12.0 / max(dist, 0.5));
  vAlpha = smoothstep(30.0, 4.0, dist) * smoothstep(0.3, 1.5, dist);
}
`;

export const RAIN_FRAG = /* glsl */ `
uniform float uOpacity;
varying float vAlpha;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float a = smoothstep(0.12, 0.0, abs(p.x)) * smoothstep(1.0, 0.2, abs(p.y));
  gl_FragColor = vec4(0.82, 0.86, 0.92, a * vAlpha * uOpacity * 0.55);
}
`;

export const SNOW_FRAG = /* glsl */ `
uniform float uOpacity;
varying float vAlpha;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float a = smoothstep(1.0, 0.3, length(p));
  gl_FragColor = vec4(1.0, 1.0, 1.0, a * vAlpha * uOpacity * 0.85);
}
`;

// 屏幕暗角（叠加在最前面的一张全屏四边形）
export const VIGNETTE_FRAG = /* glsl */ `
uniform float uVignette;
uniform float uGrain;
uniform float uTime;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec2 p = vUv * 2.0 - 1.0;
  float v = smoothstep(0.55, 1.45, length(p * vec2(0.9, 1.1))) * uVignette;
  float g = (hash(vUv * 900.0 + uTime) - 0.5) * uGrain;
  gl_FragColor = vec4(vec3(0.16, 0.11, 0.07) + g, v + abs(g) * 0.5);
}
`;
