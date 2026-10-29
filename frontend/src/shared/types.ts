// 与后端 backend/app/scene/models.py 对应的类型（只列前端用到的字段）

export type AcousticProfile = "clean" | "old_radio" | "loudspeaker_horn" | "gramophone";

export interface Placement {
  viewpoint_id: string;
  yaw_deg: number;
  distance_m: number;
  elevation_m: number;
  occluded: number;
}

export interface SoundSpec {
  clip_id: string;
  variant_index: number;
  gain_db: number;
  mode: "loop" | "event";
  interval_range_s: [number, number];
  acoustic_profile: AcousticProfile;
  ref_distance_m: number;
  rolloff: number;
  max_distance_m: number;
}

export interface SceneObject {
  id: string;
  type: string;
  label_zh: string;
  asset_id: string | null;
  model_asset_id: string | null;
  asset_status: "pending" | "generating" | "ready" | "failed";
  landmark: string;
  placement: Placement | null;
  height_m: number;
  sound: SoundSpec | null;
}

export interface Viewpoint {
  id: string;
  name_zh: string;
  panorama_path: string | null;
  depth_path: string | null;
  neighbors: string[];
  status: string;
}

export interface Landmark {
  id: string;
  name_zh: string;
  viewpoint_id: string;
  yaw_deg: number;
  distance_m: number;
  linked_viewpoint_id: string | null;
  builtin: boolean;
}

export interface AmbienceLayer {
  id: string;
  clip_id: string;
  name_zh: string;
  gain_db: number;
  role: string;
}

export interface Story {
  id: string;
  title_zh: string;
  transcript: string;
  audio_path: string | null;
  anchor: { kind: "object" | "landmark" | "viewpoint"; id: string };
  include_in_export: boolean;
}

export interface SceneManifest {
  scene_id: string;
  version: number;
  condition: "audio_only" | "generic" | "personal";
  region_style: string;
  era: string;
  time_of_day: string;
  weather: string;
  season: string;
  entry_viewpoint: string | null;
  viewpoints: Viewpoint[];
  landmarks: Landmark[];
  objects: SceneObject[];
  ambience: AmbienceLayer[];
  stories: Story[];
}

/** 后端随场景一起下发的素材解析表：clip_id / asset_id → 可访问的 URL */
export interface AssetInfo {
  urls: string[];
  name?: string;
  width?: number;
  height?: number;
  kind?: string;
}

export type AssetTable = Record<string, AssetInfo>;

export interface ScenePayload {
  manifest: SceneManifest;
  assets: AssetTable;
  highlight?: string[];
}

export const EYE_HEIGHT_M = 1.6;

export function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** 方位角/俯仰角/距离 → three.js 世界坐标（相机在原点，yaw=0 朝 -Z，向右为正） */
export function dirToVec(yawDeg: number, pitchDeg: number, dist = 1): [number, number, number] {
  const y = (yawDeg * Math.PI) / 180;
  const p = (pitchDeg * Math.PI) / 180;
  return [dist * Math.cos(p) * Math.sin(y), dist * Math.sin(p), -dist * Math.cos(p) * Math.cos(y)];
}

export function placementToVec(pl: Placement, heightAboveBase = 0): [number, number, number] {
  const y = (pl.yaw_deg * Math.PI) / 180;
  return [pl.distance_m * Math.sin(y), -EYE_HEIGHT_M + pl.elevation_m + heightAboveBase, -pl.distance_m * Math.cos(y)];
}

export function normYaw(y: number): number {
  let v = ((((y + 180) % 360) + 360) % 360) - 180;
  if (v === -180) v = 180;
  return v;
}
