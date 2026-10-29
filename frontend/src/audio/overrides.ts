// 调参试听页保存的效果参数（本机浏览器）。会谈页创建效果链时会合并这些参数。
import { ProfileParams } from "./effects";

export type Overrides = Partial<Record<"old_radio" | "loudspeaker_horn" | "gramophone", Partial<ProfileParams>>>;
const KEY = "profileOverrides";

export function loadOverrides(): Overrides {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "{}");
  } catch {
    return {};
  }
}

export function saveOverrides(o: Overrides) {
  try {
    localStorage.setItem(KEY, JSON.stringify(o));
  } catch { /* 存储不可用时忽略 */ }
}
