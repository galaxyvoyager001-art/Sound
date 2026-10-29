"""单目深度估计（等距柱状投影全景）。

做法：把全景按方位角切成 8 块相互重叠的竖条（每块 90°，覆盖仰角 +50°~-70°），
分别送入 Depth Anything V2 Small（本地 ONNX），得到相对视差；
再用“地面几何先验”（眼高 1.6 m，俯角 φ 处地面距离 = 1.6 / tan φ）把每块视差
对齐成米制深度，最后按方位角余弦权重拼接。没有模型时退回纯几何先验。

深度图存为 8 位灰度 PNG，按对数编码：v = 255·ln(d/D_MIN)/ln(D_MAX/D_MIN)，前后端共用此约定。
"""
from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..scene.models import EYE_HEIGHT_M

log = logging.getLogger("depth")

D_MIN, D_MAX = 0.5, 200.0
OUT_W, OUT_H = 1024, 512


def encode_depth_png(depth_m: np.ndarray) -> bytes:
    d = np.clip(depth_m, D_MIN, D_MAX)
    v = np.log(d / D_MIN) / math.log(D_MAX / D_MIN) * 255.0
    b = io.BytesIO()
    Image.fromarray(v.round().astype(np.uint8), "L").save(b, "PNG")
    return b.getvalue()


def decode_depth_png(data: bytes) -> np.ndarray:
    v = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
    return D_MIN * np.exp(v / 255.0 * math.log(D_MAX / D_MIN))


def pitch_of_rows(h: int) -> np.ndarray:
    """每一行中心的俯仰角（度），第 0 行 ≈ +90°。"""
    return 90.0 - (np.arange(h) + 0.5) / h * 180.0


def geometric_prior(h: int = OUT_H, w: int = OUT_W, open_distance: float = 15.0) -> np.ndarray:
    pitch = pitch_of_rows(h)
    rows = np.where(pitch < -0.5, EYE_HEIGHT_M / np.tan(np.radians(-np.minimum(pitch, -0.5))), open_distance)
    rows = np.minimum(rows, open_distance * 2)
    return np.repeat(rows[:, None], w, axis=1).astype(np.float32)


class DepthEstimator:
    name = "base"

    def estimate(self, pano: Image.Image, provider_depth: Optional[bytes] = None) -> np.ndarray:
        raise NotImplementedError


class GeometricDepth(DepthEstimator):
    name = "geometric"

    def estimate(self, pano: Image.Image, provider_depth: Optional[bytes] = None) -> np.ndarray:
        if provider_depth:
            return align_disparity_map(np.asarray(Image.open(io.BytesIO(provider_depth)).convert("L"),
                                                  dtype=np.float32))
        return geometric_prior()


def align_disparity_map(disp: np.ndarray) -> np.ndarray:
    """把整张全景的相对视差（亮=近）对齐成米制深度。"""
    disp = np.asarray(Image.fromarray(disp).resize((OUT_W, OUT_H), Image.BILINEAR), dtype=np.float32)
    pitch = pitch_of_rows(OUT_H)
    return _align(disp, pitch)


def _align(disp: np.ndarray, pitch_rows: np.ndarray) -> np.ndarray:
    """相对视差 → 米制深度。

    用俯角 -6°~-70° 的行（大部分是地面）建立“视差 → 逆深度”的单调分段线性映射：
    每行取视差中位数，与该行的地面逆深度 tan(φ)/1.6 配对。比这些行更远的像素（视差更小，
    如远处房屋、天空）在逆深度上线性外推到 0（视差最小处 = 无穷远）。"""
    rows = np.where((pitch_rows < -6) & (pitch_rows > -70))[0]
    if len(rows) < 4:
        return geometric_prior(disp.shape[0], disp.shape[1])
    xs = np.median(disp[rows], axis=1)
    ys = np.tan(np.radians(-pitch_rows[rows])) / EYE_HEIGHT_M
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])  # 强制单调
    if xs[-1] - xs[0] < 1e-6:
        return geometric_prior(disp.shape[0], disp.shape[1])
    inv = np.interp(disp, xs, ys)
    d_sky = float(np.quantile(disp, 0.01))
    below = disp < xs[0]
    if xs[0] - d_sky > 1e-6:
        inv[below] = ys[0] * np.clip((disp[below] - d_sky) / (xs[0] - d_sky), 0, 1)
    above = disp > xs[-1]  # 比最近的地面还近：按最后一段斜率外推
    if above.any() and len(xs) >= 2:
        slope = (ys[-1] - ys[-2]) / max(1e-6, xs[-1] - xs[-2])
        inv[above] = ys[-1] + slope * (disp[above] - xs[-1])
    depth = 1.0 / np.maximum(inv, 1.0 / D_MAX)
    return np.clip(depth, D_MIN, D_MAX).astype(np.float32)


class OnnxDepthAnything(DepthEstimator):
    name = "local_onnx"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _infer(self, tile: Image.Image) -> np.ndarray:
        w, h = 378, 504  # 14 的倍数，宽高比 3:4（90° × 120°）
        x = np.asarray(tile.convert("RGB").resize((w, h), Image.BICUBIC), np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: x})[0][0]
        return out.astype(np.float32)

    def estimate(self, pano: Image.Image, provider_depth: Optional[bytes] = None) -> np.ndarray:
        W, H = 2048, 1024
        img = pano.convert("RGB").resize((W, H), Image.BICUBIC)
        top_pitch, bot_pitch = 50.0, -70.0
        r0 = int((90 - top_pitch) / 180 * H)
        r1 = int((90 - bot_pitch) / 180 * H)
        tile_w = W // 4
        n_tiles = 8
        acc = np.zeros((OUT_H, OUT_W), np.float64)
        wsum = np.zeros((OUT_H, OUT_W), np.float64)
        wide = np.concatenate([np.asarray(img), np.asarray(img)], axis=1)  # 方便跨越接缝取块
        out_r0 = int(r0 / H * OUT_H)
        out_r1 = int(r1 / H * OUT_H)
        rows_pitch = pitch_of_rows(OUT_H)[out_r0:out_r1]
        for k in range(n_tiles):
            x0 = k * W // n_tiles
            tile = Image.fromarray(wide[r0:r1, x0:x0 + tile_w])
            disp = self._infer(tile)
            ow = OUT_W // 4
            disp_r = np.asarray(Image.fromarray(disp).resize((ow, out_r1 - out_r0), Image.BILINEAR), np.float32)
            depth = _align(disp_r, rows_pitch)
            weight = np.sin(np.linspace(0, math.pi, ow)) ** 2 + 1e-3  # 中间权重大
            ox = k * OUT_W // n_tiles
            cols = (np.arange(ow) + ox) % OUT_W
            # 在对数域融合，避免远近混合偏差
            acc[out_r0:out_r1][:, cols] += np.log(depth) * weight[None, :]
            wsum[out_r0:out_r1][:, cols] += weight[None, :]
        prior = geometric_prior()
        valid = wsum > 0
        result = prior.copy()
        result[valid] = np.exp(acc[valid] / wsum[valid])
        # 覆盖范围以外：上方沿用最上一行，下方用地面先验
        if out_r0 > 0:
            result[:out_r0] = result[out_r0][None, :]
        result[out_r1:] = prior[out_r1:]
        return np.clip(result, D_MIN, D_MAX).astype(np.float32)


def make_depth_estimator(kind: str, model_path: Path) -> DepthEstimator:
    if kind == "local_onnx":
        if model_path.exists():
            try:
                return OnnxDepthAnything(model_path)
            except Exception as e:  # 模型损坏或 onnxruntime 不可用
                log.warning("加载深度模型失败，改用几何先验：%s", e)
        else:
            log.warning("找不到深度模型 %s，改用几何先验（运行 scripts/download_models.py 下载）", model_path)
    return GeometricDepth()


class DepthMapSampler:
    """后端用的深度查询（物件放置、声音遮挡粗算）。"""

    def __init__(self) -> None:
        self.maps: dict[str, np.ndarray] = {}

    def set(self, viewpoint_id: str, depth: np.ndarray) -> None:
        self.maps[viewpoint_id] = depth

    def load_png(self, viewpoint_id: str, path: Path) -> None:
        if path.exists():
            self.maps[viewpoint_id] = decode_depth_png(path.read_bytes())

    def depth_at(self, viewpoint_id: str, yaw_deg: float, pitch_deg: float) -> Optional[float]:
        d = self.maps.get(viewpoint_id)
        if d is None:
            return None
        h, w = d.shape
        u = ((yaw_deg / 360.0) + 0.5) % 1.0
        v = (90.0 - pitch_deg) / 180.0
        x = min(w - 1, max(0, int(u * w)))
        y = min(h - 1, max(0, int(v * h)))
        # 取 3×3 邻域最小值，偏保守
        ys = slice(max(0, y - 1), min(h, y + 2))
        xs = [(x + dx) % w for dx in (-1, 0, 1)]
        return float(d[ys][:, xs].min())
