"""地标 + 相对偏移 → 具体位置；用深度图防止物件“穿墙”。"""
from __future__ import annotations

import math
import random
from typing import Optional, Protocol

from .models import EYE_HEIGHT_M, Landmark, Offset, Placement, Relation, SceneManifest, norm_yaw

DIRECTION_YAW = {
    "front": 0.0, "front_right": 45.0, "right": 80.0, "back_right": 135.0,
    "back": 180.0, "back_left": -135.0, "left": -80.0, "front_left": -45.0,
}
DISTANCE_M = {"near": 4.0, "mid": 9.0, "far": 22.0}

# 每个新点位预置的方向地标，保证“左边”“身后”这类说法总有落点
BUILTIN_LANDMARKS = [
    ("正前方", ["前面", "前头", "对面"], 0.0, 10.0),
    ("左边", ["左手边", "左面"], -75.0, 6.0),
    ("右边", ["右手边", "右面"], 75.0, 6.0),
    ("身后", ["后面", "后头", "背后"], 180.0, 5.0),
    ("远处", ["那头", "尽头"], 0.0, 28.0),
]

MIN_DIST, MAX_DIST = 1.5, 60.0


class DepthSampler(Protocol):
    def depth_at(self, viewpoint_id: str, yaw_deg: float, pitch_deg: float) -> Optional[float]:
        """返回该方向上场景表面的距离（米）；没有深度图时返回 None。"""


class NoDepth:
    def depth_at(self, viewpoint_id: str, yaw_deg: float, pitch_deg: float) -> Optional[float]:
        return None


def offset_for(relation: Relation, rng: Optional[random.Random] = None) -> Offset:
    """相对关系 → 偏移量。“旁边”随机落在左侧或右侧，避免和已有物件重叠。"""
    rng = rng or random.Random()
    if relation == "farther":
        return Offset(relation=relation, ddist_m=+0.6)  # ddist 为比例：+60%
    if relation == "nearer":
        return Offset(relation=relation, ddist_m=-0.45)
    if relation == "left":
        return Offset(relation=relation, dyaw_deg=-25.0)
    if relation == "right":
        return Offset(relation=relation, dyaw_deg=+25.0)
    if relation == "beside":
        return Offset(relation=relation, dyaw_deg=rng.choice([-1, 1]) * 9.0)
    if relation == "opposite":
        return Offset(relation=relation, dyaw_deg=180.0)
    return Offset(relation="at")


def ground_pitch(distance_m: float, elevation_m: float = 0.0) -> float:
    """距离为 d、离地 h 的点相对眼睛的俯仰角。"""
    return -math.degrees(math.atan2(EYE_HEIGHT_M - elevation_m, distance_m))


def resolve_placement(
    manifest: SceneManifest,
    landmark: Landmark,
    offset: Offset,
    depth: DepthSampler,
    elevation_m: float = 0.0,
    avoid_ids: tuple[str, ...] = (),
) -> Placement:
    yaw = landmark.yaw_deg + offset.dyaw_deg
    dist = landmark.distance_m * (1.0 + offset.ddist_m)
    if offset.relation == "opposite":
        dist = max(3.0, landmark.distance_m * 0.6)
    dist = min(max(dist, MIN_DIST), MAX_DIST)

    # 避免和同一点位、同一方向的物件完全重叠：轻微错开
    for o in manifest.objects:
        if o.id in avoid_ids or not o.placement or o.placement.viewpoint_id != landmark.viewpoint_id:
            continue
        if abs(norm_yaw(o.placement.yaw_deg - yaw)) < 4 and abs(o.placement.distance_m - dist) < 1.5:
            yaw += 7.0

    # 深度上限：物件底部方向上如果有墙（深度明显小于期望距离），就放到墙前面
    occluded = 0.0
    surface = depth.depth_at(landmark.viewpoint_id, norm_yaw(yaw), ground_pitch(dist, elevation_m))
    if surface is not None and surface < dist * 0.85:
        dist = max(MIN_DIST, surface * 0.9)
    # 中心方向的遮挡（只作参考，前端逐帧细算）
    centre = depth.depth_at(landmark.viewpoint_id, norm_yaw(yaw), ground_pitch(dist, elevation_m + 0.8))
    if centre is not None and centre < dist * 0.8:
        occluded = min(1.0, (dist - centre) / dist)

    return Placement(viewpoint_id=landmark.viewpoint_id, yaw_deg=norm_yaw(yaw),
                     distance_m=round(dist, 2), elevation_m=elevation_m, occluded=round(occluded, 2))
