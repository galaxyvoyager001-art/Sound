// 视角控制：固定人眼高度，只转头；转速有上限，地平线水平，无晃动。
// 无人操作时自动缓慢环顾（在有声音/故事的方向上稍作停留）。
import { clamp, normYaw } from "../shared/types";

export interface LookOptions {
  maxSpeedDeg: number; // 最大角速度（度/秒）
  accelDeg: number; // 最大角加速度（度/秒²）
  autoSpeedDeg: number; // 自动环顾速度
  idleBeforeAutoS: number; // 指令后多久恢复自动环顾
  pitchRange: [number, number];
}

const DEFAULTS: LookOptions = { maxSpeedDeg: 14, accelDeg: 10, autoSpeedDeg: 1.6, idleBeforeAutoS: 25, pitchRange: [-18, 12] };

export class LookController {
  yaw = 0;
  pitch = -2;
  private vyaw = 0;
  private vpitch = 0;
  private targetYaw: number | null = null;
  private targetPitch = -2;
  private lastCommandAt = -1e9;
  private dwellUntil = 0;
  private interest: number[] = [];
  private autoDir = 1;
  opts: LookOptions;
  autoEnabled = true;

  constructor(opts: Partial<LookOptions> = {}) {
    this.opts = { ...DEFAULTS, ...opts };
  }

  /** 设置“兴趣方向”（有声音或故事的物件），自动环顾时会在这些方向停留 */
  setInterest(yaws: number[]) {
    this.interest = yaws.map(normYaw);
  }

  lookAt(yawDeg: number, pitchDeg = -3, now = performance.now() / 1000) {
    this.targetYaw = normYaw(yawDeg);
    this.targetPitch = clamp(pitchDeg, this.opts.pitchRange[0], this.opts.pitchRange[1]);
    this.lastCommandAt = now;
  }

  turnBy(deltaDeg: number, now = performance.now() / 1000) {
    this.lookAt(this.yaw + deltaDeg, this.targetPitch, now);
  }

  /** 拖动（漫游页的手指操作）：直接改变朝向，不经过加速度限制但仍保持水平 */
  drag(dYaw: number, dPitch: number, now = performance.now() / 1000) {
    this.yaw = normYaw(this.yaw + dYaw);
    this.pitch = clamp(this.pitch + dPitch, this.opts.pitchRange[0], this.opts.pitchRange[1]);
    this.vyaw = this.vpitch = 0;
    this.targetYaw = null;
    this.targetPitch = this.pitch;
    this.lastCommandAt = now;
  }

  jumpTo(yawDeg: number, pitchDeg = -2) {
    this.yaw = normYaw(yawDeg);
    this.pitch = pitchDeg;
    this.vyaw = this.vpitch = 0;
    this.targetYaw = null;
  }

  update(dt: number, now: number) {
    dt = Math.min(dt, 0.1);
    const o = this.opts;
    let desiredV = 0;
    if (this.targetYaw !== null) {
      const err = normYaw(this.targetYaw - this.yaw);
      // 临界阻尼式：按剩余角度计算可在不超加速度的情况下停下的速度
      const stopV = Math.sqrt(2 * o.accelDeg * Math.abs(err)) * Math.sign(err);
      desiredV = clamp(stopV, -o.maxSpeedDeg, o.maxSpeedDeg);
      if (Math.abs(err) < 0.3 && Math.abs(this.vyaw) < 0.5) {
        this.targetYaw = null;
        this.vyaw = 0;
      }
    } else if (this.autoEnabled && now - this.lastCommandAt > o.idleBeforeAutoS) {
      desiredV = this.autoVelocity(now);
    }
    const dv = clamp(desiredV - this.vyaw, -o.accelDeg * dt, o.accelDeg * dt);
    this.vyaw += dv;
    this.yaw = normYaw(this.yaw + this.vyaw * dt);

    const perr = this.targetPitch - this.pitch;
    const pv = clamp(perr * 1.2, -o.maxSpeedDeg * 0.5, o.maxSpeedDeg * 0.5);
    this.vpitch += clamp(pv - this.vpitch, -o.accelDeg * dt, o.accelDeg * dt);
    this.pitch = clamp(this.pitch + this.vpitch * dt, o.pitchRange[0], o.pitchRange[1]);
  }

  private autoVelocity(now: number): number {
    if (now < this.dwellUntil) return 0;
    // 经过兴趣方向时停留 6 秒
    for (const y of this.interest) {
      const err = normYaw(y - this.yaw);
      if (Math.abs(err) < 1.0 && Math.sign(err) !== -this.autoDir) {
        this.dwellUntil = now + 6;
        this.yaw += 1.2 * this.autoDir; // 跳过当前点，避免反复触发
        return 0;
      }
    }
    // 接近兴趣方向时放慢
    let speed = this.opts.autoSpeedDeg;
    for (const y of this.interest) {
      const err = Math.abs(normYaw(y - this.yaw));
      if (err < 12) speed = Math.min(speed, 0.5 + (err / 12) * this.opts.autoSpeedDeg);
    }
    // 俯仰缓慢回到略低于地平线
    this.targetPitch = -3 + Math.sin(now * 0.05) * 2;
    return speed * this.autoDir;
  }
}
