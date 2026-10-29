// 麦克风持续采集（16 kHz 单声道 PCM16，每 100 ms 一帧）+ 能量型语音活动检测（自适应噪声底）。
// 开启浏览器回声消除；陪聊者说话期间提高门限，只有老人明显开口才算“插话”。

const WORKLET = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Int16Array(1600); this.n = 0; this.sum = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      this.sum += s * s;
      if (this.n === this.buf.length) {
        this.port.postMessage({ pcm: this.buf.buffer, rms: Math.sqrt(this.sum / this.n) }, [this.buf.buffer]);
        this.buf = new Int16Array(1600); this.n = 0; this.sum = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;

export interface VadEvent {
  speaking: boolean;
  t: number; // 会谈录音时间（秒）
}

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private noiseFloorDb = -55;
  private speaking = false;
  private speakingFrames = 0;
  private aboveCount = 0;
  private belowCount = 0;
  private framesSent = 0;
  systemSpeaking = false; // 陪聊者正在说话
  onFrame?: (pcm: ArrayBuffer) => void;
  onVad?: (e: VadEvent) => void;
  onLevel?: (db: number, floorDb: number) => void;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    this.ctx = new AudioContext({ sampleRate: 16000 });
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    const src = this.ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(this.ctx, "pcm-capture");
    node.port.onmessage = (e) => this.handle(e.data.pcm as ArrayBuffer, e.data.rms as number);
    src.connect(node);
    // 不连到扬声器：只采集
  }

  get timeS(): number {
    return this.framesSent * 0.1;
  }

  private handle(pcm: ArrayBuffer, rms: number) {
    this.framesSent++;
    this.onFrame?.(pcm);
    const db = 20 * Math.log10(Math.max(rms, 1e-6));
    // 噪声底：安静时缓慢跟随；“说话”持续 20 秒以上多半是持续噪声（电视、风扇），也慢慢上浮
    this.speakingFrames = this.speaking ? this.speakingFrames + 1 : 0;
    if (!this.speaking || this.speakingFrames > 200) {
      const rate = db < this.noiseFloorDb ? 0.2 : this.speaking ? 0.03 : 0.01;
      this.noiseFloorDb += (db - this.noiseFloorDb) * rate;
      this.noiseFloorDb = Math.max(-75, Math.min(-20, this.noiseFloorDb));
    }
    const margin = this.systemSpeaking ? 18 : 9;
    const isVoice = db > this.noiseFloorDb + margin && db > -52;
    this.onLevel?.(db, this.noiseFloorDb);
    if (isVoice) {
      this.aboveCount++;
      this.belowCount = 0;
      if (!this.speaking && this.aboveCount >= (this.systemSpeaking ? 4 : 2)) {
        this.speaking = true;
        this.onVad?.({ speaking: true, t: this.timeS - this.aboveCount * 0.1 });
      }
    } else {
      this.belowCount++;
      this.aboveCount = 0;
      if (this.speaking && this.belowCount >= 6) { // 600 ms 挂起时间
        this.speaking = false;
        this.onVad?.({ speaking: false, t: this.timeS - 0.6 });
      }
    }
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
  }
}
