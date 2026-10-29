# 「听见从前」实现计划（简版）

> 写于开发开始前；实际实现与此一致的地方不再重复，偏离的地方在 README 中说明。

## 1. 模块划分

| 层 | 目录 | 职责 |
|---|---|---|
| Provider 层 | `backend/app/providers/` | LLM、语音识别、语音合成、图像生成、深度估计、文本向量、文生音效、3D 生成。全部可配置、可切换，统一重试与降级；没有 Key 时自动使用本地“演示”实现，保证系统能跑通。 |
| 场景清单 | `backend/app/scene/` | Pydantic 定义的 `SceneManifest`；修改指令（`EditCommand`）的校验与执行；版本历史与撤销（SQLite）；地标 + 相对偏移 → 具体方位与距离的解析（结合深度图）。 |
| 生成 | `backend/app/generation/` | 全景图生成与后处理（无缝接缝、两极柔化、深度图）、物件插画生成与抠图、风格提示词组装（`config/style.yaml`）。 |
| 素材 | `backend/app/assets/` + `scripts/` | `assets/manifest.json` 登记、`REFERENCES.md` 自动生成、检索（标签/同义词/拼音/向量融合排序）、在线检索下载（Wikimedia Commons、Freesound）、文生音效。 |
| 会谈 | `backend/app/conversation/` | 会谈编排器（状态机）、陪聊提示词、话语理解（分类 + 指令）、确认与撤销、故事归档、逐步生成场景、结束流程。 |
| 研究 | `backend/app/research/` | 事件日志、会后自动分析（时长、段数、主动开口、细节抽取）、HTML/CSV 报告、跨条件对比表。 |
| 导出 | `backend/app/export/` | “老家漫游”单文件离线网页（素材全部内嵌）。 |
| 前端 | `frontend/src/` | three.js 全景 + billboard 物件（逐像素深度遮挡）、后期调色/雨雪雾、自动环顾；Web Audio 空间音频引擎、时代声学效果链；麦克风采集 + VAD；设置页、试听页、漫游页。 |

## 2. 选用的服务（2026-10 查阅官方文档）

| 能力 | 默认 | 备选 | 说明 |
|---|---|---|---|
| LLM | 百炼 OpenAI 兼容接口 `qwen3.7-plus`（结构化输出用 `json_schema` + `strict`） | `qwen3.8-flash`（快速分类）、`qwen3.8-max` | 官方文档：JSON Schema 模式仅部分模型支持；不支持时自动退回 `json_object` 模式 + 提示词内嵌 Schema，统一用 Pydantic 校验、失败带错误信息重试。 |
| 实时语音识别 | `fun-asr-realtime`（WebSocket，`run-task`/`result-generated` 协议，支持方言、热词） | `paraformer-realtime-v2` | 协议两者相同，模型名可配置。 |
| 语音合成 | `cosyvoice-v3-flash`，音色 `longanrou_v3`（温和知心女声），`rate=0.85` | `longyan_v3`、`longanyun_v3` | WebSocket `run-task → continue-task → finish-task`，二进制帧返回 mp3。 |
| 全景图 | 通义万相 `wan2.6-t2i`（2016×1008，提示词约束等距柱状投影，后处理做无缝接缝与两极柔化） | Blockade Labs Skybox（原生等距柱状投影 + 自带深度图，需付费 Key） | 万相不保证严格的球面投影，Skybox 是“真正的”全景服务。两者可切换。 |
| 物件插画 | 万相 `wan2.6-t2i` 纯白背景生成 → 本地抠图（边缘连通区域去背景 + 羽化） | 安装 `rembg` 后自动启用更好的抠图 | 文档确认两家都不输出透明背景，所以本地抠图。 |
| 深度估计 | 本地 Depth Anything V2 Small（ONNX，onnxruntime，按全景分块推理再拼接，用地面几何先验对齐成米制深度） | Skybox 自带深度图；纯几何先验（无模型时） | |
| 文本向量 | `qwen3.7-text-embedding-flash`（OpenAI 兼容 `/embeddings`） | `text-embedding-v4` | |
| 声音素材 | Wikimedia Commons（无需 Key，授权信息完整） + Freesound（需 Key，只取 CC0 / CC-BY） | ElevenLabs 文生音效（可选） | Pixabay 的公开 API 不提供音频检索，因此不做自动下载。 |
| 3D 生成 | 默认关闭；预留 Tripo / Meshy 接口 | | |

## 3. 关键技术难点与对策

1. **万相全景不是严格的等距柱状投影**：提示词约束 + 后处理（左右接缝交叉淡化后裁切、两极逐行水平模糊）；深度用几何先验对齐，物件放置以“地面 + 深度上限”为准，不依赖全景几何完全正确。
2. **物件与全景的遮挡**：billboard 着色器逐像素把自身距离与深度图比较，实现“部分遮挡”；声音用同一深度图在声源方向多点采样得到遮挡系数 → 低通 + 衰减，平滑过渡。
3. **免手动对话的轮次**：ASR 自带断句 + 自己的“老人沉默 ≥ 4 秒才回应”计时；控制类短语（“停一下”“太吵了”）走快速通道立即执行；回声：浏览器回声消除 + 陪聊说话期间提高 VAD 门限，老人插话时陪聊立即停下。
4. **LLM 指令可靠性**：LLM 只输出一个扁平的指令对象列表（`op` 枚举 + 可选字段），后端转换成强类型指令并逐条校验（物件是否存在、地标是否存在、素材是否敏感），带置信度阈值和确认流程。
5. **永不报错、永不白屏**：所有外部调用都有重试 + 缓存 + 降级；前端第一帧就有暖色占位背景，新全景加载完成后交叉淡入；错误只写日志和隐藏的工作人员面板。
6. **离线导出**：手机上 `file://` 打开时 WebGL 不能加载本地图片，因此“老家漫游”导出为单个 HTML，素材以 data URI 内嵌。

## 4. 推进顺序

按需求第 12 节的 7 步推进，每步一个 commit。
