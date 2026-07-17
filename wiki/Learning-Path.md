# Learning Path

本文给出 mini-vllm 的**推荐阅读顺序**、逐层代码导览、动手实验清单与扩展方向。目标：让你从"跑起来"到"读懂"再到"改得动"。

## 学习地图

```
第 0 步：跑起来          Getting-Started
   │
第 1 步：建立全局认知     Architecture（分层 + 数据流 + 线程模型）
   │
第 2 步：两大核心机制     PagedAttention（省显存） + Continuous-Batching（提吞吐）
   │
第 3 步：模型计算         Transformer（Qwen3 五大特性）
   │
第 4 步：支撑机制         KV-Cache-Memory + Weight-Loading + OpenAI-API
   │
第 5 步：动手改造         见下方"扩展方向"
```

## 推荐阅读顺序（含理由）

1. **[Getting-Started](Getting-Started.md)** — 先把服务跑起来，用 curl 或 Web 页面对话一次，建立"它能干什么"的直觉。
2. **[Architecture](Architecture.md)** — 看懂分层、`LlmModel` 抽象、一条请求的完整旅程、单引擎线程 + 多 HTTP 线程模型。这是后续所有细节的骨架。
3. **[PagedAttention](PagedAttention.md)** — vLLM 的灵魂之一。理解"分页 KV cache"如何消除显存碎片，以及 decode 阶段逐 block 累加如何模拟 CUDA kernel。
4. **[Continuous-Batching](Continuous-Batching.md)** — vLLM 的灵魂之二。理解 `admit → decode → sweep` 三段循环如何让请求随时进出、batch 始终填满。
5. **[Transformer](Transformer.md)** — 理解 Qwen3 的 RMSNorm / RoPE / GQA / SwiGLU / QK-Norm 五大特性，以及它们如何用纯 Java 实现并与 HF 对齐。
6. **[KV-Cache-Memory](KV-Cache-Memory.md)** — 深入内存管理：按需分配、引用计数回收、前缀共享、显存估算。
7. **[Weight-Loading](Weight-Loading.md)** — 理解 safetensors 解析、BF16 转换、权重命名映射、自动下载。
8. **[OpenAI-API](OpenAI-API.md)** — 理解 HTTP/SSE、ChatML、增量解码，把整条链路首尾闭合。

## 逐层代码导览（自底向上读源码）

若你偏好"从最小单元读起"，推荐这个顺序：

| 层 | 建议阅读顺序 | 关键文件 |
|---|---|---|
| math | Tensor → Softmax → RmsNorm → Silu → Matmul/DotKernel → Sampler | `math/*.java` |
| memory | BlockPool → BlockTable → KVCacheManager | `memory/*.java` |
| model | Embedding → Linear → RotaryEmbedding → SwiGluFfn → Qwen3Attention → Qwen3Block → Qwen3Model | `model/*.java` |
| weights | Bf16 → SafetensorsLoader → Qwen3Loader → ModelDownloader | `weights/*.java` |
| tokenizer | ByteLevelBpe → BpeTokenizer → ChatTemplate | `tokenizer/*.java` |
| core | Sequence → Scheduler → LLMEngine | `core/*.java` |
| api | SseWriter → OpenAiHandler → WebUiHandler | `api/*.java` |
| 入口 | MiniVllmServer | `MiniVllmServer.java` |

每个类的顶部注释都写了"学习要点"，配合本 Wiki 对照阅读效果最佳。

## 用测试驱动理解

测试是"可执行的文档"。推荐先读测试再读实现：

```bash
mvn test
```

| 想理解 | 读这个测试 |
|---|---|
| 引擎端到端行为（批处理/EOS/流式/采样） | `core/LLMEngineE2ETest` |
| 分页分配与引用计数 | `memory/BlockPoolTest`、`KVCacheManagerTest` |
| RoPE 数值 | `model/RotaryEmbeddingTest` |
| Qwen3 前向 | `model/Qwen3ModelTest` |
| BPE 编解码与 HF 一致 | `tokenizer/BpeTokenizerTest` |
| 算子正确性 | `math/*Test` |
| **与 HF 逐层数值对齐（终极验证）** | `model/Qwen3AlignmentTest` |

跑对齐测试（需真实权重）：

```bash
mvn test -Dtest=Qwen3AlignmentTest -Dqwen3.align=true -DargLine="-Xmx6g"
```

## 动手实验清单（从易到难）

### 入门（改参数，观察行为）

1. 用 `--quiet` 开关对比引擎调度日志，观察 `running/waiting/freeBlocks` 随 step 变化。
2. 调 `--max-seqs` 和 `--num-blocks`，观察启动日志里的 KV 池显存估算变化。
3. 用 `--random` / `--gpt3` 模式对比：随机权重输出无语义，但流程完全跑通。
4. 改 `temperature`/`top_k`/`top_p`，观察输出多样性变化（`Sampler`）。

### 进阶（读懂 + 小改）

5. 在 `Qwen3Model` 打开 `layerTrace` 钩子，dump 某层 hidden，和 `Qwen3AlignmentTest` 对照。
6. 给 `Matmul` 加一个基准，对比 `VectorDotKernel` 与 `ScalarDotKernel` 的速度差（用 `-XX:-UseSuperWord` 或不加 `--add-modules` 触发回退）。
7. 实现一个新的采样策略（如 typical sampling）并加测试。

### 挑战（补齐引擎能力）

8. **接入前缀共享**：`KVCacheManager.trySharePrefix/registerPrefix` 已实现，但 `LLMEngine.admitNew` 未使用。改造 admit 逻辑，让含相同 system prompt 的请求复用 KV block（注意 `startIdx` 与各层 BlockTable 的一致性）。
9. **实现 preemption**：当前 decode 显存不足直接 `ABORTED`。参考 vLLM 实现"swap 到 CPU"或"recompute"，让长请求在高压下也能完成。
10. **batched 前向**：当前多请求 decode 是逐个串行前向，尝试把同一 step 的多个请求 Q 拼成矩阵做真正的 batched GEMM。
11. **支持更多模型**：仿照 `Qwen3Loader` 写一个新架构的加载器（如 Llama），复用 math/memory/core 层。

## 概念对照表：mini-vllm ↔ vLLM

| 概念 | mini-vllm | vLLM |
|---|---|---|
| 显存块池 | `BlockPool`（Java 堆数组） | GPU 显存 block 池 |
| 页表 | `BlockTable` | per-seq block table |
| 分页注意力 | `Qwen3Attention.decodePaged`（逐 block） | PagedAttention CUDA kernel |
| 调度循环 | `LLMEngine.step`（单线程） | scheduler + 多 GPU worker |
| 请求队列 | `Scheduler`（waiting/running） | vLLM Scheduler |
| 前缀共享 | 哈希指纹（接口就绪，未接入） | RadixAttention 前缀树 |
| 显存不足 | ABORTED（无抢占） | preemption（swap/recompute） |
| API server | `OpenAiHandler`（JDK HttpServer） | OpenAI-compatible server |

## 常见误区提醒

- **RoPE 是 half-split 不是交错式**：与 HF 对齐时最常见的坑（见 [Transformer](Transformer.md)）。
- **QK-Norm 顺序是 Norm→RoPE**，反了数值就不对。
- **KVCacheManager 的 dModel 字段实为 kvDim**：GQA 下每 token 只存 `nKVHead×headDim`。
- **默认关闭思考模式**：否则短请求容易被 max_tokens 截断在推理中途（见 [OpenAI-API](OpenAI-API.md)）。
- **EOS token 不外泄**：`emitToken` 遇到停止 token 直接 return，避免 `<|im_end|>` 污染输出与多轮上下文。
- **随机模式输出无意义**：仅验证流程，真实对话请用默认 Qwen3 模式。

## 推荐延伸阅读（外部）

- vLLM 论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP'23)
- Qwen3 技术报告与 HuggingFace `transformers` 的 `Qwen3` 实现（对拍参考）
- RoPE 原始论文：*RoFormer: Enhanced Transformer with Rotary Position Embedding*

## 回到目录

- [Home](Home.md) — Wiki 首页与导航
- [Architecture](Architecture.md) — 全局架构
