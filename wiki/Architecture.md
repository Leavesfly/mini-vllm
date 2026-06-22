# Architecture

mini-vllm 的分层架构与请求生命周期。

## 分层总览

```
┌─────────────────────────────────────────────────────┐
│  API 层 (api)                                        │
│  OpenAiHandler · SseWriter                           │  OpenAI 兼容 HTTP + SSE 流式
├─────────────────────────────────────────────────────┤
│  Core 层 (core)                                      │
│  LLMEngine · Scheduler · Sequence                    │  continuous batching 调度
├─────────────────────────────────────────────────────┤
│  Model 层 (model)                                    │
│  Transformer · Attention · Ffn · Block · Embedding   │  极简 CPU Transformer 前向
├─────────────────────────────────────────────────────┤
│  Memory 层 (memory)                                  │
│  KVCacheManager · BlockPool · BlockTable             │  PagedAttention 内存管理
├─────────────────────────────────────────────────────┤
│  Math 层 (math)                                      │
│  Tensor · Matmul · Softmax · LayerNorm · Gelu · Sampler │  纯 Java 张量算子
├─────────────────────────────────────────────────────┤
│  辅助层                                               │
│  weights(SafetensorsLoader/ModelLoader)              │  权重加载与模型组装
│  tokenizer(Byte/CharTokenizer)                       │  分词
│  json(SimpleJson)                                    │  手写 JSON 编解码
└─────────────────────────────────────────────────────┘
```

## 依赖关系

各层**只向下依赖**，不反向引用：

- `api` → `core`、`json`
- `core` → `model`、`memory`、`math`、`tokenizer`
- `model` → `math`、`memory`
- `memory` → 无（纯数据结构）
- `math` → 无（纯计算）

这种单向依赖让每层可独立理解与测试。

## 请求生命周期

一个请求从到达到完成，经历以下阶段：

```
HTTP POST /v1/chat/completions
        │
        ▼
OpenAiHandler.handle()
   ├─ 解析 JSON（SimpleJson）
   ├─ 从 messages 提取 prompt
   ├─ tokenizer.encode(prompt) → promptTokens
   └─ engine.addRequest(promptTokens, onToken)
        │
        ▼
Sequence 创建（持有 BlockTable[nLayer]）→ Scheduler.waiting 队列
        │
        ▼  (engine 线程持续 step)
LLMEngine.step() 三段循环：
   ┌─ admitNew() ──────────────────────────────┐
   │  从 waiting 取请求                         │
   │  kvMgr.ensureCapacity(每层 BlockTable)     │
   │  model.prefill(promptTokens) → 第一个 token │
   │  sampler.sample(logits)                    │
   │  加入 running 队列                          │
   └────────────────────────────────────────────┘
   ┌─ decodeStep() ─────────────────────────────┐
   │  对每个 running 请求：                      │
   │  kvMgr.ensureCapacity(新增 token)          │
   │  model.decode(lastToken) → PagedAttention  │  ← 从 BlockTable 读散落 KV
   │  sampler.sample(logits) → 新 token          │
   │  onToken 回调（推 SSE 或收集）              │
   └────────────────────────────────────────────┘
   ┌─ sweepFinished() ──────────────────────────┐
   │  检查 isFinished()（maxTokens/EOS）         │
   │  kvMgr.free(每层 BlockTable) 释放引用       │
   │  移出 running                               │
   └────────────────────────────────────────────┘
        │
        ▼
（流式）SseWriter 逐 token 推送 → [DONE]
（非流式）收集完成 → 返回完整 JSON
```

## 线程模型

mini-vllm 有两类线程协作：

| 线程 | 职责 | 涉及类 |
|---|---|---|
| HTTP 线程池 | 接收请求、addRequest、阻塞等 Sequence 完成 | `OpenAiHandler`、`SseWriter` |
| Engine 线程（单） | 持续跑 `step()` 循环，驱动 prefill/decode | `LLMEngine` |

关键设计：
- `Scheduler.waiting` 用 `ConcurrentLinkedDeque`，HTTP 线程安全 add，engine 线程 poll。
- `LLMEngine.step()` 由单线程执行，避免并发修改 running 队列与 KV cache。
- `onToken` 回调在 engine 线程触发，写 SSE 需 `synchronized`（见 `SseWriter`）。

## KV cache 与模型层的关系

Transformer 有 `nLayer` 层，**每层有独立的 KV cache**。因此每个 `Sequence` 持有 `BlockTable[nLayer]`，前向时第 i 层用 `bts[i]`：

```
Transformer.prefill(tokenIds, kvMgr, bts[0..nLayer-1], startIdx)
  for i in 0..nLayer-1:
      blocks[i].prefill(x, kvMgr, bts[i])   ← 每层写各自的 KV block
```

`BlockPool` 是全局共享的物理池，所有层、所有请求的 block 都从中分配，靠 `BlockTable` 区分归属。

## 与真实 vLLM 的对应

| mini-vllm | vLLM | 差异 |
|---|---|---|
| `BlockPool` | GPU 显存 block 池 | 本项目用 Java 堆数组模拟显存 |
| `BlockTable` | per-seq block table | 一致（逻辑→物理映射） |
| `Attention.decodePaged` | PagedAttention CUDA kernel | 本项目用 Java 逐 block 累加 |
| `LLMEngine.step` | scheduler + worker loop | 本项目单线程，vLLM 多 GPU 并行 |
| `Scheduler` | vLLM Scheduler | 一致（waiting/running 队列） |
| `OpenAiHandler` | OpenAI API server | 本项目用 JDK HttpServer |

---

**核心机制深入**：[PagedAttention](PagedAttention.md) · [Continuous Batching](Continuous-Batching.md)
