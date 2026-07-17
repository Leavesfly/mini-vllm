# Architecture

本文自顶向下梳理 mini-vllm 的分层架构、模块职责、依赖关系与数据流向。读完你会知道一条请求从 HTTP 进来到文本流式返回，途中经过哪些模块、每个模块做了什么。

## 分层总览

```
┌──────────────────────────────────────────────────────────┐
│  API 层 (api)                                             │
│  OpenAiHandler · SseWriter · WebUiHandler                 │  OpenAI 兼容 HTTP + SSE 流式 + Web 对话页
├──────────────────────────────────────────────────────────┤
│  Core 层 (core)                                           │
│  LLMEngine · Scheduler · Sequence                         │  Continuous Batching 调度
├──────────────────────────────────────────────────────────┤
│  Model 层 (model)                                         │
│  Qwen3Model/Block/Attention · TransformerModel · RoPE     │  Qwen3 与 GPT-2/3 两套前向
├──────────────────────────────────────────────────────────┤
│  Memory 层 (memory)                                       │
│  KVCacheManager · BlockPool · BlockTable                  │  PagedAttention 内存管理
├──────────────────────────────────────────────────────────┤
│  Math 层 (math)                                           │
│  Tensor · Matmul · DotKernel · RmsNorm · Softmax · Sampler│  纯 Java 张量算子（含 SIMD）
├──────────────────────────────────────────────────────────┤
│  辅助层                                                    │
│  weights · tokenizer · json                               │  safetensors/下载 · BPE/ChatML · JSON
└──────────────────────────────────────────────────────────┘
```

依赖方向严格自上而下：上层依赖下层，下层不反向依赖上层。这让每一层都能独立测试（见各层的 `*Test`）。

## 包结构与职责

包根：`io.leavesfly.minivllm`

| 包 | 关键类 | 职责 |
|---|---|---|
| （根） | `MiniVllmServer` | 入口：解析参数、装配模型、启动引擎与 HTTP |
| `api` | `OpenAiHandler`、`SseWriter`、`WebUiHandler` | OpenAI 兼容端点、SSE 流式、静态页 |
| `core` | `LLMEngine`、`Scheduler`、`Sequence` | Continuous Batching 调度与请求状态机 |
| `model` | `LlmModel`、`Qwen3Model`、`TransformerModel`、`Qwen3Block/Attention`、`SwiGluFfn`、`RotaryEmbedding`、`Linear`、`Embedding`、`ModelConfig` | 模型前向计算 |
| `memory` | `KVCacheManager`、`BlockPool`、`BlockTable` | PagedAttention KV cache 内存管理 |
| `math` | `Matmul`、`DotKernel`、`RmsNorm`、`LayerNorm`、`Softmax`、`Silu`、`Gelu`、`Sampler`、`Tensor` | 纯 Java 数值算子 |
| `weights` | `SafetensorsLoader`、`Bf16`、`Qwen3Loader`、`ModelLoader`、`ModelDownloader` | 权重解析、格式转换、组装、下载 |
| `tokenizer` | `BpeTokenizer`、`ByteLevelBpe`、`ChatTemplate`、`ByteTokenizer`、`SimpleTokenizer` | 分词与对话模板 |
| `json` | `SimpleJson` | 零依赖手写 JSON 编解码 |

## 核心抽象：LlmModel 接口

引擎与具体模型解耦的关键是 [LlmModel](../src/main/java/io/leavesfly/minivllm/model/LlmModel.java) 接口。引擎只关心三件事：

```java
public interface LlmModel {
    // 处理整段 prompt，写入各层 KV cache，返回最后位置 logits
    float[] prefillLogits(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx);
    // 处理单个新 token，复用 KV cache，返回当前位置 logits
    float[] decodeLogits(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts);
    ModelConfig config();
    long numParameters();
}
```

有两套实现，引擎对其无感：

- [Qwen3Model](../src/main/java/io/leavesfly/minivllm/model/Qwen3Model.java)：RMSNorm + RoPE + GQA + SwiGLU + QK-Norm，加载真实 Qwen3-0.6B 权重
- [TransformerModel](../src/main/java/io/leavesfly/minivllm/model/TransformerModel.java)：GPT-2/GPT-3 风格（LayerNorm + 学习式位置嵌入 + MHA + GELU-MLP，GPT-3 含交替稀疏注意力）

两套实现都同时提供**引擎路径**（prefill/decode，带 PagedAttention KV cache）和**PyTorch 风格路径**（`forward()` 无状态整段前向，用于与 HF 对拍）。

## 数据流：一条请求的完整旅程

以 `POST /v1/chat/completions`（stream=true）为例：

```
HTTP 请求
   │  [api] OpenAiHandler.handleChatCompletions
   │    · SimpleJson.parseObject 解析 body
   │    · extractPrompt: messages → ChatTemplate.applyChatML（ChatML 提示词）
   │    · 读取 stream / max_tokens / temperature / top_p / top_k / enable_thinking
   ▼
[core] LLMEngine.addRequest(prompt, ...)
   │    · BpeTokenizer.encode(prompt) → int[] promptTokens
   │    · new Sequence(...)：每层一张 BlockTable，注入增量解码器 onToken 回调
   │    · Scheduler.add(seq) → waiting 队列（并发安全）
   ▼
[core] 引擎线程 step() 循环：admit → decode → sweep
   │  admitNew：
   │    · KVCacheManager.ensureCapacity 为各层分配 block（显存不足则回滚等待）
   │    · model.prefillLogits(...) 一次性前向整段 prompt
   │    · Sampler.sample(logits) → 首个 token，进入 running
   │  decodeStep：对每个 running 请求
   │    · model.decodeLogits(lastToken, curIdx, ...) 逐 token 前向（PagedAttention）
   │    · Sampler.sample → 下一个 token
   │    · emitToken：增量解码 → onToken 回调
   │  sweepFinished：命中 EOS/达到 maxTokens → 释放 KV cache，移出 running
   ▼
[api] onToken 回调 → SseWriter.write（每 token 一个 SSE chunk）
   ▼
HTTP 流式响应（... data: {...} ... data: [DONE]）
```

其中 model 前向内部又向下调用 memory 层（读写 KV cache）与 math 层（matmul/softmax/norm/采样）。

## 模块协作关系图

```
                MiniVllmServer（装配）
                       │
        ┌──────────────┼───────────────┐
        ▼              ▼                ▼
   OpenAiHandler   WebUiHandler      （启动 HttpServer）
        │
        ▼
     LLMEngine ──────── Scheduler（waiting/running 队列）
        │  持有                │ 管理
        │                      ▼
        │                  Sequence（promptTokens/outputTokens/BlockTable[]/onToken）
        ├──────────────► LlmModel（Qwen3Model / TransformerModel）
        │                      │ 调用
        │                      ├──► KVCacheManager ──► BlockPool ──► KVBlock[]（模拟显存）
        │                      │                  └──► BlockTable（逻辑→物理映射）
        │                      └──► math：Matmul(DotKernel) / RmsNorm / Softmax / RoPE / SwiGLU
        └──────────────► Sampler（temperature/top-k/top-p）
```

## 线程模型

mini-vllm 是**单引擎线程 + 多 HTTP 线程**的解耦设计：

- **引擎线程**（`mini-vllm-engine`，守护线程）：`LLMEngine.start()` 启动，持续 `step()`，独占推进所有请求的 prefill/decode。这对应 vLLM 的 scheduler+worker，但这里是单线程串行（学习版不做多 GPU 并行）。
- **HTTP 线程池**（`Executors.newFixedThreadPool(max(16, maxSeqs*4))`）：每个请求一个线程，负责解析请求、`addRequest` 入队、然后**阻塞轮询** `seq.isFinished()`，通过 `onToken` 回调写 SSE。
- **Matmul 线程池**（`mini-vllm-matmul`，按 CPU 核数）：引擎线程内的矩阵乘按输出行分块并行，加速单请求前向。

跨线程数据交接点：
- `Scheduler.waiting` 用 `ConcurrentLinkedDeque`（HTTP 线程写、引擎线程读）
- `Sequence.stage` 用 `volatile`（引擎线程写、HTTP 线程读判完成）
- `SseWriter.write` 用 `synchronized`（onToken 在引擎线程触发，与 handler 线程并发写）

## 与真实 vLLM 的对应

| mini-vllm | vLLM | 差异 |
|---|---|---|
| `BlockPool` | GPU 显存 block 池 | 用 Java 堆数组模拟显存 |
| `BlockTable` | per-seq block table | 一致（逻辑→物理映射） |
| `Qwen3Attention.decodePaged` | PagedAttention CUDA kernel | 用 Java 逐 block 累加 |
| `LLMEngine.step` | scheduler + worker loop | 单线程，vLLM 多 GPU 并行 |
| `Scheduler` | vLLM Scheduler | 一致（waiting/running 队列），学习版不做 preemption |
| `KVCacheManager.trySharePrefix` | RadixAttention/PrefixCache | 简化为 block 级哈希指纹 |
| `OpenAiHandler` | OpenAI API server | 用 JDK HttpServer |

## 设计取舍（为什么这样写）

1. **零外部依赖**：只用 JDK 标准库（含 `com.sun.net.httpserver`、`jdk.incubator.vector`），让代码"可阅读、可改造"，不被框架细节淹没。
2. **两套模型并存**：Qwen3 提供真实可对话体验，GPT 微模型提供无权重快速验证——降低学习门槛。
3. **CPU 逐 block 模拟 PagedAttention**：牺牲性能换取可读性，让 KV cache 的分页语义清晰可见。
4. **单线程调度**：把 Continuous Batching 的 admit/decode/sweep 三段逻辑摊平在一个 `step()` 里，一眼看懂。

## 继续深入

- 内存机制：[PagedAttention](PagedAttention.md) 与 [KV-Cache-Memory](KV-Cache-Memory.md)
- 调度机制：[Continuous-Batching](Continuous-Batching.md)
- 模型计算：[Transformer](Transformer.md)
- 权重加载：[Weight-Loading](Weight-Loading.md)
- 对外接口：[OpenAI-API](OpenAI-API.md)
