# mini-vllm

> 用纯 Java（零外部依赖）从零实现的学习型 vLLM 引擎：PagedAttention + Continuous Batching + CPU Transformer 推理，默认加载真实 Qwen3-0.6B 权重，可对话、可阅读、可改造。

## 项目简介

mini-vllm 是一个**学习型项目**，用 Java + Maven（仅 JDK 标准库，无任何第三方业务依赖）重新实现 vLLM 的核心机制。它不是生产级推理引擎，而是一座"可运行、可阅读、可改造"的教学实验室——让你通过代码而非论文，真正理解 vLLM 为什么快、为什么省显存。

默认模式下，它会自动下载并加载真实的 **Qwen3-0.6B** 模型（BF16 safetensors 权重 + byte-level BPE 分词器 + ChatML 模板），在纯 CPU 上完成可对话的推理；同时也保留随机初始化的 GPT-2/GPT-3 微模型模式，用于无权重快速验证全流程。

## 核心特性

- **真实模型推理**：完整实现 Qwen3 架构（RMSNorm、RoPE、GQA、SwiGLU、QK-Norm），加载 Qwen3-0.6B 真实权重，与 HuggingFace 参考输出数值对齐
- **PagedAttention**：`BlockPool` + `BlockTable` 按需分页分配 KV cache，含引用计数与前缀共享接口
- **Continuous Batching**：`LLMEngine` 的 admit → decode → sweep 三段循环，请求随时进出，KV 显存不足时回滚等待
- **双模型架构**：除 Qwen3 外，内置 GPT-2/GPT-3 学习微模型（含 GPT-3 交替稠密/局部带状稀疏注意力、标准规模预设）
- **BPE 分词器**：纯 Java 实现 byte-level BPE（vocab.json + merges.txt），ChatML 对话模板，增量 UTF-8 解码避免流式乱码
- **权重加载**：纯 Java 解析 safetensors 二进制（含 BF16 解码）；模型自动下载（ModelScope 优先、HF 兜底，断点续传 + 原子改名）
- **OpenAI 兼容 API**：JDK HttpServer 实现 `/v1/chat/completions`（SSE 流式）、`/v1/completions`、`/v1/models`
- **内置 Web 对话页**：零前端依赖的单页应用，开箱即用
- **SIMD 加速**：`DotKernel` 基于 JDK Vector API（incubator），缺失时自动回退标量实现
- **随机初始化兜底**：无权重文件也能启动跑通全流程

## 技术栈

| 维度 | 选型 |
|---|---|
| 语言 | Java 17（`jdk.incubator.vector` 可选加速） |
| 构建 | Maven（零业务依赖，仅 JUnit 5 用于测试） |
| HTTP | JDK 内置 `com.sun.net.httpserver.HttpServer` |
| 默认模型 | Qwen3-0.6B（28 层 / hidden 1024 / 16 Q 头 + 8 KV 头 / headDim 128 / vocab 151936） |
| 学习微模型 | vocab=256, dModel=64, nHead=4, nLayer=2（GPT-2 风格，可切 GPT-3 nano） |

## 架构总览

```
┌──────────────────────────────────────────────────────────┐
│  API 层 (api)                                             │
│  OpenAiHandler · SseWriter · WebUiHandler                 │  OpenAI 兼容 HTTP + SSE 流式 + Web 对话页
├──────────────────────────────────────────────────────────┤
│  Core 层 (core)                                           │
│  LLMEngine · Scheduler · Sequence                         │  continuous batching 调度
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

## 快速开始

### 前置要求

- JDK 17+
- Maven 3.6+

### 编译

```bash
mvn compile
```

> 零外部依赖，无需下载任何第三方库，编译通常在 1 秒内完成。

### 启动（默认：Qwen3-0.6B）

不传任何参数即默认加载 Qwen3-0.6B：优先复用本地缓存，缺失时自动下载（约 1.5GB）：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --port 8080
```

指定本地模型目录（跳过下载）：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --model-dir ./models/Qwen3-0.6B --port 8080
```

也可以打包成 jar 运行（manifest 已声明 Vector API 模块）：

```bash
mvn package
java -jar target/mini-vllm-0.0.1-SNAPSHOT.jar --port 8080
```

### 启动（随机初始化学习模式）

无需任何权重文件，用随机初始化的 GPT 微模型跑通 PagedAttention + batching 全流程：

```bash
# GPT-2 风格微模型
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --random --port 8080

# GPT-3 nano（交替稠密/稀疏注意力）
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --gpt3 --port 8080
```

### Web 对话页面

启动后浏览器访问 `http://localhost:8080/`，内置单页对话应用，支持流式输出与多轮对话。

### 测试 API

```bash
# 查看模型列表
curl -s http://localhost:8080/v1/models

# 非流式补全
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":false}'

# 流式补全（SSE）
curl -s -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hi"}],"max_tokens":32,"stream":true}'

# 开启 Qwen3 思考模式（输出 <think>...</think> 推理过程）
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"9.11 和 9.9 哪个大"}],"enable_thinking":true}'
```

支持的请求参数：`messages`、`stream`、`max_tokens`（默认 512）、`temperature`（默认 0.8）、`top_p`（默认 0.9）、`top_k`、`enable_thinking`（默认 false）。

### 使用 OpenAI SDK 连接

接口完全兼容 OpenAI，可直接用官方 SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mini-vllm",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | 8080 | HTTP 服务端口 |
| `--model-dir` | 无 | 本地 Qwen3 模型目录（含 config.json / model.safetensors / 分词器文件） |
| `--model-repo` | Qwen/Qwen3-0.6B | 自动下载的模型仓库名 |
| `--mirror` | auto（env `MINIVLLM_MIRROR`） | 下载源：`auto` / `hf` / `modelscope` |
| `--weights` | 无 | 学习模式 GPT 微模型的 safetensors 权重路径 |
| `--tokenizer-dir` | 无 | 学习模式使用 BPE 分词器的模型目录（缺省用 ByteTokenizer） |
| `--random` | false | 随机初始化（Qwen3 模式下也可用于验证流程） |
| `--gpt3` | false | 学习模式使用 GPT-3 nano（交替稀疏注意力） |
| `--max-seqs` | Qwen3=2 / 学习模式=8 | 最大并发请求数（continuous batching 上限） |
| `--num-blocks` | 自动估算 | KV cache block 池大小 |
| `--max-seq-len` | 2048 | Qwen3 上下文上限（RoPE 表与 KV 池按此分配） |
| `--quiet` | false | 关闭引擎每步调度日志 |

## 模型下载与缓存

`ModelDownloader` 按以下顺序解析模型目录，命中即返回：

1. 项目内 `./models/<name>`（如 `./models/Qwen3-0.6B`，最优先）
2. HuggingFace 缓存 `~/.cache/huggingface/hub`
3. mini-vllm 本地缓存 `~/.cache/mini-vllm/models`
4. 在线下载（ModelScope 优先，HuggingFace 兜底）

下载可靠性：`.part` 临时文件 + 完成后原子改名；`Range` 请求头断点续传，大文件中断后不必重下。

## 项目结构

```
src/main/java/io/leavesfly/minivllm/
├── api/            # OpenAI 兼容 HTTP 服务 + SSE + Web 对话页
├── core/           # LLMEngine 引擎与 Continuous Batching 调度
├── model/          # Qwen3 / GPT-2 / GPT-3 模型前向计算（RoPE、GQA、SwiGLU）
├── memory/         # PagedAttention KV cache 内存管理
├── math/           # 纯 Java 张量算子（含 Vector API SIMD 点积）
├── tokenizer/      # byte-level BPE 分词器、ChatML 模板、字节/字符分词器
├── weights/        # safetensors 解析（含 BF16）、模型下载与组装
├── json/           # 手写 JSON 编解码
├── tools/          # Benchmark 压测工具
└── MiniVllmServer.java  # 入口

src/main/resources/web/    # 内置 Web 对话单页
src/test/                  # JUnit 5 单测 + HF 数值对齐测试
tools/                     # HF 参考数据导出脚本（Python，供对齐测试用）
models/Qwen3-0.6B/         # 默认模型本地缓存（已 gitignore）
```

## 与真实 vLLM 的对应

| mini-vllm | vLLM | 差异 |
|---|---|---|
| `BlockPool` | GPU 显存 block 池 | 用 Java 堆数组模拟显存 |
| `BlockTable` | per-seq block table | 一致（逻辑→物理映射） |
| `Attention.decodePaged` | PagedAttention CUDA kernel | 用 Java 逐 block 累加 |
| `LLMEngine.step` | scheduler + worker loop | 单线程，vLLM 多 GPU 并行 |
| `Scheduler` | vLLM Scheduler | 一致（waiting/running 队列），学习版不做 preemption |
| `OpenAiHandler` | OpenAI API server | 用 JDK HttpServer |

## 测试与数值对齐

```bash
mvn test
```

测试覆盖：core 调度与端到端生成、memory 分页分配、math 全部算子、tokenizer（含 HF 参考用例）、Qwen3 模型前向。其中 `Qwen3AlignmentTest` 使用 `tools/dump_reference.py` 从 HuggingFace transformers 导出的参考数据（`src/test/resources/qwen3/`），逐层校验 Java 实现与 HF 的数值一致性。

## 文档

详细文档见 [wiki/](wiki/) 目录：

- [Getting Started](wiki/Getting-Started.md) — 编译、运行、curl 测试
- [Architecture](wiki/Architecture.md) — 分层架构与数据流
- [PagedAttention](wiki/PagedAttention.md) — 分页注意力机制详解
- [Continuous Batching](wiki/Continuous-Batching.md) — 连续批处理调度循环
- [Transformer](wiki/Transformer.md) — 极简 CPU Transformer 前向实现
- [KV Cache Memory](wiki/KV-Cache-Memory.md) — KV cache 内存管理
- [Weight Loading](wiki/Weight-Loading.md) — safetensors 解析与模型组装
- [OpenAI API](wiki/OpenAI-API.md) — HTTP 服务与流式输出
- [Learning Path](wiki/Learning-Path.md) — 推荐学习顺序与扩展方向

## License

本项目为学习用途。Qwen3-0.6B 模型权重遵循其原始 License（见 `models/Qwen3-0.6B/LICENSE`）。
