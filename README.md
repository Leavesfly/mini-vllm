# mini-vllm

> 用纯 Java（零外部依赖）从零实现的学习型 vLLM 引擎，吃透 PagedAttention、Continuous Batching 与 Transformer 推理的底层原理。

## 项目简介

mini-vllm 是一个**学习型项目**，用 Java + Maven（仅 JDK 标准库，无任何第三方依赖）重新实现 vLLM 的核心机制。它不是生产级推理引擎，而是一座"可运行、可阅读、可改造"的教学实验室——让你通过代码而非论文，真正理解 vLLM 为什么快、为什么省显存。

## 核心特性

- **PagedAttention**：`BlockPool` + `BlockTable` 按需分页分配 KV cache，含引用计数与前缀共享接口
- **Continuous Batching**：`LLMEngine` 的 admit→decode→sweep 三段循环，请求随时进出
- **极简 CPU Transformer**：手写 matmul/softmax/layernorm/gelu/attention，prefill + decode 完整前向
- **权重加载**：纯 Java 解析 safetensors 二进制格式，加载真实微模型权重
- **OpenAI 兼容 API**：JDK HttpServer 实现 `/v1/chat/completions`，支持 SSE 流式输出
- **随机初始化兜底**：无权重文件也能启动跑通全流程

## 技术栈

| 维度 | 选型 |
|---|---|
| 语言 | Java 17 |
| 构建 | Maven（零业务依赖） |
| HTTP | JDK 内置 `com.sun.net.httpserver.HttpServer` |
| 模型尺寸 | vocab=256, dModel=64, nHead=4, nLayer=2（学习用小模型） |

## 架构总览

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
│  Tensor · Matmul · Softmax · LayerNorm · Gelu        │  纯 Java 张量算子
├─────────────────────────────────────────────────────┤
│  辅助层                                               │
│  weights · tokenizer · json                          │  权重加载 / 分词 / JSON
└─────────────────────────────────────────────────────┘
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

### 启动（随机初始化模式）

无需任何权重文件，用随机初始化模型即可跑通全流程：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --random --port 8088
```

### 启动（加载真实权重）

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --weights ./model.safetensors --port 8088
```

### 测试 API

```bash
# 查看模型列表
curl -s http://localhost:8088/v1/models

# 非流式补全
curl -s -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":false}'

# 流式补全（SSE）
curl -s -N -X POST http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"stream":true}'
```

### 使用 OpenAI SDK 连接

接口完全兼容 OpenAI，可直接用官方 SDK：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8088/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mini-vllm",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=32,
)
print(resp.choices[0].message.content)
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | 8080 | HTTP 服务端口 |
| `--weights` | 无 | safetensors 权重文件路径 |
| `--random` | false | 使用随机初始化模型 |
| `--max-seqs` | 8 | 最大并发请求数（continuous batching 上限） |
| `--num-blocks` | 1024 | KV cache block 池大小 |
| `--quiet` | false | 关闭引擎每步调度日志 |

## 项目结构

```
src/main/java/minivllm/
├── api/            # OpenAI 兼容 HTTP 服务
├── core/           # LLMEngine 引擎与 Continuous Batching 调度
├── model/          # Transformer 模型前向计算
├── memory/         # PagedAttention KV cache 内存管理
├── math/           # 纯 Java 张量运算算子
├── tokenizer/      # 分词器
├── weights/        # safetensors 权重加载
├── json/           # 手写 JSON 编解码
└── MiniVllmServer.java  # 入口
```

## 与真实 vLLM 的对应

| mini-vllm | vLLM | 差异 |
|---|---|---|
| `BlockPool` | GPU 显存 block 池 | 用 Java 堆数组模拟显存 |
| `BlockTable` | per-seq block table | 一致（逻辑→物理映射） |
| `Attention.decodePaged` | PagedAttention CUDA kernel | 用 Java 逐 block 累加 |
| `LLMEngine.step` | scheduler + worker loop | 单线程，vLLM 多 GPU 并行 |
| `Scheduler` | vLLM Scheduler | 一致（waiting/running 队列） |
| `OpenAiHandler` | OpenAI API server | 用 JDK HttpServer |

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

本项目为学习用途。
