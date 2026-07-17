# mini-vllm Wiki

> 用纯 Java（零外部依赖）从零实现的学习型 vLLM 引擎：PagedAttention + Continuous Batching + CPU Transformer 推理，默认加载真实 Qwen3-0.6B 权重。

本 Wiki 基于**当前项目代码**逐文件梳理而成，覆盖从编译运行到底层算子的全部实现细节。所有代码引用均可在 `src/main/java/io/leavesfly/minivllm/` 下找到对应文件。

## 文档导航

| 文档 | 主题 | 适合场景 |
|---|---|---|
| [Getting-Started](Getting-Started.md) | 编译、构建、运行、启动参数 | 第一次跑起来 |
| [Architecture](Architecture.md) | 分层架构、模块关系、数据流 | 建立全局认知 |
| [PagedAttention](PagedAttention.md) | 分页注意力机制与逐 block 累加 | 理解省显存的核心 |
| [Continuous-Batching](Continuous-Batching.md) | admit→decode→sweep 调度循环 | 理解高吞吐的核心 |
| [Transformer](Transformer.md) | Qwen3 前向：RMSNorm/RoPE/GQA/SwiGLU/QK-Norm | 理解模型计算 |
| [KV-Cache-Memory](KV-Cache-Memory.md) | 分页分配、回收、前缀共享 | 理解内存管理 |
| [Weight-Loading](Weight-Loading.md) | safetensors 解析、BF16、权重映射、自动下载 | 理解权重加载 |
| [OpenAI-API](OpenAI-API.md) | HTTP 服务、SSE 流式、ChatML | 理解对外接口 |
| [Learning-Path](Learning-Path.md) | 推荐阅读顺序与扩展方向 | 规划学习路线 |

## 项目一句话

mini-vllm 不是生产级引擎，而是一座"可运行、可阅读、可改造"的教学实验室。它用 Java 堆数组模拟 GPU 显存、用逐 block 循环模拟 PagedAttention CUDA kernel、用单线程 step 循环模拟 vLLM 的 scheduler+worker，让你通过代码而非论文理解 vLLM 为什么快、为什么省显存。

## 关键事实速查

| 维度 | 值 |
|---|---|
| 语言 / 构建 | Java 17 / Maven（零业务依赖，仅 JUnit 5 测试） |
| 默认模型 | Qwen3-0.6B（28 层 / hidden 1024 / 16 Q 头 + 8 KV 头 / headDim 128 / vocab 151936，约 596M 参数） |
| 入口类 | `io.leavesfly.minivllm.MiniVllmServer` |
| 默认端口 | 8080 |
| 兼容接口 | `/v1/chat/completions`、`/v1/completions`、`/v1/models` |
| 加速 | `DotKernel` 基于 JDK Vector API（incubator），缺失自动回退标量 |

从 [Getting-Started](Getting-Started.md) 开始。
