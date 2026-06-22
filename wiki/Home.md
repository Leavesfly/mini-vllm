# mini-vllm Wiki

> 用纯 Java（零外部依赖）从零实现一个学习型 vLLM 引擎，吃透 PagedAttention、Continuous Batching 与 Transformer 推理的底层原理。

## 这是什么

mini-vllm 是一个**学习型项目**，用 Java + Maven（仅 JDK 标准库，无任何第三方依赖）重新实现 vLLM 的核心机制。它不是生产级推理引擎，而是一座"可运行、可阅读、可改造"的教学实验室——让你通过代码而非论文，真正理解 vLLM 为什么快、为什么省显存。

vLLM 的两大灵魂：
- **PagedAttention**：把 KV cache 切成固定小块按需分配，像操作系统管理虚拟内存一样管理显存。
- **Continuous Batching**：请求随时进出，每个 decode 步都能换入新请求、换出完成请求，GPU 始终满载。

本项目用纯 Java 把这两个机制真实跑通，并附带一个极简 CPU Transformer，让 prefill→decode 全链路可端到端验证。

## 为什么用 Java + 零依赖

| 维度 | 说明 |
|---|---|
| 为什么不直接读 vLLM 源码 | vLLM 是 Python + C++/CUDA，门槛高；把调度/内存逻辑抽离出来用 Java 重写，反而更清晰地暴露机制本质 |
| 为什么零依赖 | 不引 Spring/DJL/Netty，强制手写 HTTP、JSON、张量运算，学习收益最大化 |
| 为什么能跑通 | GPU kernel 用 Java 堆数组模拟；Transformer 用极小尺寸（d=64），CPU 秒出结果 |

## 核心特性

- ✅ **PagedAttention**：`BlockPool` + `BlockTable` 按需分页分配 KV cache，含引用计数与前缀共享接口
- ✅ **Continuous Batching**：`LLMEngine` 的 admit→decode→sweep 三段循环，请求随时进出
- ✅ **极简 Transformer**：手写 matmul/softmax/layernorm/gelu/attention，prefill + decode 完整前向
- ✅ **权重加载**：纯 Java 解析 safetensors 二进制格式，加载真实微模型权重
- ✅ **OpenAI 兼容 API**：JDK HttpServer 实现 `/v1/chat/completions`，支持 SSE 流式
- ✅ **随机初始化兜底**：无权重文件也能启动跑通全流程

## Wiki 导航

| 文档 | 内容 |
|---|---|
| [Getting Started](Getting-Started.md) | 编译、运行、curl 测试 |
| [Architecture](Architecture.md) | 分层架构与数据流 |
| [PagedAttention](PagedAttention.md) | 分页注意力机制详解（核心） |
| [Continuous Batching](Continuous-Batching.md) | 连续批处理调度循环 |
| [Transformer](Transformer.md) | 极简 CPU Transformer 前向实现 |
| [KV Cache Memory](KV-Cache-Memory.md) | KV cache 内存管理 |
| [Weight Loading](Weight-Loading.md) | safetensors 解析与模型组装 |
| [OpenAI API](OpenAI-API.md) | HTTP 服务与流式输出 |
| [Learning Path](Learning-Path.md) | 推荐学习顺序与扩展方向 |

## 技术栈

- **语言**：Java 17
- **构建**：Maven（仅 `maven-compiler-plugin`，零业务依赖）
- **HTTP**：JDK 内置 `com.sun.net.httpserver.HttpServer`
- **模型尺寸**：vocab=256, dModel=64, nHead=4, nLayer=2（学习用小模型）

## 项目状态

- 编译：`mvn compile` ✅ BUILD SUCCESS（29 源文件）
- 端到端验证：`/v1/models`、非流式补全、流式 SSE 补全全部通过 ✅

---

**下一步**：从 [Getting Started](Getting-Started.md) 开始，5 分钟跑通引擎。
