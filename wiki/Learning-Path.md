# Learning Path

推荐的学习顺序、动手实验与扩展方向，帮你从 mini-vllm 最大化吸收 vLLM 的核心思想。

## 推荐学习顺序

按"先跑通 → 再读懂 → 后改造"的节奏：

### 阶段 1：跑通（1 小时）
1. 读 [Home](Home.md) 了解项目定位
2. 按 [Getting Started](Getting-Started.md) 编译、启动、curl 测试
3. 观察启动日志中的 `running/waiting/freeBlocks` 变化，感性认识 continuous batching

**目标**：确认全链路跑通，建立整体印象。

### 阶段 2：读懂架构（2 小时）
1. 读 [Architecture](Architecture.md) 理解分层与请求生命周期
2. 浏览 `math` 层（[Matmul](../src/main/java/io/leavesfly/minivllm/math/Matmul.java)/[Softmax](../src/main/java/io/leavesfly/minivllm/math/Softmax.java) 等），理解纯 Java 如何做张量运算
3. 读 [Transformer](Transformer.md) 走通前向流程：prefill 与 decode

**目标**：理解"一个请求如何从 token 变成回答"。

### 阶段 3：吃透核心机制（3 小时）
1. 读 [PagedAttention](PagedAttention.md) —— 最重要的一篇
2. 对照 [KV Cache Memory](KV-Cache-Memory.md) 理解内存管理细节
3. 读 [Continuous Batching](Continuous-Batching.md) 理解 admit→decode→sweep 循环
4. 读 [Attention.java](../src/main/java/io/leavesfly/minivllm/model/Attention.java) 的 `decodePaged` 与 `decodeGather`，对比两种实现

**目标**：真正理解 vLLM 为什么快、为什么省显存。

### 阶段 4：读懂周边（2 小时）
1. [Weight Loading](Weight-Loading.md)：纯 Java 解析 safetensors
2. [OpenAI API](OpenAI-API.md)：JDK HttpServer + SSE 流式

**目标**：理解零依赖如何搞定"加载权重"和"对外服务"。

## 动手实验

边读边做，加深理解：

### 实验 1：对比 decodePaged vs decodeGather
写一个 main，用随机模型对同一 prompt 分别调用 `decodePaged` 和 `decodeGather`，断言两者输出一致，验证 PagedAttention 实现正确。

### 实验 2：观察显存回收
启动时设小 `--num-blocks`（如 64），并发发多个长请求，观察 `freeBlocks` 如何随请求完成回升、admit 如何因显存不足暂停。

### 实验 3：关闭 continuous batching
临时改造 `LLMEngine.step`，让 admit 只在 running 空时执行（模拟静态 batching），对比吞吐差异。

### 实验 4：调整采样参数
用 curl 传不同 `temperature`/`top_p`/`top_k`，观察随机模型输出的分布变化。

### 实验 5：加载真实微模型
用 [Weight Loading](Weight-Loading.md) 的导出脚本，把 nanoGPT Shakespeare 模型转成 safetensors，加载后生成有意义文本。

## 扩展方向

读完代码后，这些是值得动手的进阶改造：

### 1. 启用前缀共享（中等）
当前 `KVCacheManager.trySharePrefix` 已实现但未在 `LLMEngine.admitNew` 中调用。改造要点：
- prefix cache 改为 per-layer（每层独立匹配）
- admitNew 中先 `trySharePrefix`，命中的部分跳过 prefill（startIdx > 0）
- 完成后 `registerPrefix` 注册新 block

参考真实 vLLM 的 RadixAttention 树做精确匹配。

### 2. PagedAttention 正确性测试（简单）
为 `Attention` 写单元测试：构造小输入，对比 `decodePaged` 与 `decodeGather`、对比 prefill 后 decode 与全量 prefill 的 logits。

### 3. 性能优化（中等）
- batch 内多请求 decode 并行（当前串行遍历 running）
- matmul 用分块循环提升 CPU 缓存命中
- 采样器 `indexSort` 改用部分排序（当前 O(n²) 插入排序）

### 4. 支持 BPE Tokenizer（中等）
当前只有 Byte/Char Tokenizer。可实现一个简化版 BPE，或通过子进程调用 Python tokenizer。

### 5. Preemption（困难）
显存不足时，当前直接 abort 请求。真实 vLLM 会抢占（preempt）：把低优先级请求的 KV cache 换出到 CPU 内存，腾出显存给高优先级。实现 preemption 是理解 vLLM 调度深度的关键。

### 6. 多模型支持（中等）
当前单模型。扩展为模型注册表，按请求的 `model` 字段路由到不同 Transformer 实例。

## 对照 vLLM 源码

理解 mini-vllm 后，读 vLLM 源码会顺畅很多。对应关系：

| mini-vllm | vLLM 源码位置 | 
|---|---|
| `BlockPool` | `vllm/core/block_manager.py` |
| `BlockTable` | `vllm/core/block_manager.py` (BlockTable) |
| `KVCacheManager` | `vllm/core/kv_cache_manager.py` |
| `Attention.decodePaged` | `vllm/attention/backends/*.py` + `csrc/attention*.cu` |
| `Scheduler` | `vllm/core/scheduler.py` |
| `LLMEngine.step` | `vllm/engine/llm_engine.py` (step) |
| `Sequence` | `vllm/sequence.py` (SequenceGroup) |
| `OpenAiHandler` | `vllm/entrypoints/openai/api_server.py` |

读 vLLM 时，先在 mini-vllm 找到对应概念，再用 Python/CUDA 知识理解实现细节。

## 学习成果

完成上述路径后，你将真正理解：
- ✅ PagedAttention 如何消除 KV cache 内存浪费
- ✅ Continuous batching 如何保持 GPU 满载
- ✅ Transformer 推理的 prefill/decode 两阶段与 KV cache 复用
- ✅ vLLM 调度器的 waiting/running 队列与显存约束
- ✅ OpenAI 兼容 API 与 SSE 流式的实现
- ✅ 零依赖下如何手写张量运算、JSON、HTTP、权重加载

这些是 vLLM 论文和博客讲不透、必须靠代码才能内化的知识。

---

**回到**：[Home](Home.md)
