# Getting Started

本文覆盖 mini-vllm 的**编译、构建、运行、测试**全流程，以及全部命令行参数与常见问题。目标：让你在几分钟内跑起一个 OpenAI 兼容的本地推理服务。

## 前置要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| JDK | 17+ | 需要 `jdk.incubator.vector` 模块以启用 SIMD 加速（缺失会自动回退标量，不影响功能） |
| Maven | 3.6+ | 仅用于编译与测试，运行时零业务依赖 |
| 磁盘 | ~2 GB | 首次自动下载 Qwen3-0.6B 权重（约 1.5 GB） |
| 内存 | ~4 GB | 加载 Qwen3-0.6B（BF16→F32 展开后约 2.4 GB 权重驻留） |

项目只有一个测试期依赖（JUnit 5），运行期完全依赖 JDK 标准库。见 [pom.xml](../pom.xml)。

## 构建配置说明（pom.xml）

关键配置有两处，都与 [DotKernel](../src/main/java/io/leavesfly/minivllm/math/DotKernel.java) 的 Vector API 相关：

```xml
<!-- 编译期：Vector API 属于 incubator 模块，必须显式加入 -->
<plugin>
  <artifactId>maven-compiler-plugin</artifactId>
  <configuration>
    <release>17</release>
    <compilerArgs>
      <arg>--add-modules</arg>
      <arg>jdk.incubator.vector</arg>
    </compilerArgs>
  </configuration>
</plugin>

<!-- 打包期：jar manifest 声明 Add-Modules，java -jar 运行时可用 SIMD -->
<plugin>
  <artifactId>maven-jar-plugin</artifactId>
  <configuration>
    <archive>
      <manifest><mainClass>io.leavesfly.minivllm.MiniVllmServer</mainClass></manifest>
      <manifestEntries><Add-Modules>jdk.incubator.vector</Add-Modules></manifestEntries>
    </archive>
  </configuration>
</plugin>
```

> 注意：`jdk.incubator.vector` 是孵化模块，编译时通常会有 warning，可忽略。若运行环境缺失该模块，`Matmul` 的静态初始化会捕获异常并回退到标量内核（见下文"SIMD 加速"）。

## 编译

```bash
mvn compile
```

零外部依赖，无需下载第三方库，编译通常在 1 秒内完成。

## 运行

入口类 [MiniVllmServer](../src/main/java/io/leavesfly/minivllm/MiniVllmServer.java) 支持三种模式，通过命令行参数切换。

### 模式一：默认（加载真实 Qwen3-0.6B）

不传任何模型参数即进入 Qwen3 模式。启动流程：

1. `ModelDownloader.resolve()` 按优先级查找模型目录（项目内 `./models/Qwen3-0.6B` → HF 缓存 → mini-vllm 本地缓存 → 在线下载）
2. 从 `config.json` 解析 `ModelConfig`
3. `SafetensorsLoader.load()` 读取 BF16 权重并转 F32
4. `Qwen3Loader.load()` 组装成 `Qwen3Model`（含 shape 与参数量校验）
5. `BpeTokenizer.fromModelDir()` 加载 byte-level BPE 分词器
6. 构建 `KVCacheManager` + `LLMEngine`，启动 HTTP 服务

```bash
# 编译后直接运行（首次会自动下载权重）
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --port 8080

# 指定本地模型目录，跳过下载
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --model-dir ./models/Qwen3-0.6B --port 8080
```

打包成 jar 运行（manifest 已声明 Vector API 模块）：

```bash
mvn package
java -jar target/mini-vllm-0.0.1-SNAPSHOT.jar --port 8080
```

### 模式二：随机初始化学习模型（无需权重）

用随机初始化的 GPT-2/GPT-3 微模型跑通 PagedAttention + Continuous Batching 全流程。输出无语义，但过程完全确定，适合快速验证引擎链路：

```bash
# GPT-2 风格微模型（vocab=256, dModel=64, nHead=4, nLayer=2）
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --random --port 8080

# GPT-3 nano（交替稠密/局部带状稀疏注意力）
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --gpt3 --port 8080
```

### 模式三：加载自定义 GPT 权重

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer \
  --weights ./model.safetensors --tokenizer-dir ./my-tokenizer --port 8080
```

## 命令行参数全集

参数解析见 [MiniVllmServer.main()](../src/main/java/io/leavesfly/minivllm/MiniVllmServer.java)。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--port` | 8080 | HTTP 服务端口 |
| `--model-dir` | 无 | 本地 Qwen3 模型目录（含 config.json / model.safetensors / vocab.json / merges.txt / tokenizer_config.json） |
| `--model-repo` | `Qwen/Qwen3-0.6B` | 自动下载的模型仓库名 |
| `--mirror` | `auto` | 下载源：`auto`（ModelScope 优先，HF 兜底）/ `hf` / `modelscope`；也可用环境变量 `MINIVLLM_MIRROR` |
| `--weights` | 无 | 学习模式 GPT 微模型的 safetensors 权重路径 |
| `--tokenizer-dir` | 无 | 学习模式 BPE 分词器目录（缺省用 `ByteTokenizer` 字节级词表） |
| `--random` | false | 随机初始化（Qwen3 模式下亦可用于验证流程） |
| `--gpt3` | false | 学习模式使用 GPT-3 nano（交替稀疏注意力） |
| `--max-seqs` | Qwen3=2 / 学习模式=8 | 最大并发请求数（Continuous Batching 上限） |
| `--num-blocks` | 自动估算 | KV cache block 池大小 |
| `--max-seq-len` | 2048 | Qwen3 上下文上限（RoPE 表与 KV 池按此分配，会与 config 取小） |
| `--quiet` | false | 关闭引擎每步调度日志 |

### 关于 KV 池大小的自动估算

未显式传 `--num-blocks` 时，`MiniVllmServer` 用如下公式估算，保证满并发不会 OOM：

```
blocksPerSeq = ceil(maxSeqLen / blockSize)
numBlocks    = max(1024, maxNumSeqs * nLayer * blocksPerSeq)
```

注意 KV cache 是**按层独立**的，所以要乘以 `nLayer`。启动日志会打印满载估算显存（Java 堆数组模拟），见 [KV-Cache-Memory](KV-Cache-Memory.md)。

## 测试 API

服务启动后：

```bash
# 模型列表
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

请求参数：`messages`、`stream`、`max_tokens`（默认 512）、`temperature`（默认 0.8）、`top_p`（默认 0.9）、`top_k`、`enable_thinking`（默认 false）。详见 [OpenAI-API](OpenAI-API.md)。

### Web 对话页

浏览器访问 `http://localhost:8080/`，内置零前端依赖的单页对话应用（[WebUiHandler](../src/main/java/io/leavesfly/minivllm/api/WebUiHandler.java) 提供，页面打包在 `src/main/resources/web/index.html`）。

### 用 OpenAI SDK 连接

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

## 运行单元测试

```bash
mvn test
```

测试覆盖各层核心逻辑（源码见 `src/test/java/io/leavesfly/minivllm/`）：

| 测试类 | 验证内容 |
|---|---|
| `math/*Test` | Matmul / Softmax / RmsNorm / Sampler / Tensor 等算子正确性 |
| `memory/BlockPoolTest`、`BlockTableTest`、`KVCacheManagerTest` | 分页分配、引用计数、逻辑→物理映射 |
| `core/LLMEngineE2ETest` | 端到端：maxTokens 约束、greedy 可复现、EOS 提前停止、Continuous Batching 全部完成、流式回调、采样合法性 |
| `tokenizer/BpeTokenizerTest`、`ByteTokenizerTest` | BPE 编解码与 HF 参考用例一致 |
| `model/Qwen3ModelTest`、`RotaryEmbeddingTest` | Qwen3 前向、RoPE 数值 |
| `model/Qwen3AlignmentTest` | 与 HuggingFace Qwen3-0.6B 逐层数值对齐（默认跳过，需真实权重） |

### Qwen3 数值对齐测试（可选）

`Qwen3AlignmentTest` 默认跳过，需显式开启并提供真实权重：

```bash
mvn test -Dtest=Qwen3AlignmentTest -Dqwen3.align=true -DargLine="-Xmx6g"
```

它用 `tools/dump_reference.py` 从 HF transformers 导出的参考数据（`src/test/resources/qwen3/reference_*.json`），逐项校验：ChatML 模板字符串、BPE 分词 ids、逐层 hidden states、最后位置 logits + top-5 token、greedy 生成 32 token 与 HF 完全一致。详见 [Weight-Loading](Weight-Loading.md) 与 [Transformer](Transformer.md)。

## SIMD 加速

热点算子（decode 阶段的逐行点积）由 [Matmul](../src/main/java/io/leavesfly/minivllm/math/Matmul.java) 驱动，它在类初始化时探测 Vector API：

```java
static {
    DotKernel k;
    try {
        Class.forName("jdk.incubator.vector.FloatVector");
        k = new VectorDotKernel();   // SIMD 分块 fma 累加
    } catch (Throwable t) {
        k = new ScalarDotKernel();   // 朴素循环，HotSpot 自动向量化
    }
    KERNEL = k;
}
```

- 编译/运行都带 `--add-modules jdk.incubator.vector` 时，用 `VectorDotKernel`（按 `SPECIES_PREFERRED` 的 lane 宽度分块，fma 累加）
- 缺失该模块时自动回退 `ScalarDotKernel`，功能完全不变，仅速度略降
- 同时 `Matmul.parallelRows` 会把输出行分块到固定线程池（按 CPU 核数），行间无依赖，可获接近核数的加速比

## 常见问题

- **首次启动卡在下载？** 权重约 1.5 GB，`ModelDownloader` 默认 ModelScope 优先。可用 `--mirror hf` 或提前手动放到 `./models/Qwen3-0.6B/`。下载支持 `.part` 断点续传，中断后重启继续。
- **OutOfMemoryError？** Qwen3-0.6B 的 BF16 权重转 F32 后约 2.4 GB，加 KV 池，建议 `-Xmx6g`。或降低 `--max-seqs` / `--max-seq-len`。
- **想看每步调度？** 默认打开引擎日志（`[engine] running=.. waiting=.. freeBlocks=..`），加 `--quiet` 关闭。
- **随机模式输出乱码？** 正常——随机权重无语义，仅用于验证流程。要真实对话请用默认 Qwen3 模式。

下一步：读 [Architecture](Architecture.md) 建立全局认知。
