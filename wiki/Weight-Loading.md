# Weight Loading

本文讲清楚 mini-vllm 如何**零依赖**地把 HuggingFace 的 Qwen3-0.6B 权重加载进内存并组装成可推理的模型：safetensors 二进制解析、BF16→F32 转换、mmap 按需调页、权重命名映射、参数量校验，以及模型自动下载。

## 为什么选 safetensors

PyTorch 的 `.pt`/`.bin` 是 Python pickle 格式，纯 Java 几乎无法安全解析。而 **safetensors** 用"明文 JSON header + 裸张量数据"，格式极简，非常适合零依赖 Java 解析——这正是本项目选它作为权重载体的原因。

文件格式（[SafetensorsLoader](../src/main/java/io/leavesfly/minivllm/weights/SafetensorsLoader.java) 顶部注释）：

```
[8 字节 little-endian uint64]  header JSON 的字节数 N
[N 字节]                       header JSON：{tensor_name: {dtype, shape, data_offsets}, ...}
[剩余]                         原始张量数据，按 data_offsets 定位
```

## safetensors 解析

### 读 header

```java
// 1. 读 8 字节 header 长度
ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
readFully(ch, lenBuf, 0);
long headerLen = lenBuf.flip().getLong();

// 2. 读 header JSON 并用 SimpleJson 解析
ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
readFully(ch, headerBuf, 8);
Map<String, Object> header = SimpleJson.parseObject(
        new String(headerBuf.array(), StandardCharsets.UTF_8));

// 3. 数据区起始 = 8 + headerLen；每个 tensor 按 data_offsets 定位
long dataBase = 8 + headerLen;
```

### 按 tensor 分段读取

关键设计：用 `FileChannel` **按 tensor 分段读取**，而非 `readAllBytes` 一次性载入。原因：
1. 规避 Java `byte[]` 的 2GB 上限（Qwen3-0.6B 权重约 1.2GB，展开后更大）。
2. 把内存峰值从"文件两倍"降到"单个 tensor 两倍"。

每个 tensor 再按 4MB 块流式读取并即时转 F32：

```java
ByteBuffer chunk = ByteBuffer.allocate((int) Math.min(byteLen, 1 << 22)); // 4MB 块
while (remain > 0) {
    int want = (int) Math.min(remain, chunk.capacity());
    readFully(ch, chunk.clear().limit(want), pos);
    chunk.flip();
    switch (dtype) {
        case "F32":  chunk.order(LITTLE_ENDIAN).asFloatBuffer().get(data, idx, elems); break;
        case "BF16": readBf16Into(chunk, data, idx, elems); break;
        default:     readF16Into(chunk, data, idx, elems);  break; // F16
    }
    idx += elems; pos += want; remain -= want;
}
```

支持三种 dtype：`F32`（4 字节直读）、`BF16` / `F16`（2 字节，转 F32）。

## BF16 / F16 → F32 转换

[Bf16](../src/main/java/io/leavesfly/minivllm/weights/Bf16.java) 处理半精度到单精度的位运算。

### BF16：一次左移

bfloat16 = 1 符号位 + **8 指数位**（与 F32 相同）+ 7 尾数位。它只是把 F32 的尾数截断到 7 位，所以转 F32 只需**左移 16 位**：

```java
public static float bf16ToFloat(int bits) {
    return Float.intBitsToFloat(bits << 16);
}
```

这也是 BF16 成为 LLM 权重主流格式的原因——它保留了 F32 的指数范围（动态范围大），只损失尾数精度。HuggingFace 发布的 Qwen3 权重就是 BF16。

### F16：需按位重建

IEEE half = 1 符号位 + **5 指数位** + 10 尾数位，指数偏置和位宽都与 F32 不同，需分情况按位重建（正规数/次正规数/inf/NaN）：

```java
public static float f16ToFloat(int h) {
    int sign = (h>>15)&1, exp = (h>>10)&0x1F, frac = h&0x3FF;
    if (exp == 0)       return 次正规数按算术构造;                       // exp=0
    else if (exp == 31) f32Bits = (sign<<31)|0x7F800000|(frac<<13);    // inf/NaN
    else                f32Bits = (sign<<31)|((exp-15+127)<<23)|(frac<<13); // 正规数：偏置 15→127
    return Float.intBitsToFloat(f32Bits);
}
```

## mmap 按需调页（--precision mmap）

前文的加载路径都把权重**物化进 JVM 堆**（f32/bf16 数组或量化块），堆需求与模型体积成正比。`--precision mmap` 走另一条路：权重留在磁盘，只把文件**映射**进进程地址空间，真正读数据由 OS 按需调页完成——堆里没有任何权重副本，**物理内存小于模型体积也能服务**（实测 `-Xmx512m` 跑 1.5GB 的 Qwen3-0.6B）。

### 映射原理

[MmapWeights.open](../src/main/java/io/leavesfly/minivllm/weights/MmapWeights.java) 对每个 safetensors 分片调用一次 `FileChannel.map(READ_ONLY, 0, size)`：

```java
MappedByteBuffer mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size());
// 只建页表，不读任何数据 —— Qwen3-0.6B 三个分片 1.5GB 映射耗时 0.0s
```

之后每次访问映射区触发缺页中断，OS 从磁盘调入 4KB 页进**页缓存**；内存紧张时页缓存自行回收冷页。权重常驻量由 OS 而非 JVM 管理，这正是 llama.cpp / vLLM 加载大模型的方式。

生命周期零成本：`Linear`/`Embedding` 持有的是 `ShortBuffer` 视图，间接强引用 `MappedByteBuffer`，无需 close；unmap 依赖 GC Cleaner，学习项目足够。

### 约束：仅磁盘 BF16

零拷贝的前提是磁盘格式就是算子要的格式。bf16 张量直接 `asShortBuffer()` 得到视图（无拷贝无转换），与堆内 `Linear.ofBf16` 用同一份 `short` 位表示；F32/F16 需要加宽/转换，失去零拷贝意义，直接抛带指引的 IOException。另有两个格式细节：

- safetensors data 区相对文件头按 8 字节对齐（HF 官方 writer 用空格补齐 header），满足 `short` 的 2 字节对齐要求；不对齐的文件会报错提示可能损坏。
- `MappedByteBuffer` 用 int 寻址，单分片 ≤ 2GB；超大模型按 HF 惯例切分片即可（复用 [SafetensorsLoader](../src/main/java/io/leavesfly/minivllm/weights/SafetensorsLoader.java) 的分片发现逻辑）。

### 行拷贝保 SIMD：核心取舍

堆内路径的 SIMD 内核（`DotKernel` 基于 Vector API）作用在 `short[]`/`float[]` 数组上，无法直接吃 `ShortBuffer`。两条路：

1. **逐元素读**：`ShortBuffer.get(i)` 每元素一次边界检查 + 方法调用，丢掉 SIMD，慢数倍；
2. **行拷贝**（本项目采用）：每行权重 bulk 读进线程私有行缓冲，再走**原封不动**的 `KERNEL.dotBf16`：

```java
// Matmul：ThreadLocal 行缓冲只增不缩，并发线程各自持有
public static float dotBf16Mapped(ShortBuffer w, int wOff, float[] x, int xOff, int len) {
    short[] row = mmapRowBuffer(len);
    w.get(wOff, row, 0, len);          // 绝对位置 bulk 读，不移动 position，线程安全
    return KERNEL.dotBf16(row, 0, x, xOff, len);  // SIMD 内核与堆内路径逐位同一份
}
```

行拷贝的代价是每行权重多一次内存复制（decode 每步全量权重过一遍），换来的是与堆内 bf16 **bitwise 一致**的输出和完整的 Vector API 加速——同一行数据、同一个点积内核，数值不可能分叉。

与 llama.cpp 对照：llama.cpp 用 C++ 直接在 mmap 内存上做 SIMD（无对齐/数组限制，无行拷贝），mini-vllm 受限于 Vector API 只接受 Java 数组才引入行缓冲。两者共同的本质是**页缓存承载权重、堆/进程内存只放激活与 KV**。

### 算子与装配

- [Linear](../src/main/java/io/leavesfly/minivllm/model/Linear.java) / [Embedding](../src/main/java/io/leavesfly/minivllm/model/Embedding.java) 新增 mmap 模式（`ofMmapBf16` 工厂 + `isMmapBf16()` 判定），forward/forwardBatch/lookup/projectToVocab 全路径覆盖——lm_head 是全模型最大投影，必须走 mapped 内核；
- [Qwen3Attention](../src/main/java/io/leavesfly/minivllm/model/Qwen3Attention.java) 的 fusedQKV 融合需要拼接三份权重（会物化回堆），mmap 模式跳过融合、decode 回退三路独立投影，数值逐行等价；
- [Qwen3Loader.loadMmap](../src/main/java/io/leavesfly/minivllm/weights/Qwen3Loader.java) 经 `MmapWeightSource` 装配，复用与堆内路径相同的 `buildModel` + 三重校验；RMSNorm gamma 是小数组，一次性 bulk 读入转 f32 常驻堆；
- [ModelFamily](../src/main/java/io/leavesfly/minivllm/family/ModelFamily.java) 对 MMAP 跳过 `ensureHeapFor` 堆预检（权重不占堆，预检无意义）。

### 实测（Qwen3-0.6B，Apple Silicon）

| 指标 | bf16 堆内 | mmap |
|---|---|---|
| 堆需求 | ~3GB | 数百 MB（行缓冲 + KV cache） |
| 加载耗时 | 秒级（读盘 + 转换） | 0.0s（仅映射） |
| 生成速度（页缓存热后） | ~35 tok/s | ~23 tok/s |

冷页首次访问要承担缺页读盘，顺序扫描一遍权重后趋于稳定。

## 权重命名映射：从 HF 张量到 Qwen3Model

[Qwen3Loader](../src/main/java/io/leavesfly/minivllm/weights/Qwen3Loader.java) 把 `Map<String, float[]>`（张量名→数据）组装成 [Qwen3Model](../src/main/java/io/leavesfly/minivllm/model/Qwen3Model.java)。

### HF Qwen3ForCausalLM 权重命名（共 338 个张量）

| 张量名 | 形状 | 装配为 |
|---|---|---|
| `model.embed_tokens.weight` | [vocab, dModel] | `Embedding`（tied 时兼 lm_head） |
| `model.layers.{i}.input_layernorm.weight` | [dModel] | ln1 (RmsNorm) |
| `model.layers.{i}.self_attn.q_proj.weight` | [qDim, dModel] | q_proj (Linear, 无 bias) |
| `model.layers.{i}.self_attn.k_proj.weight` | [kvDim, dModel] | k_proj |
| `model.layers.{i}.self_attn.v_proj.weight` | [kvDim, dModel] | v_proj |
| `model.layers.{i}.self_attn.o_proj.weight` | [dModel, qDim] | o_proj |
| `model.layers.{i}.self_attn.q_norm.weight` | [headDim] | qNorm (QK-Norm) |
| `model.layers.{i}.self_attn.k_norm.weight` | [headDim] | kNorm (QK-Norm) |
| `model.layers.{i}.post_attention_layernorm.weight` | [dModel] | ln2 (RmsNorm) |
| `model.layers.{i}.mlp.gate_proj.weight` | [dFfn, dModel] | SwiGLU gate |
| `model.layers.{i}.mlp.up_proj.weight` | [dFfn, dModel] | SwiGLU up |
| `model.layers.{i}.mlp.down_proj.weight` | [dModel, dFfn] | SwiGLU down |
| `model.norm.weight` | [dModel] | 最终 RmsNorm |

组装代码（每层循环）：

```java
Linear q = linear(weights, consumed, p+"self_attn.q_proj.weight", cfg.dModel, cfg.qDim());
Linear k = linear(weights, consumed, p+"self_attn.k_proj.weight", cfg.dModel, cfg.kvDim());
Linear v = linear(weights, consumed, p+"self_attn.v_proj.weight", cfg.dModel, cfg.kvDim());
Linear o = linear(weights, consumed, p+"self_attn.o_proj.weight", cfg.qDim(), cfg.dModel);
RmsNorm qNorm = rms(weights, consumed, p+"self_attn.q_norm.weight", cfg.headDim(), cfg.rmsNormEps);
RmsNorm kNorm = rms(weights, consumed, p+"self_attn.k_norm.weight", cfg.headDim(), cfg.rmsNormEps);
Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
```

三个要点：
1. **全部 Linear 无 bias**（`attention_bias=false`），用 `Linear.of` 构造。
2. **tied embeddings**：`tie_word_embeddings=true` 时没有 `lm_head.weight`，logits 投影复用 `embed_tokens`。
3. **RoPE 表全模型共享**：cos/sin 与层无关（同 headDim、同 theta），只建一个 `RotaryEmbedding` 实例传给每层。

### 加载即校验

`Qwen3Loader.load` 在加载时做三重校验，尽早暴露不匹配：

```java
// 1. 每个权重的长度必须等于期望的 out×in
private static float[] get(..., String name, int expect) {
    float[] d = w.get(name);
    if (d == null) throw ...("缺少权重: " + name);
    if (d.length != expect) throw ...("权重长度不符");
    consumed.add(name);
    return d;
}

// 2. 所有张量必须被消费（tied 下容忍多余的 lm_head.weight 拷贝）
Set<String> extra = new HashSet<>(weights.keySet());
extra.removeAll(consumed);
if (cfg.tieWordEmbeddings) extra.remove("lm_head.weight");
if (!extra.isEmpty()) throw ...("存在未消费的权重张量: " + extra);

// 3. 参数量必须与按配置计算的期望值一致
if (model.numParameters() != expectedParams(cfg)) throw ...("参数量不符");
```

`expectedParams` 精确计算参数量（tied 下 lm_head 不重复计）：

```
perLayer = 2·dModel (ln1+ln2) + qDim·dModel (q) + 2·kvDim·dModel (k,v)
         + dModel·qDim (o) + 2·headDim (q/k norm) + 3·dFfn·dModel (gate/up/down)
total    = vocab·dModel (embed) + perLayer·nLayer + dModel (final norm)
```

Qwen3-0.6B 代入得约 **596M**。

## 随机初始化（无权重跑通流程）

`Qwen3Loader.randomInit` 用 Box-Muller 正态分布（std=0.02）填充权重、RMSNorm gamma 置 1，输出无语义但结构完整，用于验证前向链路：

```java
Linear q = Linear.of(randN(rnd, cfg.qDim()*cfg.dModel, 0.02f), cfg.dModel, cfg.qDim());
RmsNorm qNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps); // gamma 全 1
```

GPT 路径的对应加载器是 [ModelLoader](../src/main/java/io/leavesfly/minivllm/weights/ModelLoader.java)。

## 模型自动下载

[ModelDownloader](../src/main/java/io/leavesfly/minivllm/weights/ModelDownloader.java) 用纯 JDK `HttpClient` 实现零依赖下载。

### 四级解析顺序（命中即返回）

```java
public Path resolve() {
    if (isComplete(projectLocalDir())) return 项目内 ./models/<name>;   // 最优先
    Path hf = findInHfCache();  if (hf != null) return hf;              // HF 缓存
    if (isComplete(localCacheDir())) return ~/.cache/mini-vllm/models;  // 本地缓存
    downloadAll(local); return local;                                  // 在线下载
}
```

- 项目内目录：`./models/Qwen3-0.6B`
- HF 缓存：`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev>`（多 revision 取最近修改）
- 本地缓存：`~/.cache/mini-vllm/models/Qwen-Qwen3-0.6B`

"完整"的判据：`config.json`、`model.safetensors`、`vocab.json`、`merges.txt`、`tokenizer_config.json` 五个必需文件齐全且非空。

### 镜像策略

`auto`（默认）ModelScope 优先、HuggingFace 兜底；也可用 `--mirror hf`/`modelscope` 或环境变量 `MINIVLLM_MIRROR`：

```java
default -> { urls.add(ModelScope URL); urls.add(HuggingFace URL); } // auto
```

### 下载可靠性

- **`.part` 临时文件 + 原子改名**：下载写入 `xxx.part`，完成后 `Files.move(ATOMIC_MOVE)`，避免半成品被当作完整文件。
- **断点续传**：`.part` 已有字节时带 `Range: bytes=<existing>-` 请求头，1.5GB 权重中断后不必重下。
- **来源记录（sidecar）**：`.part.url` 记录来源镜像。不同镜像的同名文件版本可能不一致，跨来源拼接会得到损坏文件，所以镜像变更时放弃旧进度重下。
- **HTTP 302**：HF 的 resolve URL 会跳转到 CDN，`HttpClient` 开启 `followRedirects(NORMAL)`。
- **不设请求级 timeout**：`HttpClient` 的请求超时计时器在响应体流式读取期间仍生效，大文件必被误杀；连接超时由 client 的 `connectTimeout` 控制。
- 状态码处理：200（全新下载）、206（续传）、416（服务端拒绝区间→`.part` 多半已完整，直接采用）。

## 分词器加载

Qwen3 模式下，[BpeTokenizer.fromModelDir](../src/main/java/io/leavesfly/minivllm/tokenizer/BpeTokenizer.java) 从模型目录加载 byte-level BPE：
- `vocab.json`：token→id（151936 项），用专用扁平解析器（避免 15 万个 Double 装箱，快一个数量级）。
- `merges.txt`：BPE 合并规则（整行 "A B" 作为 rank key）。
- `tokenizer_config.json` 的 `added_tokens_decoder`：26 个 special token（如 `<|im_start|>`、`<|im_end|>`），不在 vocab.json 中。

详见 [OpenAI-API](OpenAI-API.md) 的分词与 ChatML 章节。

## 测试验证

- **与 HF 逐层对齐**：[Qwen3AlignmentTest](../src/test/java/io/leavesfly/minivllm/model/Qwen3AlignmentTest.java) 加载真实 safetensors 权重后，验证 hidden/logits/greedy 生成与 HF 完全一致（见 [Transformer](Transformer.md)）。这间接验证了 safetensors 解析、BF16 转换、权重映射全部正确。
- **参考数据生成**：`tools/dump_reference.py`（HF transformers F32 eager 前向）导出 `src/test/resources/qwen3/reference_*.json`；`tools/dump_tokenizer_refs.py` 导出分词参考。
- 运行：`mvn test -Dtest=Qwen3AlignmentTest -Dqwen3.align=true -DargLine="-Xmx6g"`。

## 延伸

- 加载后的权重如何参与计算：[Transformer](Transformer.md)
- 分词器与 ChatML：[OpenAI-API](OpenAI-API.md)
- 首次启动的下载流程：[Getting-Started](Getting-Started.md)
- mmap 的数值一致性测试：[MmapWeightsTest](../src/test/java/io/leavesfly/minivllm/weights/MmapWeightsTest.java)（mmap vs 堆内 bf16 逐位对齐）
