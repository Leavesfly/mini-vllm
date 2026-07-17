# Weight Loading

本文讲清楚 mini-vllm 如何**零依赖**地把 HuggingFace 的 Qwen3-0.6B 权重加载进内存并组装成可推理的模型：safetensors 二进制解析、BF16→F32 转换、权重命名映射、参数量校验，以及模型自动下载。

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
