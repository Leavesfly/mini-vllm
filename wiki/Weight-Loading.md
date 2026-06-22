# Weight Loading

如何用纯 Java 加载深度学习权重并组装成 Transformer。聚焦 [`weights`](../src/main/java/io/leavesfly/minivllm/weights) 包。

## 1. 为什么选 safetensors

加载权重需要一种文件格式。常见选择：

| 格式 | 零依赖可行性 |
|---|---|
| PyTorch `.pt/.pth` | ❌ 基于 pickle，零依赖下几乎无法解析 |
| ONNX | ❌ protobuf，解析复杂 |
| **safetensors** | ✅ 明文 JSON header + 裸二进制，极简 |

safetensors 的设计非常适合纯 Java 解析：header 是标准 JSON，数据是连续的原始字节。这正是本项目选它的原因。

## 2. safetensors 文件格式

```
┌──────────────────────────────────────────┐
│ 8 字节  little-endian uint64              │  header JSON 的字节数 N
├──────────────────────────────────────────┤
│ N 字节  header JSON                        │  {tensor_name: {dtype, shape, data_offsets}}
├──────────────────────────────────────────┤
│ 剩余    原始张量数据                        │  按 data_offsets 定位
└──────────────────────────────────────────┘
```

header JSON 示例：

```json
{
  "wte.weight": {"dtype": "F32", "shape": [256, 64], "data_offsets": [0, 65536]},
  "h.0.attn.q_proj.weight": {"dtype": "F32", "shape": [64, 64], "data_offsets": [65536, 67584]},
  "__metadata__": {"format": "pt"}
}
```

每个张量记录：`dtype`（数据类型）、`shape`（形状）、`data_offsets`（在数据区的起止字节偏移）。

## 3. SafetensorsLoader 解析

[`SafetensorsLoader.load`](../src/main/java/io/leavesfly/minivllm/weights/SafetensorsLoader.java) 三步解析：

```java
public static Map<String, float[]> load(Path path) throws IOException {
    byte[] all = Files.readAllBytes(path);

    // 1. 读 8 字节 header 长度
    long headerLen = ByteBuffer.wrap(all, 0, 8)
            .order(LITTLE_ENDIAN).getLong();

    // 2. 解析 header JSON（复用手写的 SimpleJson）
    String headerJson = new String(all, 8, (int) headerLen, UTF_8);
    Map<String, Object> header = SimpleJson.parseObject(headerJson);

    // 3. 按 data_offsets 从数据区读取每个张量
    int dataBase = 8 + (int) headerLen;
    for (each tensor) {
        long start = offsets[0], end = offsets[1];
        int n = (end - start) / 4;  // F32 每元素 4 字节
        float[] data = new float[n];
        ByteBuffer.wrap(all, dataBase + start, n * 4)
                .order(LITTLE_ENDIAN).asFloatBuffer().get(data);
        tensors.put(name, data);
    }
    return tensors;
}
```

关键点：
- **little-endian**：safetensors 固定小端序，必须 `ByteOrder.LITTLE_ENDIAN`。
- **F32 only**：当前仅支持 float32，学习用微模型足够。
- **零拷贝读取**：`ByteBuffer.asFloatBuffer().get()` 高效填充 float[]。

## 4. 权重命名约定

[`ModelLoader`](../src/main/java/io/leavesfly/minivllm/weights/ModelLoader.java) 按 nanoGPT 风格命名（分离的 q/k/v/o 投影）：

| 张量名 | 形状 | 说明 |
|---|---|---|
| `wte.weight` | [vocab, d] | 词嵌入 |
| `wpe.weight` | [maxSeqLen, d] | 位置嵌入 |
| `h.{i}.ln_1.weight/bias` | [d] | 第 i 层第一个 LayerNorm |
| `h.{i}.attn.q_proj.weight/bias` | [d, d] | Q 投影（行优先 [out,in]） |
| `h.{i}.attn.k_proj.weight/bias` | [d, d] | K 投影 |
| `h.{i}.attn.v_proj.weight/bias` | [d, d] | V 投影 |
| `h.{i}.attn.o_proj.weight/bias` | [d, d] | 输出投影 |
| `h.{i}.mlp.fc1.weight/bias` | [dFfn, d] | FFN 第一层 |
| `h.{i}.mlp.fc2.weight/bias` | [d, dFfn] | FFN 第二层 |
| `ln_f.weight/bias` | [d] | 最终 LayerNorm |

> **注意**：Linear 权重是 `[outFeatures, inFeatures]` 行优先，与 PyTorch `nn.Linear.weight` 一致，无需转置。

## 5. ModelLoader 组装

[`ModelLoader.load`](../src/main/java/io/leavesfly/minivllm/weights/ModelLoader.java) 从权重字典 + ModelConfig 构造 Transformer：

```java
public static Transformer load(ModelConfig cfg, Map<String, float[]> weights) {
    Embedding wte = new Embedding(get(weights, "wte.weight", vocab*d), vocab, d);
    Embedding wpe = new Embedding(get(weights, "wpe.weight", maxSeqLen*d), maxSeqLen, d);

    TransformerBlock[] blocks = new TransformerBlock[nLayer];
    for (int i = 0; i < nLayer; i++) {
        String p = "h." + i + ".";
        LayerNorm ln1 = layerNorm(weights, p+"ln_1", d, eps);
        LayerNorm ln2 = layerNorm(weights, p+"ln_2", d, eps);
        Linear q = linear(weights, p+"attn.q_proj", d, d);
        Linear k = linear(weights, p+"attn.k_proj", d, d);
        Linear v = linear(weights, p+"attn.v_proj", d, d);
        Linear o = linear(weights, p+"attn.o_proj", d, d);
        Attention attn = new Attention(cfg, q, k, v, o);
        Ffn ffn = new Ffn(linear(weights, p+"mlp.fc1", d, dFfn),
                          linear(weights, p+"mlp.fc2", dFfn, d));
        blocks[i] = new TransformerBlock(ln1, ln2, attn, ffn, d);
    }
    LayerNorm lnF = layerNorm(weights, "ln_f", d, eps);
    return new Transformer(cfg, wte, wpe, blocks, lnF);
}
```

每个权重都校验长度（`get` 方法），不匹配即抛异常，避免静默错误。

## 6. 随机初始化兜底

无权重文件时，[`ModelLoader.randomInit`](../src/main/java/io/leavesfly/minivllm/weights/ModelLoader.java) 用 GPT-2 风格正态(std=0.02)初始化，让全流程能立即跑通：

```java
public static Transformer randomInit(ModelConfig cfg) {
    Random rnd = new Random(42);
    // wte/wpe 正态初始化，LayerNorm gamma=1/beta=0，Linear bias=0
    ...
}
```

随机模型输出无意义，但 PagedAttention + batching + API 全链路真实运转——这是验证引擎机制的方式。

## 7. 导出真实微模型权重

若想生成有意义文本，需用 Python 把训练好的小模型导出为符合命名约定的 safetensors。以 nanoGPT 的 char-level Shakespeare 模型为例：

```python
# export_safetensors.py
import torch
from safetensors.torch import save_file

# 加载训练好的 nanoGPT 模型
ckpt = torch.load("ckpt.pt", map_location="cpu")
model = ckpt["model"]  # nanoGPT 的 state_dict

# 拆分合并的 c_attn 为 q/k/v/o（本项目要求分离投影）
def split_attn(state, i):
    prefix = f"blocks.{i}.attn."
    w = state[prefix + "c_attn.weight"]  # [3*d, d]  QKV 合并
    b = state[prefix + "c_attn.bias"]
    d = w.shape[1]
    state[prefix + "q_proj.weight"] = w[:d]
    state[prefix + "k_proj.weight"] = w[d:2*d]
    state[prefix + "v_proj.weight"] = w[2*d:]
    state[prefix + "q_proj.bias"] = b[:d]
    state[prefix + "k_proj.bias"] = b[d:2*d]
    state[prefix + "v_proj.bias"] = b[2*d:]
    state[prefix + "o_proj.weight"] = state.pop(prefix + "c_proj.weight")
    state[prefix + "o_proj.bias"] = state.pop(prefix + "c_proj.bias")
    del state[prefix + "c_attn.weight"], state[prefix + "c_attn.bias"]

# 重命名为本项目约定
def rename(state):
    out = {}
    for k, v in state.items():
        # wte.weight, wpe.weight, ln_f.* 保持
        # blocks.{i}.* -> h.{i}.*
        # blocks.{i}.ln_1 -> h.{i}.ln_1
        # blocks.{i}.mlp.c_fc -> h.{i}.mlp.fc1, blocks.{i}.mlp.c_proj -> h.{i}.mlp.fc2
        ...
    return out

for i in range(model["blocks.0.attn.c_attn.weight"].shape[0] // d):  # nLayer
    split_attn(model, i)

tensors = rename(model)
save_file(tensors, "model.safetensors")
```

导出后启动：

```bash
java -cp target/classes io.leavesfly.minivllm.MiniVllmServer --weights ./model.safetensors --port 8088
```

> 导出时需确保 ModelConfig 与源模型一致（vocab/dModel/nHead/nLayer）。CharTokenizer 需用训练语料的字符表构造。

---

**相关**：[Transformer](Transformer.md) · [Getting Started](Getting-Started.md)
