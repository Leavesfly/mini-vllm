# Transformer

本文讲清楚 mini-vllm 的 CPU 前向计算，重点是 **Qwen3 架构的五大特性**：RMSNorm、RoPE、GQA、SwiGLU、QK-Norm。所有实现都是纯 Java 零依赖，且与 HuggingFace Qwen3-0.6B 逐层数值对齐。

## 两套模型实现

mini-vllm 有两套平行的 `LlmModel` 实现，共享同一套 math 算子与 PagedAttention KV cache：

| 模型 | 归一化 | 位置编码 | 注意力 | FFN | 用途 |
|---|---|---|---|---|---|
| [Qwen3Model](../src/main/java/io/leavesfly/minivllm/model/Qwen3Model.java) | RMSNorm | RoPE | GQA + QK-Norm | SwiGLU | 真实对话 |
| [TransformerModel](../src/main/java/io/leavesfly/minivllm/model/TransformerModel.java) | LayerNorm | 学习式 wpe | MHA（GPT-3 含稀疏） | GELU-MLP | 学习/验证 |

本文以 Qwen3 为主线。

## Qwen3-0.6B 配置

来自 `config.json`（[ModelConfig.fromConfigJson](../src/main/java/io/leavesfly/minivllm/model/ModelConfig.java)）：

| 字段 | 值 | 说明 |
|---|---|---|
| `hidden_size` (dModel) | 1024 | 隐藏层维度 |
| `num_hidden_layers` (nLayer) | 28 | Transformer 层数 |
| `num_attention_heads` (nHead) | 16 | Q 头数 → qDim = 16×128 = 2048 |
| `num_key_value_heads` (nKVHead) | 8 | KV 头数（GQA）→ kvDim = 8×128 = 1024 |
| `head_dim` | 128 | 显式头维度（注意 ≠ dModel/nHead） |
| `intermediate_size` (dFfn) | 3072 | SwiGLU 中间维度（约 3×dModel） |
| `vocab_size` | 151936 | 词表大小 |
| `rope_theta` | 1000000 | RoPE 基频 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | true | lm_head 复用 embed_tokens |
| `eos_token_id` | 151645 | 停止 token |

约 **596M 参数**（`Qwen3Loader.expectedParams` 精确计算并在加载时校验）。注意 `qDim ≠ dModel`：headDim 是独立超参，这是 Qwen3 与 GPT-2 的一个显著区别。

## 整体前向流程

[Qwen3Model](../src/main/java/io/leavesfly/minivllm/model/Qwen3Model.java)：

```
x = embed_tokens(tokens)          // 仅词嵌入，无位置嵌入（位置由 RoPE 提供）
for block in 28 blocks:
    x = block(x)                  // Qwen3Block：GQA + QK-Norm + RoPE + SwiGLU
x = RmsNorm(x)                    // model.norm
logits = x · embed_tokensᵀ        // tied lm_head
```

与 GPT-2 的关键差异：**无 wpe（学习式位置嵌入）**，位置信息完全由注意力内部的 RoPE 提供；嵌入层只做查表。

## Qwen3Block：pre-norm 残差结构

[Qwen3Block](../src/main/java/io/leavesfly/minivllm/model/Qwen3Block.java) 是标准的两段 pre-norm 残差：

```java
// x = x + Attention(RmsNorm(x))
float[] h = x.clone();
ln1.forwardRowsInPlace(h, seqLen);           // input_layernorm
float[] a = attn.prefill(h, seqLen, kvMgr, bt, startIdx);
for (int i = 0; i < x.length; i++) x[i] += a[i];

// x = x + SwiGLU(RmsNorm(x))
float[] h2 = x.clone();
ln2.forwardRowsInPlace(h2, seqLen);          // post_attention_layernorm
float[] f = ffn.forwardBatch(h2, seqLen);
for (int i = 0; i < x.length; i++) x[i] += f[i];
```

三种入口：`prefill`（整段，写 KV cache）、`decode`（单 token，PagedAttention）、`forward`（无状态稠密，供对拍）。结构完全一致，只是 attention 子层调用不同方法。

---

## 特性一：RMSNorm

[RmsNorm](../src/main/java/io/leavesfly/minivllm/math/RmsNorm.java) —— LLaMA/Qwen 系列的标准归一化。与 LayerNorm 相比**不减均值、无 bias**，只做缩放：

```
y = x / sqrt(mean(x²) + eps) * gamma
```

```java
public void forwardInPlace(float[] x, int offset) {
    float ss = 0f;
    for (int i = 0; i < dim; i++) { float v = x[offset+i]; ss += v*v; }
    float inv = 1f / (float) Math.sqrt(ss / dim + eps);
    for (int i = 0; i < dim; i++) x[offset+i] = x[offset+i] * inv * weight[i];
}
```

带 `offset` 的版本是为 **QK-Norm** 服务的：Q/K 投影后按头切分，逐头（dim=headDim=128）归一化。

## 特性二：RoPE（旋转位置编码）

[RotaryEmbedding](../src/main/java/io/leavesfly/minivllm/model/RotaryEmbedding.java) —— 把位置信息以"旋转"方式注入 Q/K。核心思想：把 headDim 维向量按**前半/后半配对**（half-split，GPT-NeoX 风格，Qwen 的实际实现），每对 `(x[i], x[i+half])` 在二维平面旋转角度 `pos * invFreq[i]`：

```java
public void applyInPlace(float[] x, int offset, int pos) {
    int row = pos * halfDim;
    for (int i = 0; i < halfDim; i++) {
        float c = cos[row+i], s = sin[row+i];
        float x1 = x[offset+i], x2 = x[offset+i+halfDim];
        x[offset+i]         = x1*c - x2*s;
        x[offset+i+halfDim] = x2*c + x1*s;
    }
}
```

其中 `invFreq[i] = theta^(-2i/headDim)`，Qwen3 的 theta=1000000。cos/sin 表在构造时按 `[maxSeqLen, halfDim]` 预计算，推理时查表。

**三个易踩的坑**（源码注释已标注）：
1. **half-split 而非交错式**：Qwen 用前半/后半配对，不是 GPT-J 的相邻交错（interleaved）。这是与 HF 对齐时最常见的错误来源。
2. **只作用 Q/K，不作用 V**：RoPE 保证 Q·K 点积只依赖相对位置 (m−n)，V 不需要旋转。
3. **RoPE 表与层无关**：同 headDim、同 theta，全模型 28 层共享一个 `RotaryEmbedding` 实例。

`RotaryEmbeddingTest` 验证 cos/sin 值与旋转正确性。

## 特性三：GQA（分组查询注意力）

[Qwen3Attention](../src/main/java/io/leavesfly/minivllm/model/Qwen3Attention.java) —— nHead 个 Q 头共享 nKVHead 个 KV 头，`group = nHead / nKVHead`（Qwen3-0.6B 为 16/8 = 2）。Q 头 h 读取 KV 头 `h / group`：

```java
for (int h = 0; h < nHead; h++) {
    int qOff = h * headDim;
    int kvOff = (h / group) * headDim;   // GQA 映射：Q 头 → KV 头
    // ... 用 q[qOff..] 与 k[kvOff..] 做点积
}
```

收益：KV cache 每 token 只存 `kvDim = nKVHead*headDim`（1024）而非 `qDim`（2048），**显存与带宽都省 group 倍**。这也直接影响 KV cache 的存储维度（见 [PagedAttention](PagedAttention.md)）。

维度关系：`dModel=1024 → q_proj 输出 qDim=2048（≠dModel）；k/v_proj 输出 kvDim=1024；o_proj 把 qDim=2048 投回 dModel=1024`。

## 特性四：QK-Norm

Qwen3 特有：Q/K 投影后**先按头做 RMSNorm，再施加 RoPE**（顺序不能反）。稳定注意力 logits：

```java
private void applyQkNormAndRope(float[] q, int qBase, float[] k, int kBase, int pos) {
    for (int h  = 0; h  < nHead;   h++)  qNorm.forwardInPlace(q, qBase + h*headDim);   // 1. Q 逐头 RMSNorm
    for (int kh = 0; kh < nKVHead; kh++) kNorm.forwardInPlace(k, kBase + kh*headDim);  //    K 逐头 RMSNorm
    for (int h  = 0; h  < nHead;   h++)  rope.applyInPlace(q, qBase + h*headDim, pos); // 2. Q RoPE
    for (int kh = 0; kh < nKVHead; kh++) rope.applyInPlace(k, kBase + kh*headDim, pos);//    K RoPE
}
```

- `qNorm` / `kNorm` 权重形状均为 `[headDim]`（128），层内各头共享。
- 顺序是 **Norm → RoPE**，反了会与 HF 数值不一致。

## 特性五：SwiGLU FFN

[SwiGluFfn](../src/main/java/io/leavesfly/minivllm/model/SwiGluFfn.java) —— 结构为 `down( silu(gate(x)) * up(x) )`，三个无 bias 线性层：

```java
public float[] forward(float[] x) {
    float[] g = gate.forward(x);   // dModel → dFfn
    float[] u = up.forward(x);     // dModel → dFfn
    for (int i = 0; i < g.length; i++) g[i] = Silu.silu(g[i]) * u[i];  // 门控
    return down.forward(g);        // dFfn → dModel
}
```

- 激活 [Silu](../src/main/java/io/leavesfly/minivllm/math/Silu.java)：`silu(x) = x * sigmoid(x) = x / (1 + e^-x)`，也叫 swish。
- 相比 GPT-2 的 `fc1→GELU→fc2`，SwiGLU 多一路 `gate` 门控投影，同参数量下表达更强。
- Qwen3-0.6B 的 dFfn=3072 约为 3×dModel（而非 GPT-2 的 4×）。

---

## 基础算子

### Linear：无 bias 的 y = x·Wᵀ

[Linear](../src/main/java/io/leavesfly/minivllm/model/Linear.java) 权重布局与 PyTorch `nn.Linear` 一致：`W` 形状 `[outFeatures, inFeatures]` 行优先，`y[i] = Σ_p x[p]·W[i*in+p]`。Qwen3 全部 Linear 无 bias（`attention_bias=false`），用 `Linear.of` 构造。计算退化为 `Matmul.matVec` 的逐行点积。

### Embedding：查表 + tied lm_head

[Embedding](../src/main/java/io/leavesfly/minivllm/model/Embedding.java)：`lookup` 查表得 token 向量；`projectToVocab` 反向用隐状态与权重表点积得 logits。tied 模式下 lm_head 复用 embed_tokens，省一半参数（`logits[i] = hidden · weight[i]`）。

### Matmul：SIMD + 行并行

[Matmul](../src/main/java/io/leavesfly/minivllm/math/Matmul.java) 是最热的算子。`matVec`（decode 热点）把权重每行与激活向量做点积，按输出行分块并行 + SIMD 点积：

```java
public static float[] matVec(float[] w, float[] x, int m, int k) {
    float[] y = new float[m];
    parallelRows(m, i -> y[i] = KERNEL.dot(w, i*k, x, 0, k)); // 行并行 + DotKernel
    return y;
}
```

`KERNEL` 由 [DotKernel](../src/main/java/io/leavesfly/minivllm/math/DotKernel.java) 运行时探测：有 Vector API 用 `VectorDotKernel`（fma 分块累加），否则 `ScalarDotKernel`。详见 [Getting-Started](Getting-Started.md) 的 SIMD 章节。

### Softmax：数值稳定

[Softmax](../src/main/java/io/leavesfly/minivllm/math/Softmax.java) 先减最大值再 exp，避免溢出。attention 沿 key 维、采样沿词表维。

## causal attention 核心

prefill 阶段 K/V 连续在手，直接做因果注意力（第 i 个 query 只看 j≤i）：

```java
private float[] causalAttention(float[] q, float[] k, float[] v, int seqLen) {
    float invSqrt = 1f / (float) Math.sqrt(headDim);
    for (int h = 0; h < nHead; h++) {
        int qOff = h*headDim, kvOff = (h/group)*headDim;   // GQA
        for (int i = 0; i < seqLen; i++) {
            float[] scores = new float[i+1];               // 只算 j≤i（causal）
            for (int j = 0; j <= i; j++) scores[j] = dot(q_i, k_j) * invSqrt;
            Softmax.softmaxInPlace(scores);
            for (int j = 0; j <= i; j++) out_i += scores[j] * v_j;
        }
    }
}
```

decode 阶段则改为逐 block 读取散落的 KV cache（PagedAttention，见 [PagedAttention](PagedAttention.md)）。两条路径在 dense 层数值一致。

## 数值对齐：与 HuggingFace 逐层比对

Qwen3 实现的正确性由 [Qwen3AlignmentTest](../src/test/java/io/leavesfly/minivllm/model/Qwen3AlignmentTest.java) 严格保证。它加载真实权重，用 `tools/dump_reference.py` 从 HF transformers（F32 eager 前向）导出的参考数据逐项比对：

| 对齐项 | 判据 |
|---|---|
| ChatML 模板字符串 | 与 HF `apply_chat_template` 完全一致 |
| BPE 分词 ids | 完全一致 |
| 第 0/1 层 block 输出 hidden | maxAbs < 2e-2 |
| final norm 后 hidden | maxAbs < 5e-3 |
| 最后位置 logits | maxAbs < 0.2，且 top-5 token 一致 |
| **greedy 生成 32 token** | 与 HF **完全一致**（终极标准） |

其中 greedy 生成走的是引擎的 `prefillLogits` + 逐 token `decodeLogits`（即 PagedAttention 路径），所以这个测试同时验证了模型前向、RoPE/GQA/QK-Norm、以及 KV cache 分页读写的全链路正确性。

运行（需真实权重）：
```bash
mvn test -Dtest=Qwen3AlignmentTest -Dqwen3.align=true -DargLine="-Xmx6g"
```

`Qwen3ModelTest`、`RmsNormTest`、`RotaryEmbeddingTest`、`SoftmaxTest`、`MatmulTest` 等则做单元级验证，无需真实权重即可跑。

## GPT-2/GPT-3 路径（对照）

[TransformerModel](../src/main/java/io/leavesfly/minivllm/model/TransformerModel.java) 提供 GPT 风格前向：`x = wte(tokens) + wpe(positions)`，用 LayerNorm、MHA、GELU-MLP。GPT-3 相对 GPT-2 的唯一架构差异是**交替稠密/局部带状稀疏注意力**（`useSparseAttention`，奇数层退化为只看最近 sparseWindow 个 token）。这条路径用于无权重的快速验证与教学对照。

## 延伸

- attention 如何读写 KV cache：[PagedAttention](PagedAttention.md)
- prefill/decode 何时被调用：[Continuous-Batching](Continuous-Batching.md)
- 权重如何从 safetensors 映射到这些算子：[Weight-Loading](Weight-Loading.md)
