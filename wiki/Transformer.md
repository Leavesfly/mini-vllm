# Transformer 实现

mini-vllm 手写了一个极简的 GPT-2 风格 Decoder-Only Transformer，跑在 CPU 上。本文走读 [`model`](../src/main/java/io/leavesfly/minivllm/model) 包的实现。

## 1. 模型架构

```
输入 token ids
  → Token Embedding (查表 [vocab, dModel])
  → + Positional Embedding (可学习 [maxSeqLen, dModel])
  → ×N 层 TransformerBlock:
      ├─ LayerNorm → Multi-Head Causal Self-Attention → 残差
      └─ LayerNorm → FFN(Linear→GELU→Linear) → 残差
  → Final LayerNorm
  → LM Head (投影到词表, tied 复用 token embedding)
  → logits → 采样
```

学习用配置（[`ModelConfig.small()`](../src/main/java/io/leavesfly/minivllm/model/ModelConfig.java)）：

| 参数 | 值 | 说明 |
|---|---|---|
| vocabSize | 256 | 字节级词表 |
| dModel | 64 | 隐藏层维度 |
| nHead | 4 | 注意力头数（headDim=16） |
| nLayer | 2 | Transformer 层数 |
| dFfn | 256 | FFN 中间层（4×dModel） |
| blockSize | 16 | PagedAttention 块大小 |

小到 CPU 秒出结果，又能完整体现架构。

## 2. 前向流程：prefill 与 decode

模型有两种前向模式，对应生成的两个阶段：

### Prefill：一次性处理整段 prompt

[`Transformer.prefill`](../src/main/java/io/leavesfly/minivllm/model/Transformer.java)：

```java
public float[] prefill(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
    // 1. 词嵌入 + 位置嵌入
    float[] x = wte.lookupBatch(tokenIds);       // [seqLen, d]
    for (t) x[t] += wpe[startIdx + t];           // 加位置

    // 2. 逐层 Block（每层用各自的 BlockTable）
    for (int i = 0; i < blocks.length; i++)
        x = blocks[i].prefill(x, seqLen, kvMgr, bts[i], startIdx);

    // 3. 最终 LayerNorm
    lnF.forwardRowsInPlace(x, seqLen);
    return x;  // [seqLen, d]，取最后一行算 logits
}
```

prefill 时 K/V 在手边是连续的，用标准 causal attention 高效计算，同时写入物理 block 供后续 decode 复用。

### Decode：逐 token 推进

[`Transformer.decode`](../src/main/java/io/leavesfly/minivllm/model/Transformer.java)：

```java
public float[] decode(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
    float[] x = wte.lookup(tokenId);              // [d] 单 token
    x += wpe[curIdx];                             // 加位置

    for (int i = 0; i < blocks.length; i++)
        x = blocks[i].decode(x, curIdx, kvMgr, bts[i]);  // PagedAttention 读历史 KV

    lnF.forwardInPlace(x);
    return x;  // [d]，算 logits 采样下一个 token
}
```

decode 每步只算 1 个新 token 的 Q，历史 K/V 全在散落的 block 中，经 PagedAttention 读取（详见 [PagedAttention](PagedAttention.md)）。

## 3. TransformerBlock：残差子层

[`TransformerBlock`](../src/main/java/io/leavesfly/minivllm/model/TransformerBlock.java) 采用 GPT-2 的 **pre-LayerNorm** 结构：

```
x = x + Attention(LayerNorm(x))      // 第一个残差
x = x + FFN(LayerNorm(x))            // 第二个残差
```

```java
public float[] prefill(float[] x, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
    // 自注意力残差块
    float[] h = x.clone();
    ln1.forwardRowsInPlace(h, seqLen);
    float[] a = attn.prefill(h, seqLen, kvMgr, bt, startIdx);
    for (i) x[i] += a[i];                      // 残差

    // FFN 残差块
    float[] h2 = x.clone();
    ln2.forwardRowsInPlace(h2, seqLen);
    float[] f = ffn.forwardBatch(h2, seqLen);
    for (i) x[i] += f[i];                      // 残差
    return x;
}
```

pre-LN 的特点：先归一化再进子层，残差路径不经 LN，梯度流更稳，是现代 Transformer 主流。

## 4. Attention：多头自注意力

[`Attention`](../src/main/java/io/leavesfly/minivllm/model/Attention.java) 持有四个投影：`qProj / kProj / vProj / oProj`（均为 dModel→dModel 的 Linear）。

**Prefill 路径**（标准 causal attention）：

```java
float[] q = qProj.forwardBatch(input, seqLen);  // [seqLen, d]
float[] k = kProj.forwardBatch(input, seqLen);
float[] v = vProj.forwardBatch(input, seqLen);

// 写入 KV cache
for (t) kvMgr.writeKV(bt, startIdx + t, k[t], v[t]);

// causal attention：每个 query 只看 0..i 的 key
for (h in heads)
  for (i in seqLen)
    scores[j] = dot(q[i,h], k[j,h]) / sqrt(headDim), j=0..i
    softmax(scores)
    out[i,h] = sum_j scores[j] * v[j,h]

return oProj.forwardBatch(out, seqLen);
```

**Decode 路径**（PagedAttention block-wise 累加）：

```java
float[] q = qProj.forward(hidden);   // [d]
float[] k = kProj.forward(hidden);
float[] v = vProj.forward(hidden);
kvMgr.writeKV(bt, curIdx, k, v);     // 当前 token 入 block

// 逐 block 读历史 K/V 累加（不拷贝）
for (h in heads) {
    for (b in blocks) scores += dot(q[h], blockK[b][h])
    softmax(scores)
    for (b in blocks) out[h] += scores * blockV[b][h]
}
return oProj.forward(out);
```

还提供 `decodeGather` 对照版（先 gather 成连续再标准 attention），两者数学等价，可断言对比。详见 [PagedAttention](PagedAttention.md)。

## 5. FFN：前馈网络

[`Ffn`](../src/main/java/io/leavesfly/minivllm/model/Ffn.java) 结构简单：

```
fc1(dModel→dFfn) → GELU → fc2(dFfn→dModel)
```

```java
public float[] forwardBatch(float[] x, int seqLen) {
    float[] h = fc1.forwardBatch(x, seqLen);  // [seqLen, dFfn]
    Gelu.applyInPlace(h);                     // 逐元素激活
    return fc2.forwardBatch(h, seqLen);       // [seqLen, dModel]
}
```

FFN 对每个 token 独立作用（不同 token 间无交互，那是 attention 的职责）。dFfn=4×dModel 是参数量主要来源。

## 6. Embedding：查表与投影

[`Embedding`](../src/main/java/io/leavesfly/minivllm/model/Embedding.java) 既是输入查表，也是输出投影（tied weights）：

```java
// 输入：token id → 向量
public float[] lookup(int id) {
    return weight[id * dModel ..];  // 第 id 行
}

// 输出：hidden → 词表 logits（lm_head 与 wte 共享权重）
public float[] projectToVocab(float[] hidden) {
    for (i in vocabSize) logits[i] = dot(hidden, weight[i]);
    return logits;
}
```

tied embeddings 让 lm_head 复用 wte 权重，省一半参数（GPT-2 默认）。

## 7. 手写算子（math 层）

所有张量运算退化为对一维 `float[]` 的逐元素操作，详见 [Architecture](Architecture.md) 的 math 层。

| 算子 | 实现 | 用在哪 |
|---|---|---|
| `Matmul.matVec` | 三重循环点积 | Linear 前向 |
| `Softmax.softmaxInPlace` | 减最大值+exp+归一化 | attention、采样 |
| `LayerNorm` | 均值方差+仿射 | 每个 sublayer 前后 |
| `Gelu.gelu` | tanh 多项式近似 | FFN 激活 |
| `Sampler.sample` | temperature+topK+topP+多项采样 | decode 选 token |

手写这些算子是用 PyTorch 调 API 学不到的——你会真正理解每个矩阵乘、每个 softmax 在干什么。

## 8. 权重布局约定

Linear 权重按 PyTorch `nn.Linear` 约定：`[outFeatures, inFeatures]` 行优先，计算 `y = x·Wᵀ + b`。这与 safetensors 加载直接对应，无需转置。详见 [Weight Loading](Weight-Loading.md)。

---

**相关**：[PagedAttention](PagedAttention.md) · [Weight Loading](Weight-Loading.md)
