# PagedAttention

PagedAttention 是 vLLM 的核心创新，也是本项目最值得吃透的机制。本文讲清楚它解决什么问题、mini-vllm 如何用纯 Java 逐 block 模拟它，以及 GQA 场景下的实现细节。

## 问题：KV cache 的显存浪费

自回归生成时，每个 token 的 K/V 向量要缓存下来供后续 token 复用（否则每步都要重算整段历史）。传统做法是为每个请求**预留一整段连续显存**（按 `max_seq_len` 上限）：

- **内部碎片**：实际生成远短于 max_seq_len，预留的尾部全部浪费。
- **外部碎片**：请求长度不一，连续大块难以复用，显存越用越零散。
- 结果：显存利用率常常只有 20~40%，能同时跑的请求数被严重限制。

## 解法：像操作系统分页一样管理 KV cache

PagedAttention 把 KV cache 切成**固定大小的物理块（block）**，每块容纳 `blockSize` 个 token 的 K 和 V。逻辑上连续的 KV 序列，物理上可以散落在任意空闲块中，靠一张"页表"（BlockTable）串起来。

这正是操作系统虚拟内存的思想：

| 操作系统 | PagedAttention | mini-vllm 类 |
|---|---|---|
| 物理页框 | 物理 KV block | `BlockPool.KVBlock` |
| 页表 | per-seq block table | `BlockTable` |
| 虚拟页号→物理页框号 | 逻辑 token→物理 block | `tokenIdx / blockSize` → `blockTable.blockIdAt(...)` |
| 页框分配器 | block 分配器 | `BlockPool.allocate/release` |

收益：
- **几乎无内部碎片**：block 等大，最多浪费最后一个 block 的尾部。
- **无外部碎片**：任意空闲 block 都能用，不要求连续。
- **天然支持共享**：多个请求指向同一 block（前缀共享 / beam search），靠引用计数管理。

## mini-vllm 的三个核心类

### BlockPool —— 模拟显存的物理块池

[BlockPool](../src/main/java/io/leavesfly/minivllm/memory/BlockPool.java) 用 Java 堆数组模拟 GPU 显存。每个 `KVBlock` 持有两段行优先数组：

```java
public static final class KVBlock {
    public final float[] k; // [blockSize * dModel]，token t 的 K 起始偏移 = t * dModel
    public final float[] v; // [blockSize * dModel]
    private int refCount = 0;
}
```

分配与释放靠一个空闲 id 队列（`freeList`）：

```java
public int allocate() {
    Integer id = freeList.pollFirst();
    if (id == null) return -1;              // 池满 → 调度器据此暂停接纳新请求
    if (blocks[id] == null) blocks[id] = new KVBlock(id, blockSize, dModel); // 懒分配
    blocks[id].refCount = 1;
    usedBlocks++;
    return id;
}

public void release(int blockId) {
    KVBlock b = blocks[blockId];
    b.refCount--;
    if (b.refCount == 0) { freeList.addFirst(blockId); usedBlocks--; } // 归零才真正归还
    else if (b.refCount < 0) throw new IllegalStateException("引用计数下溢");
}
```

两个学习要点：
- **懒分配**：`KVBlock` 底层数组首次 `allocate` 时才创建，避免启动即占满堆。
- **LIFO 归还**（`addFirst`）：刚释放的 block 优先复用，利于缓存局部性。
- `allocate()` 返回 -1 是显存不足信号，调度器据此暂停接纳（见 [Continuous-Batching](Continuous-Batching.md)）。

### BlockTable —— 逻辑到物理的页表

[BlockTable](../src/main/java/io/leavesfly/minivllm/memory/BlockTable.java) 只是一个物理 block id 列表，记录 KV cache 的逻辑顺序：

```java
logicalBlockIdx = tokenIdx / blockSize;   // 第几个逻辑块
slotInBlock     = tokenIdx % blockSize;   // 块内第几槽
physicalBlockId = blockTable.blockIdAt(logicalBlockIdx);
```

**每个请求每层一张 BlockTable**——因为每层注意力有独立的 KV cache。见 [Sequence](../src/main/java/io/leavesfly/minivllm/core/Sequence.java) 的 `blockTables = new BlockTable[nLayer]`。

### KVCacheManager —— 内存管理总指挥

[KVCacheManager](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 封装"按需分配、读写 KV、释放、前缀共享"。按需分配是精髓——只在容量不足时申请新 block：

```java
public boolean ensureCapacity(BlockTable bt, int requiredTokens) {
    int needed = (requiredTokens + blockSize - 1) / blockSize; // 向上取整
    while (bt.numBlocks() < needed) {
        int id = pool.allocate();
        if (id < 0) return false; // 池满，调度器阻塞新请求
        bt.append(id);
    }
    return true;
}
```

写入某个 token 的 K/V，先算逻辑块和槽位，再拷贝到物理 block：

```java
public void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v) {
    int logicalBlock = tokenIdx / blockSize;
    int slot = tokenIdx % blockSize;
    BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
    int off = slot * dModel; // 这里的 dModel 实为 kvDim（构造时传入）
    System.arraycopy(k, 0, blk.k, off, dModel);
    System.arraycopy(v, 0, blk.v, off, dModel);
}
```

> 关于 `dModel`：`KVCacheManager` 构造时传入的第三参数是**每 token 的 K/V 向量长度**。Qwen3（GQA）传的是 `cfg.kvDim() = nKVHead * headDim`（Qwen3-0.6B 为 1024），GPT-2 传的是 `cfg.dModel`。字段名叫 dModel 只是历史命名，实际语义是 kvDim。详见 [KV-Cache-Memory](KV-Cache-Memory.md)。

## Attention 如何用 KV cache：prefill vs decode

以 [Qwen3Attention](../src/main/java/io/leavesfly/minivllm/model/Qwen3Attention.java) 为例，两个阶段用同一份 KV cache，但访问方式不同。

### Prefill：一次性处理整段 prompt

Prompt 的 K/V 连续在手，先写入 cache，再直接做 causal attention：

```java
public float[] prefill(float[] input, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
    float[] q = qProj.forwardBatch(input, seqLen);
    float[] k = kProj.forwardBatch(input, seqLen);
    float[] v = vProj.forwardBatch(input, seqLen);
    for (int t = 0; t < seqLen; t++) {
        int pos = startIdx + t;
        applyQkNormAndRope(q, t*qDim, k, t*kvDim, pos); // QK-Norm + RoPE
        // 抽出第 t 个 token 的 K/V 写入 cache
        kvMgr.writeKV(bt, pos, kt, vt);
    }
    float[] out = causalAttention(q, k, v, seqLen); // K/V 在手，直接算
    return oProj.forwardBatch(out, seqLen);
}
```

### Decode：逐 block 累加（模拟 PagedAttention kernel）

生成阶段只有一个新 token 的 Q，历史 K/V 散落在物理 block 中。这里就是本项目对 **PagedAttention CUDA kernel 的 Java 模拟**——遍历该请求的每个逻辑 block，逐块累加：

```java
public float[] decodePaged(float[] hidden, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
    float[] q = qProj.forward(hidden);
    float[] k = kProj.forward(hidden);
    float[] v = vProj.forward(hidden);
    applyQkNormAndRope(q, 0, k, 0, curIdx);
    kvMgr.writeKV(bt, curIdx, k, v);       // 新 token 的 K/V 落位

    int totalTokens = curIdx + 1;
    int nBlocks = bt.numBlocks();
    float invSqrt = 1f / (float) Math.sqrt(headDim);
    float[] out = new float[qDim];

    for (int h = 0; h < nHead; h++) {
        int qOff = h * headDim;
        int kvOff = (h / group) * headDim;  // GQA：Q 头 h 读 KV 头 h/group
        // 第一遍：逐 block 算 attention score
        float[] scores = new float[totalTokens];
        for (int b = 0; b < nBlocks; b++) {
            float[] kBlk = kvMgr.blockK(bt, b);            // 直接拿物理 block 的 K 数组
            int tokensInBlock = Math.min(blockSize, Math.max(0, totalTokens - b*blockSize));
            for (int s = 0; s < tokensInBlock; s++) {
                int kOff = s * kvDim + kvOff;
                float sc = 0f;
                for (int d = 0; d < headDim; d++) sc += q[qOff+d] * kBlk[kOff+d];
                scores[b*blockSize + s] = sc * invSqrt;
            }
        }
        Softmax.softmaxInPlace(scores);
        // 第二遍：逐 block 加权 V 累加输出
        for (int b = 0; b < nBlocks; b++) {
            float[] vBlk = kvMgr.blockV(bt, b);
            int tokensInBlock = Math.min(blockSize, Math.max(0, totalTokens - b*blockSize));
            for (int s = 0; s < tokensInBlock; s++) {
                float w = scores[b*blockSize + s];
                int vOff = s * kvDim + kvOff;
                for (int d = 0; d < headDim; d++) out[qOff+d] += w * vBlk[vOff+d];
            }
        }
    }
    return oProj.forward(out);
}
```

关键点：
- **两遍遍历**：第一遍算 QKᵀ 得分（需先收集全部得分再 softmax），第二遍加权 V。真实 kernel 用 online-softmax 单遍完成，这里为可读性拆成两遍。
- **block 内行 stride 为 kvDim**：第 s 个 token 的 K 起始偏移是 `s * kvDim`，再加 GQA 的头偏移 `kvOff`。
- **最后一个 block 可能没填满**：`tokensInBlock` 用 `totalTokens - b*blockSize` 截断，只遍历有效 token。

## GQA 下的分页语义

Qwen3 用 GQA（Grouped Query Attention）：`nHead` 个 Q 头共享 `nKVHead` 个 KV 头，`group = nHead / nKVHead`。这对 PagedAttention 的影响是：

- KV cache 每 token 只存 `kvDim = nKVHead * headDim`（而非 `qDim`），Qwen3-0.6B 是 1024 而非 2048——**显存与带宽都省 group 倍**（这里 group=2）。
- 分页机制完全不变，只是 block 内每 token 的 K/V 长度变成 kvDim，Q 头 h 读取时映射到 KV 头 `h / group`。

## 前缀共享：引用计数的用武之地

多个请求若含相同 token 前缀（如相同 system prompt），可复用已缓存的 block，省算省存。[KVCacheManager](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 提供接口：

```java
public int trySharePrefix(int[] tokens, BlockTable bt) {
    int shared = 0;
    int nFullBlocks = tokens.length / blockSize;
    for (int b = 0; b < nFullBlocks; b++) {
        long fp = prefixCache.fingerprint(tokens, b*blockSize, blockSize);
        Integer id = prefixCache.get(fp);
        if (id == null) break;   // 前缀不再匹配
        pool.retain(id);         // 引用 +1，共享该 block
        bt.append(id);
        shared += blockSize;
    }
    return shared;
}
```

- 共享靠 `BlockPool.retain/release` 的引用计数：多个请求指向同一 block，只有最后一个释放才真正归还。
- 匹配靠 block 级内容指纹（`h = h*131 + token` 的滚动哈希）。真实 vLLM 用 RadixAttention 做精确前缀树匹配；这里简化为哈希映射，学习项目可接受极小概率冲突（源码注释已提醒生产需校验内容）。
- 注意：`LLMEngine` 当前的 `admitNew` 走的是全量 prefill 路径，前缀共享是内存层提供的能力接口，可作为扩展练习接入。

## 测试验证

- [BlockPoolTest](../src/test/java/io/leavesfly/minivllm/memory/BlockPoolTest.java)：分配/释放/引用计数/池满返回 -1。
- [BlockTableTest](../src/test/java/io/leavesfly/minivllm/memory/BlockTableTest.java)：逻辑→物理映射、容量计算。
- [KVCacheManagerTest](../src/test/java/io/leavesfly/minivllm/memory/KVCacheManagerTest.java)：按需分配、writeKV/readK/V 往返、free 释放。
- [Qwen3AlignmentTest](../src/test/java/io/leavesfly/minivllm/model/Qwen3AlignmentTest.java) 的 `greedyGenerate` 走的正是 prefill + 逐 token `decodeLogits`（PagedAttention 路径），并验证生成结果与 HF 逐 token 一致——这是对分页注意力数值正确性的终极验证。

## 延伸

- 内存分配/回收策略与显存估算：[KV-Cache-Memory](KV-Cache-Memory.md)
- 谁在何时调用 ensureCapacity/free：[Continuous-Batching](Continuous-Batching.md)
- attention 在整个前向中的位置：[Transformer](Transformer.md)
