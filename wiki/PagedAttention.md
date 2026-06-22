# PagedAttention

PagedAttention 是 vLLM 的核心创新，也是 mini-vllm 最值得深入理解的部分。本文结合代码讲清"为什么"和"怎么做"。

## 1. 问题：KV cache 吃显存

Transformer 生成时，每个 token 都要复用之前所有 token 的 K、V（否则要重算，极慢）。这些 K、V 缓存就是 **KV cache**，且随回答变长不断增长，全部驻留 GPU 显存。

**朴素做法**：为每个请求预留一块连续显存，足够装下最长可能回答（如 2000 token）。

这导致两个浪费：
- **过度预留（over-reservation）**：多数回答只有 50 token，预留的 1950 token 空间闲置却不能给别人用。
- **碎片化（fragmentation）**：大块之间的零散空隙装不下新请求。

```
Request A: [## used 50 ##][.......... wasted 1950 ..........]
Request B: [#### used 120 ####][........ wasted 1880 ........]
free gaps: 太小太散，装不下新请求
```

结果：显存纸上够用，实际能服务的用户很少。

## 2. 核心思想：像 OS 管理虚拟内存一样管 KV cache

操作系统的解法是**分页**：进程的虚拟内存被切成固定大小的页，映射到散落的物理页框，靠页表记录对应关系。

PagedAttention 把同一思想用到 KV cache：
- 把 KV cache 显存切成**固定大小的 block**（本项目默认 16 token/block）。
- 不再为请求预留一整块，而是**用到才给一块**（按需分配）。
- block 之间**不需要连续**，靠 `BlockTable` 记录逻辑顺序。
- 因为 block 等大，任意空闲块都能用——**无内部碎片，无外部碎片**。

## 3. BlockPool：物理内存池

[`BlockPool`](../src/main/java/io/leavesfly/minivllm/memory/BlockPool.java) 是 PagedAttention 的物理层，管理所有 block 的分配与回收。

```java
public final class BlockPool {
    public final int blockSize;   // 每 block 容纳的 token 数（默认 16）
    public final int dModel;      // 每个 token 的 K/V 向量长度
    public final int numBlocks;   // block 总数（模拟显存容量）
    private final KVBlock[] blocks;
    private final Deque<Integer> freeList;  // 空闲 block id 队列
}
```

每个 `KVBlock` 持有一段 K 和一段 V 数组（行优先 `[blockSize, dModel]`）：

```java
public static final class KVBlock {
    public final int id;
    public final float[] k;  // [blockSize * dModel]
    public final float[] v;  // [blockSize * dModel]
    private int refCount = 0;  // 引用计数
}
```

关键操作：
- `allocate()`：从 freeList 取一个 block，refCount=1，返回 id；池满返回 -1。
- `retain(id)`：引用 +1（用于共享）。
- `release(id)`：引用 -1，归零才真正归还 freeList。

> **对比真实 vLLM**：那里 block 存的是 GPU 显存指针，这里用 Java 堆数组模拟，但管理逻辑完全一致。

## 4. BlockTable：逻辑→物理映射

[`BlockTable`](../src/main/java/io/leavesfly/minivllm/memory/BlockTable.java) 是每个请求的"页表"，记录其 KV cache 逻辑顺序对应的物理 block id 序列。

```java
public final class BlockTable {
    private final List<Integer> blockIds;  // 按逻辑顺序排列的物理 block id
}
```

逻辑 token 下标到物理位置的换算：

```
logicalBlockIdx = tokenIdx / blockSize
slotInBlock     = tokenIdx % blockSize
physicalBlockId = blockTable.get(logicalBlockIdx)
```

这正是操作系统中"虚拟页号 → 物理页框号"的映射。block 在物理池中可散落任意位置，`BlockTable` 把它们串成逻辑连续的 KV cache。

## 5. KVCacheManager：按需分配与读写

[`KVCacheManager`](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 协调 `BlockPool` 与各请求的 `BlockTable`。

**按需分配**——只在容量不足时申请新 block：

```java
public boolean ensureCapacity(BlockTable bt, int requiredTokens) {
    int needed = (requiredTokens + blockSize - 1) / blockSize;
    while (bt.numBlocks() < needed) {
        int id = pool.allocate();
        if (id < 0) return false;  // 池满，调度器应阻塞新请求
        bt.append(id);
    }
    return true;
}
```

**写入 KV**——把 token 的 K/V 落到正确 block 的正确槽位：

```java
public void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v) {
    int logicalBlock = tokenIdx / blockSize;
    int slot = tokenIdx % blockSize;
    KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
    int off = slot * dModel;
    System.arraycopy(k, 0, blk.k, off, dModel);
    System.arraycopy(v, 0, blk.v, off, dModel);
}
```

## 6. Attention 中的 block-wise 累加

PagedAttention 最精彩的部分在 attention 计算。decode 阶段，历史 K/V 散落在多个物理 block 中，如何计算注意力？

[`Attention.decodePaged`](../src/main/java/io/leavesfly/minivllm/model/Attention.java) 用**逐 block 累加**实现，无需全局拷贝：

```java
// 第一遍：逐 block 计算所有历史 token 的 attention scores
for (int b = 0; b < nBlocks; b++) {
    float[] kBlk = kvMgr.blockK(bt, b);   // 该 block 的 K 数组
    for (int s = 0; s < tokensInBlock; s++) {
        scores[idx++] = dot(q, kBlk[s]) / sqrt(headDim);
    }
}
Softmax.softmaxInPlace(scores);

// 第二遍：逐 block 加权 V 累加输出
for (int b = 0; b < nBlocks; b++) {
    float[] vBlk = kvMgr.blockV(bt, b);
    for (int s = 0; s < tokensInBlock; s++) {
        out += scores[idx++] * vBlk[s];
    }
}
```

这就是 PagedAttention 的真正语义：**不假设 K/V 连续，而是通过 BlockTable 找到散落的 block，逐 block 读取并累加**。真实 vLLM 用自定义 CUDA kernel 做同样的事，只是跑在 GPU 上。

### 对照版：decodeGather

为帮助理解，项目还提供 [`decodeGather`](../src/main/java/io/leavesfly/minivllm/model/Attention.java)：先把散落 block 的 K/V **拷贝**成连续数组，再做标准 attention。

| | decodePaged | decodeGather |
|---|---|---|
| 拷贝 | 无 | 有 O(n·d) 拷贝 |
| 逻辑 | 逐 block 累加 | gather 后标准 attention |
| 接近原意 | ✅ 真正 PagedAttention | ❌ 仅为教学对照 |
| 结果 | 数学等价 | 数学等价 |

> 学习建议：写断言对比两者输出，确认实现正确。

## 7. 引用计数与前缀共享

因为 block 是独立单元，多个请求可以指向同一 block——这就是**共享**。

[`KVCacheManager.trySharePrefix`](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 实现前缀共享：新请求若与已缓存请求有相同 token 前缀，复用已有 block 而非重算。

```java
public int trySharePrefix(int[] tokens, BlockTable bt) {
    for (int b = 0; b < nFullBlocks; b++) {
        long fp = fingerprint(tokens, b * blockSize, blockSize);
        Integer id = prefixCache.get(fp);
        if (id == null) break;      // 前缀不再匹配，停止
        pool.retain(id);            // 引用 +1
        bt.append(id);
    }
}
```

两个典型应用场景：
- **相同系统提示**：多个用户共用同一段 system prompt，只存一份 KV。
- **Beam search**：多个候选回答共享前缀，只在分叉处用各自 block。

释放时靠引用计数：`release` 让 refCount-1，归零才真正回收，保证共享 block 不会在使用中被释放。

> **注意**：当前版本的前缀缓存是全局单层（学习用），真实 vLLM 用 RadixAttention 树做精确匹配，且需 per-layer 管理。启用前缀共享是进阶练习。

## 8. 完整数据流

```
请求到来，prompt = "Hello" (5 token)
  │
  ▼
ensureCapacity(bt, 5) → 分配 1 个 block (可装 16 token)
  │  BlockPool.allocate() → block #7, bt = [7]
  ▼
prefill: 5 个 token 的 K/V 写入 block #7 的 slot 0-4
  │  writeKV(bt, 0..4, k, v)
  ▼
decode token 5: ensureCapacity(bt, 6) → 仍只需 1 block
  decodePaged: 从 block #7 读 slot 0-5 的 K/V 累加 attention
  ▼
decode token 16: ensureCapacity(bt, 17) → 需要 2 block
  BlockPool.allocate() → block #3, bt = [7, 3]
  decodePaged: 遍历 block #7 (slot 0-15) + block #3 (slot 0) 累加
  ▼
请求完成: free(bt) → release(7), release(3) → 归还 freeList
```

---

**相关**：[KV Cache Memory](KV-Cache-Memory.md) · [Continuous Batching](Continuous-Batching.md)
