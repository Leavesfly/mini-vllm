# KV Cache Memory

本文聚焦 KV cache 的**内存管理**：分页分配、按需扩容、引用计数回收、前缀共享，以及显存占用如何估算。它与 [PagedAttention](PagedAttention.md) 互补——PagedAttention 讲"attention 怎么用 cache"，本文讲"cache 的内存怎么管"。

## 为什么需要 KV cache

自回归生成中，第 t 步要计算当前 token 对**所有历史 token** 的注意力。历史 token 的 Key/Value 向量不变，若每步都重算整段历史，复杂度是 O(n²) 且大量重复。KV cache 把每个 token 的 K/V 存下来，decode 时只算新 token 的 Q，与缓存的历史 K/V 做注意力——把每步复杂度降到 O(n)。

代价是**显存**：序列越长、并发越多、层数越深，KV cache 越大。如何高效管理这块显存，正是 PagedAttention 内存管理要解决的。

## 三层结构

```
KVCacheManager（总指挥：分配/读写/释放/前缀共享）
     │ 持有
     ├── BlockPool（物理块池，模拟显存）
     │      └── KVBlock[]（每块 blockSize 个 token 的 K 与 V）
     └── 被 Sequence 持有的 BlockTable[nLayer]（逻辑→物理页表，每层一张）
```

- [BlockPool](../src/main/java/io/leavesfly/minivllm/memory/BlockPool.java)：固定数量的等大物理块，用空闲 id 队列管理。
- [BlockTable](../src/main/java/io/leavesfly/minivllm/memory/BlockTable.java)：一个请求某一层的逻辑顺序 → 物理 block id 映射。
- [KVCacheManager](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java)：对外统一入口。

## 内存布局

每个 `KVBlock` 持有两段行优先数组，形状 `[blockSize, dim]`：

```java
public final float[] k; // [blockSize * dim]
public final float[] v; // [blockSize * dim]
```

- `dim` 是**每 token 的 K/V 向量长度**，由 `KVCacheManager` 构造时传入。
  - Qwen3（GQA）：`dim = kvDim = nKVHead × headDim = 8 × 128 = 1024`
  - GPT-2：`dim = dModel`
- token 在块内的偏移：第 slot 个 token 的 K 起始 = `slot * dim`。
- 逻辑 token 下标 → 物理位置：
  ```
  logicalBlockIdx = tokenIdx / blockSize
  slot            = tokenIdx % blockSize
  physicalBlockId = blockTable.blockIdAt(logicalBlockIdx)
  ```

> 命名提醒：`BlockPool`/`KVCacheManager` 的字段叫 `dModel`，但实际语义是 kvDim（构造时传入的向量长度）。GQA 下它是 kvDim 而非 dModel。

## 分配策略：按需 + 懒创建

### 按需分配（ensureCapacity）

只在容量不足时申请新 block，杜绝一次性预留整段显存：

```java
public boolean ensureCapacity(BlockTable bt, int requiredTokens) {
    int needed = (requiredTokens + blockSize - 1) / blockSize; // 向上取整所需块数
    while (bt.numBlocks() < needed) {
        int id = pool.allocate();
        if (id < 0) return false;   // 池满，返回 false 让调度器阻塞新请求
        bt.append(id);
    }
    return true;
}
```

- prefill 时按 `promptLen` 分配，decode 时按 `curIdx+1` 增量扩容。
- 返回 `false` 是显存不足信号，由 [LLMEngine](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java) 处理（admit 时回滚等待，decode 时置 ABORTED）。

### 懒创建（lazy allocation）

`BlockPool` 启动时只建 id 队列，`KVBlock` 底层数组首次 `allocate` 才创建：

```java
public int allocate() {
    Integer id = freeList.pollFirst();
    if (id == null) return -1;
    if (blocks[id] == null) blocks[id] = new KVBlock(id, blockSize, dModel); // 首用才建
    blocks[id].refCount = 1;
    usedBlocks++;
    return id;
}
```

这样即使 `numBlocks` 设得很大，也不会启动即占满堆——只有真正用到的块才分配内存。

## 回收策略：引用计数

释放某请求某层的全部 block（`sweepFinished` 中对每层 BlockTable 调用）：

```java
public void free(BlockTable bt) {
    for (int id : bt.toArray()) pool.release(id);
    bt.clear();
}
```

`release` 按引用计数回收，归零才真正归还到空闲队列：

```java
public void release(int blockId) {
    KVBlock b = blocks[blockId];
    b.refCount--;
    if (b.refCount == 0) { freeList.addFirst(blockId); usedBlocks--; } // LIFO 归还
    else if (b.refCount < 0) throw new IllegalStateException("引用计数下溢");
}
```

两个设计点：
- **引用计数**：共享 block（前缀共享/beam search）被多个请求指向时，只有最后一个释放才回收——防止误删仍在用的 block。
- **LIFO 归还**（`addFirst`）：刚释放的 block 最先被复用，利于 CPU 缓存局部性。
- **下溢检查**：refCount < 0 直接抛异常，帮助学习期尽早暴露"重复释放"的 bug。

回收的 block 立刻回到 freeList，下一个 step 的 admitNew 就能复用——这是与 [Continuous-Batching](Continuous-Batching.md) 协同的闭环。

## 读写 KV

写入（prefill/decode 阶段落位新 token 的 K/V）：

```java
public void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v) {
    int off = (tokenIdx % blockSize) * dModel;
    BlockPool.KVBlock blk = pool.get(bt.blockIdAt(tokenIdx / blockSize));
    System.arraycopy(k, 0, blk.k, off, dModel);
    System.arraycopy(v, 0, blk.v, off, dModel);
}
```

读取有两组接口：
- `readK/readV(bt, tokenIdx)`：返回**拷贝**，语义清晰，用于教学/测试。
- `blockK/blockV(bt, logicalBlockIdx)`：直接返回物理 block 的底层数组（**零拷贝**），attention 的 decode 逐 block 遍历用它以避免大量小数组分配。

## 前缀共享（Prefix Cache）

多个请求含相同 token 前缀（如共享 system prompt）时，可复用已缓存的 block，省算又省存。

### 复用已有前缀

```java
public int trySharePrefix(int[] tokens, BlockTable bt) {
    int shared = 0;
    int nFullBlocks = tokens.length / blockSize;   // 只共享"整块"前缀
    for (int b = 0; b < nFullBlocks; b++) {
        long fp = prefixCache.fingerprint(tokens, b*blockSize, blockSize);
        Integer id = prefixCache.get(fp);
        if (id == null) break;   // 前缀断了就停
        pool.retain(id);         // 引用 +1
        bt.append(id);
        shared += blockSize;
    }
    return shared;   // 返回命中的 token 数（这些无需重算 prefill）
}
```

### 注册前缀供后续复用

prefill 完成后调用，把本请求写满的整块登记：

```java
public void registerPrefix(int[] tokens, BlockTable bt) {
    int nFullBlocks = Math.min(tokens.length / blockSize, bt.numBlocks());
    for (int b = 0; b < nFullBlocks; b++)
        prefixCache.put(prefixCache.fingerprint(tokens, b*blockSize, blockSize), bt.blockIdAt(b));
}
```

指纹是 block 内容的滚动哈希 `h = h*131 + token`。真实 vLLM 用 **RadixAttention** 前缀树做精确匹配，这里简化为 block 级哈希映射，学习项目可接受极小概率冲突（源码已注释提醒生产需校验内容）。

> 当前 [LLMEngine.admitNew](../src/main/java/io/leavesfly/minivllm/core/LLMEngine.java) 走的是全量 prefill（`startIdx=0`），前缀共享是内存层已实现的能力接口，接入引擎可作为一个很好的扩展练习（见 [Learning-Path](Learning-Path.md)）。

## 显存占用估算

KV cache 的总内存（Java 堆数组）：

```
bytes = numBlocks × blockSize × kvDim × 2(K和V) × 4(float)
```

[MiniVllmServer](../src/main/java/io/leavesfly/minivllm/MiniVllmServer.java) 启动时会打印这个估算：

```java
long kvBytes = (long) numBlocks * cfg.blockSize * kvDim * 2 * 4;
System.out.printf("KV 池: %d blocks × blockSize=%d × kvDim=%d（满载约 %.1f GB）%n", ...);
```

`numBlocks` 未指定时自动估算，保证满并发不 OOM（注意每层独立，故乘 nLayer）：

```
blocksPerSeq = ceil(maxSeqLen / blockSize)
numBlocks    = max(1024, maxNumSeqs × nLayer × blocksPerSeq)
```

以 Qwen3-0.6B 默认参数（blockSize=16，kvDim=1024，nLayer=28，maxSeqLen=2048，maxSeqs=2）估算：
- blocksPerSeq = ceil(2048/16) = 128
- numBlocks = max(1024, 2×28×128) = 7168
- 单块 = 16×1024×2×4 = 128 KB；总计约 7168 × 128 KB ≈ **0.9 GB**

想降内存：调小 `--max-seq-len`、`--max-seqs` 或 `--num-blocks`。

## 内存碎片：为什么分页几乎无碎片

| 碎片类型 | 传统连续预留 | mini-vllm 分页 |
|---|---|---|
| 内部碎片 | 严重（按 max_seq_len 预留，实际远短） | 极小（最多浪费最后一个 block 尾部） |
| 外部碎片 | 严重（长度不一，大块难复用） | 无（block 等大，任意空闲块可用） |

这正是 vLLM 显存利用率远高于传统实现的根本原因。

## 测试验证

| 测试类 | 验证点 |
|---|---|
| [BlockPoolTest](../src/test/java/io/leavesfly/minivllm/memory/BlockPoolTest.java) | allocate/release、refCount retain/release、池满返回 -1、下溢抛异常 |
| [BlockTableTest](../src/test/java/io/leavesfly/minivllm/memory/BlockTableTest.java) | append/blockIdAt 映射、numBlocks/capacity |
| [KVCacheManagerTest](../src/test/java/io/leavesfly/minivllm/memory/KVCacheManagerTest.java) | ensureCapacity 按需分配、writeKV↔readK/V 往返一致、free 释放归还 |

## 延伸

- attention 如何逐 block 读取这些 KV：[PagedAttention](PagedAttention.md)
- 谁在何时 ensureCapacity/free：[Continuous-Batching](Continuous-Batching.md)
- kvDim 从何而来（GQA）：[Transformer](Transformer.md)
