# KV Cache 内存管理

聚焦 [`memory`](../src/main/java/io/leavesfly/minivllm/memory) 包的内存管理视角：KV cache 如何分配、回收、共享、容量规划。注意力计算如何使用分页 KV 见 [PagedAttention](PagedAttention.md)。

## 1. KV cache 是什么

Transformer 生成每个新 token 时，需要用到之前所有 token 的 K 和 V（注意力机制）。如果每次都重算，复杂度是 O(n²)；把已算的 K/V 存下来复用，就降到 O(n)——这就是 **KV cache**。

```
prompt: [Tell][me][a][joke]        → prefill 算出 4 个 token 的 K/V，存入 cache
decode: [Why]                       → 只算 Why 的 Q，复用 cache 里 4 个 K/V
decode: [did]                       → 算 did 的 Q，复用 5 个 K/V（含 Why）
...
```

**KV cache 随回答增长，全部驻留显存**——它是决定能同时服务多少用户的关键瓶颈，而非算力。

## 2. 朴素管理 vs PagedAttention

| | 朴素 | PagedAttention |
|---|---|---|
| 分配时机 | 请求到来即预留整块 | 用到才给一块 |
| 块大小 | 按最长回答预留（如 2000 token） | 固定小块（16 token） |
| 连续性 | 必须连续 | 可散落 |
| 内部碎片 | 严重（预留未用） | 无（按需） |
| 外部碎片 | 严重（大块间空隙） | 无（等大块） |
| 共享 | 不支持 | 引用计数支持 |

## 3. BlockPool：物理池

[`BlockPool`](../src/main/java/io/leavesfly/minivllm/memory/BlockPool.java) 是全局共享的物理内存池，所有层、所有请求的 block 都从中分配。

```java
BlockPool(int numBlocks, int blockSize, int dModel)
```

- `numBlocks`：block 总数，即"模拟显存容量"。启动时由 `--num-blocks` 控制（默认 1024）。
- `blockSize`：每 block 容纳的 token 数（默认 16）。
- 每个 block 持有 `k[blockSize*dModel]` 和 `v[blockSize*dModel]` 两段数组。

**分配回收**：

```java
int allocate()           // 取空闲 block，refCount=1；池满返回 -1
void retain(int id)      // 引用 +1（共享时）
void release(int id)     // 引用 -1，归零才归还 freeList
int freeBlocks()         // 剩余可用 block 数
int usedBlocks()         // 已用 block 数
```

freeList 用 `ArrayDeque`，`release` 时 `addFirst`（LIFO），利于缓存局部性。

## 4. BlockTable：逻辑映射

每个请求的每层各有一个 [`BlockTable`](../src/main/java/io/leavesfly/minivllm/memory/BlockTable.java)，记录该层 KV cache 的逻辑顺序对应的物理 block id 序列。

```java
BlockTable bt = ...;
bt.append(blockId);          // 追加一个物理 block
bt.blockIdAt(logicalIdx);    // 取第 logicalIdx 个逻辑 block 的物理 id
bt.numBlocks();              // 已分配的逻辑 block 数
bt.capacity(blockSize);      // 当前最多承载的 token 数 = numBlocks * blockSize
```

token 下标 → 物理位置的换算：

```
logicalBlock = tokenIdx / blockSize
slot         = tokenIdx % blockSize
physicalId   = bt.blockIdAt(logicalBlock)
```

## 5. KVCacheManager：协调者

[`KVCacheManager`](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 封装 BlockPool + BlockTable 的高层操作。

### 按需扩容

```java
boolean ensureCapacity(BlockTable bt, int requiredTokens)
```

只在当前 block 数不够时申请新 block。调度器在 prefill 前、decode 每步前调用。返回 false 表示显存不足。

### 读写 KV

```java
void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v)  // 写
float[] readK(BlockTable bt, int tokenIdx)                        // 读（返回拷贝）
float[] blockK(BlockTable bt, int logicalBlockIdx)                // 取整 block 的 K 数组（供 attention 遍历）
```

### 释放

```java
void free(BlockTable bt)
```

释放该请求所有 block（按引用计数）。共享 block 不会立即回收。

## 6. 生命周期

一个请求的 KV cache 经历：

```
admitNew:
  ensureCapacity(promptLen)   → 分配 prompt 所需 block
  prefill: writeKV(0..promptLen-1)  → 填充 K/V

decodeStep (每步):
  ensureCapacity(totalLen)    → 新 token 需要时追加 block
  decode: writeKV(curIdx)     → 写当前 token 的 K/V
  decodePaged: 读所有历史 block → 算 attention

sweepFinished:
  free(bt)                    → release 所有 block，归还 freeList
```

## 7. 前缀共享

[`trySharePrefix`](../src/main/java/io/leavesfly/minivllm/memory/KVCacheManager.java) 让新请求复用已缓存的相同前缀 block：

```java
int trySharePrefix(int[] tokens, BlockTable bt)
```

机制：用 token 序列的 block 级指纹（哈希）查 `PrefixCache`，命中则 `retain` 该 block 并加入新请求的 BlockTable，跳过这部分 prefill。

两个场景：
- **相同系统提示**：多用户共用 system prompt，KV 只存一份。
- **Beam search**：多候选共享前缀，分叉处才各自分配。

> 当前版本前缀缓存是全局单层（学习用，有哈希冲突风险）。真实 vLLM 用 RadixAttention 树做精确匹配。启用前缀共享需改为 per-layer 管理，是进阶练习。

## 8. 容量规划

block 池需要够大才能支撑满载并发。粗略估算：

```
所需 block 数 ≈ maxNumSeqs × nLayer × ceil(maxSeqLen / blockSize)
```

例：maxNumSeqs=8, nLayer=2, maxSeqLen=512, blockSize=16：

```
8 × 2 × (512/16) = 8 × 2 × 32 = 512 blocks
```

默认 `--num-blocks 1024` 足够。若显存不足（admitNew 返回 false），调度器会暂停接纳新请求，等 running 释放后再试——不会崩溃。

## 9. 监控

启动时不加 `--quiet` 可看每步调度状态：

```
[engine] running=3 waiting=2 freeBlocks=984
```

`freeBlocks` 持续走低说明显存趋紧；随请求完成回升说明回收正常。这是观察 PagedAttention + continuous batching 协同运转的窗口。

---

**相关**：[PagedAttention](PagedAttention.md) · [Continuous Batching](Continuous-Batching.md)
