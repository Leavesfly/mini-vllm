package io.leavesfly.minivllm.memory;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * KVCacheManager —— PagedAttention 内存管理总指挥。
 *
 * 职责：
 * 1. 按需分配 block（ensureCapacity）：token 数增长时才申请新 block，杜绝过度预留。
 * 2. 读写 KV：把每个 token 的 K/V 写入正确的物理 block 槽位，供 attention 读取。
 * 3. 释放：请求结束时按引用计数释放所有 block（共享 block 不会立即回收）。
 * 4. 前缀共享（PrefixCache）：多个请求若含相同 token 前缀，复用已缓存的 block，省算省存。
 *    - 链式哈希：block b 的指纹 = mix(block b-1 的指纹, block b 的 tokens)，
 *      相同内容出现在不同上下文不会被错误共享（对照 vLLM 的 parent-hash 链）。
 *    - LRU 驱逐：请求释放后已注册的 block 转入“缓存态”（不归还空闲池、不可被覆写），
 *      仅当池满且被 LRU 驱逐时才真正回收，杜绝共享到脏数据。
 * 5. INT8 量化存储（对照 vLLM --kv-cache-dtype int8）：writeKV 在线对称量化
 *    （per-token scale = absmax/127），attention 在量化域直接算点积/加权累加，
 *    不整块反量化物化 f32；KV 容量与带宽降为 f32 的约 1/4。
 *
 * 学习要点：这里没有一次性预留整段显存，而是“用到才给一块”，因此既无内部碎片
 * （block 等大，任意空闲块都能用），也几乎无外部碎片。这正是 vLLM 高吞吐的根基。
 */
public final class KVCacheManager {

    private final BlockPool pool;
    private final int blockSize;
    private final int dModel;
    private final boolean int8;

    /** 前缀缓存：block 内容指纹 -> 物理块 id，用于跨请求共享相同前缀 */
    private final PrefixCache prefixCache = new PrefixCache();

    public KVCacheManager(int numBlocks, int blockSize, int dModel) {
        this(numBlocks, blockSize, dModel, false);
    }

    /** @param int8 true 时 K/V 以 byte[] + per-token scale 量化存储（容量约为 f32 的 1/4） */
    public KVCacheManager(int numBlocks, int blockSize, int dModel, boolean int8) {
        if (numBlocks <= 0) throw new IllegalArgumentException("numBlocks 必须 > 0");
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize 必须 > 0");
        if (dModel <= 0) throw new IllegalArgumentException("dModel 必须 > 0");
        this.pool = new BlockPool(numBlocks, blockSize, dModel, int8);
        this.blockSize = blockSize;
        this.dModel = dModel;
        this.int8 = int8;
    }

    public BlockPool pool() {
        return pool;
    }

    public int blockSize() {
        return blockSize;
    }

    public int dModel() {
        return dModel;
    }

    /** KV cache 是否 INT8 量化存储（attention 层据此选择量化域读取路径） */
    public boolean isInt8() {
        return int8;
    }

    /** 剩余可用 block 数（便捷委托） */
    public int freeBlocks() {
        return pool.freeBlocks();
    }

    /**
     * 确保某请求的 BlockTable 至少能承载 requiredTokens 个 token。
     * 只在容量不足时申请新 block（按需分配）；空闲池耗尽时先尝试驱逐
     * 前缀缓存中最久未用的条目腾出 block（对照 vLLM 的 cache eviction）。
     * @return false 表示连驱逐都腾不出 block，调度器应阻塞新请求
     */
    public boolean ensureCapacity(BlockTable bt, int requiredTokens) {
        int needed = (requiredTokens + blockSize - 1) / blockSize;
        while (bt.numBlocks() < needed) {
            int id = pool.allocate();
            if (id < 0) {
                if (!evictOneCachedEntry()) {
                    return false; // 池满且无缓存可驱逐
                }
                continue; // 驱逐成功后重试分配
            }
            bt.append(id);
        }
        return true;
    }

    /**
     * 写入第 tokenIdx 个 token 的 K 和 V 到对应物理 block。
     * k/v 长度应等于 dModel。INT8 模式下在线对称量化：
     * scale = absmax/127，q = round(x/scale)（per-token，K/V 各自 scale）。
     */
    public void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        if (int8) {
            blk.kScale[slot] = quantizeRow(k, 0, blk.k8, off);
            blk.vScale[slot] = quantizeRow(v, 0, blk.v8, off);
        } else {
            // 堆外写入：绝对地址 putFloat，不影响 position/limit（并发读安全）
            int byteOff = off * 4;
            for (int d = 0; d < dModel; d++) {
                blk.k.putFloat(byteOff + d * 4, k[d]);
                blk.v.putFloat(byteOff + d * 4, v[d]);
            }
        }
    }

    /** 在线量化一行：返回 scale；dst 写入 round(src/scale)（absmax 为 0 时清零该行） */
    private float quantizeRow(float[] src, int sOff, byte[] dst, int dOff) {
        float amax = 0f;
        for (int i = 0; i < dModel; i++) {
            float a = Math.abs(src[sOff + i]);
            if (a > amax) {
                amax = a;
            }
        }
        if (amax == 0f) {
            for (int i = 0; i < dModel; i++) {
                dst[dOff + i] = 0; // block 复用会有陈旧数据，全零行必须显式清零
            }
            return 0f;
        }
        float scale = amax / 127f;
        float inv = 1f / scale;
        for (int i = 0; i < dModel; i++) {
            dst[dOff + i] = (byte) Math.round(src[sOff + i] * inv);
        }
        return scale;
    }

    /** 读取第 tokenIdx 个 token 的 K（返回拷贝，避免外部篡改；INT8 模式自动反量化） */
    public float[] readK(BlockTable bt, int tokenIdx) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        float[] r = new float[dModel];
        if (int8) {
            dequantizeRow(blk.k8, off, blk.kScale[slot], r);
        } else {
            copyFromOffHeap(blk.k, off * 4, r);
        }
        return r;
    }

    /** 读取第 tokenIdx 个 token 的 V（返回拷贝；INT8 模式自动反量化） */
    public float[] readV(BlockTable bt, int tokenIdx) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        float[] r = new float[dModel];
        if (int8) {
            dequantizeRow(blk.v8, off, blk.vScale[slot], r);
        } else {
            copyFromOffHeap(blk.v, off * 4, r);
        }
        return r;
    }

    /** 堆外段拷贝出 dModel 个 float（readK/readV 共用；测试与非热路径用） */
    private void copyFromOffHeap(java.nio.ByteBuffer buf, int byteOff, float[] dst) {
        for (int d = 0; d < dModel; d++) {
            dst[d] = buf.getFloat(byteOff + d * 4);
        }
    }

    /** 反量化一行：dst[i] = src[i] * scale */
    private void dequantizeRow(byte[] src, int sOff, float scale, float[] dst) {
        for (int i = 0; i < dModel; i++) {
            dst[i] = src[sOff + i] * scale;
        }
    }

    /**
     * 截断 BlockTable 到指定 token 容量：释放完全废弃的尾部 block（引用计数与缓存态
     * 处理同 {@link #free}）。截断点所在的部分块无需清理——KV 槽位由 tokenIdx 唯一
     * 定位，后续写入同位置时自然覆写。投机采样回滚被拒绝草稿的 KV 用。
     */
    public void truncateTo(BlockTable bt, int tokens) {
        int needed = (tokens + blockSize - 1) / blockSize;
        while (bt.numBlocks() > needed) {
            int id = bt.removeLast();
            Long hash = prefixCache.hashOfBlock(id);
            if (pool.decRef(id)) {
                if (hash != null) {
                    prefixCache.markCachedIfFullyUnreferenced(id, pool);
                } else {
                    pool.recycle(id);
                }
            }
        }
    }

    /**
     * 释放某请求一层的 block（按引用计数）。
     * 若 block 被其它请求共享，仅减引用；若已注册前缀缓存，引用归零后
     * 转入缓存态（不归还空闲池），等待 LRU 驱逐或再次被共享命中。
     */
    public void free(BlockTable bt) {
        for (int id : bt.toArray()) {
            Long hash = prefixCache.hashOfBlock(id);
            if (pool.decRef(id)) {
                // 引用归零：已注册的转入缓存态，未注册的直接归还空闲池
                if (hash != null) {
                    prefixCache.markCachedIfFullyUnreferenced(id, pool);
                } else {
                    pool.recycle(id);
                }
            }
        }
        bt.clear();
    }

    // ─── 前缀共享 ───

    /**
     * 尝试为新请求复用已缓存的前缀 block（每层各一张 BlockTable，同步共享）。
     * @param tokens 新请求的完整 token 序列
     * @param bts    每层的 BlockTable（将填入共享的 block id），长度 = 层数
     * @return 命中共享的 token 数（这些 token 无需重算 prefill）
     */
    public int trySharePrefix(int[] tokens, BlockTable[] bts) {
        int shared = 0;
        // 至少保留 1 个 token 不共享：prefill 必须算出最后一个位置的 logits 才能采样首 token
        int nFullBlocks = Math.max(0, tokens.length - 1) / blockSize;
        long parentHash = 0L;
        for (int b = 0; b < nFullBlocks; b++) {
            long fp = prefixCache.chainedHash(parentHash, tokens, b * blockSize, blockSize);
            int[] ids = prefixCache.getShareable(fp, pool);
            if (ids == null) {
                break; // 前缀不再匹配，停止共享
            }
            for (int l = 0; l < bts.length; l++) {
                pool.retain(ids[l]); // 引用 +1（缓存态 block 重新激活）
                bts[l].append(ids[l]);
            }
            prefixCache.activate(ids[0], pool); // 移出 LRU（不再是可驱逐的缓存态）
            parentHash = fp;
            shared += blockSize;
        }
        return shared;
    }

    /** 单层便捷入口（测试/单层场景） */
    public int trySharePrefix(int[] tokens, BlockTable bt) {
        return trySharePrefix(tokens, new BlockTable[]{bt});
    }

    /**
     * 注册某请求已写入的 block，供后续请求共享前缀（每层同步注册）。
     * 应在 prefill 完成后调用。同一链式哈希允许多组 block 并存（内容相同的
     * 不同请求各自注册）：共享命中优先选仍被引用的活跃组，其次选缓存态组。
     * 经 trySharePrefix 共享的请求 prefill 后重复注册同一组 block 时，
     * {@link PrefixCache#put} 按内容去重直接跳过：block 的缓存态保护由首次
     * 注册建立的 blockToHash 映射提供（free 时按 block 反查），无需重复登记；
     * 若不去重，同一组会两次进入可驱逐队列，导致双重 recycle 与跨序列脏数据。
     */
    public void registerPrefix(int[] tokens, BlockTable[] bts) {
        int nFullBlocks = tokens.length / blockSize;
        long parentHash = 0L;
        for (int b = 0; b < nFullBlocks; b++) {
            long fp = prefixCache.chainedHash(parentHash, tokens, b * blockSize, blockSize);
            parentHash = fp;
            int[] ids = new int[bts.length];
            for (int l = 0; l < bts.length; l++) {
                if (bts[l].numBlocks() <= b) {
                    ids = null; // 层间 block 数不一致（异常防御），放弃本 block 起的注册
                    break;
                }
                ids[l] = bts[l].blockIdAt(b);
            }
            if (ids == null) break;
            prefixCache.put(fp, ids);
        }
    }

    /** 单层便捷入口（测试/单层场景） */
    public void registerPrefix(int[] tokens, BlockTable bt) {
        registerPrefix(tokens, new BlockTable[]{bt});
    }

    /** 当前处于缓存态（可被 LRU 驱逐）的 block 数，供观测/测试 */
    public int cachedBlocks() {
        return prefixCache.cachedBlockCount();
    }

    /**
     * 驱逐 LRU 中最久未用的缓存条目，其各层 block 归还空闲池。
     * @return false 表示无缓存可驱逐
     */
    private boolean evictOneCachedEntry() {
        return prefixCache.evictOldest(pool);
    }

    // ─── 供 attention 层遍历的便捷接口 ───

    /**
     * 第 logicalBlockIdx 个逻辑 block 的堆外 K 缓冲（仅 f32 模式）。
     * 布局行优先 [blockSize, dModel]，token t 的第 d 维位于字节偏移 (t*dModel + d) * 4。
     * 热路径经 Matmul.dot/axpy 的堆外重载直读，无需拷回堆上。
     */
    public java.nio.ByteBuffer blockK(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).k;
    }

    /** 第 logicalBlockIdx 个逻辑 block 的堆外 V 缓冲（仅 f32 模式） */
    public java.nio.ByteBuffer blockV(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).v;
    }

    /** 第 logicalBlockIdx 个 block 的量化 K（行优先 [blockSize, dModel]；仅 INT8 模式） */
    public byte[] blockK8(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).k8;
    }

    /** 第 logicalBlockIdx 个 block 的量化 V（仅 INT8 模式） */
    public byte[] blockV8(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).v8;
    }

    /** 第 logicalBlockIdx 个 block 的 K per-token scale [blockSize]（仅 INT8 模式） */
    public float[] blockKScale(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).kScale;
    }

    /** 第 logicalBlockIdx 个 block 的 V per-token scale（仅 INT8 模式） */
    public float[] blockVScale(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).vScale;
    }

    // ─── 前缀缓存（链式哈希 + LRU） ───
    // 对照 vLLM PrefixCachingBlockManager 的简化版：
    // 1. 链式哈希：block b 的指纹由“父 block 指纹 + 本 block tokens”混合而成，
    //    保证相同内容在不同上下文（前缀不同）下指纹不同，避免错误共享。
    // 2. 缓存态 + LRU：引用归零的已注册 block 不直接归还空闲池，而是留在缓存中；
    //    仅当池满且被 LRU 驱逐时才真正回收——内容不会被覆写，共享永远安全。
    // 3. 学习版不校验 block 内容（64-bit 链式哈希碰撞概率可忽略；生产需内容校验）。

    private static final class PrefixCache {
        /** 链式哈希 -> 多组各层物理 block id（内容相同的不同请求各自注册一组） */
        private final Map<Long, Deque<int[]>> hashToBlocks = new HashMap<>();
        /** 物理 block id -> 其所属条目的链式哈希（释放时判断是否转缓存态） */
        private final Map<Integer, Long> blockToHash = new HashMap<>();
        /** 可驱逐条目（引用全部归零的条目组），按进入缓存的先后排序（头部最旧） */
        private final LinkedHashMap<EntryRef, Boolean> evictable = new LinkedHashMap<>();

        /**
         * 可驱逐条目定位：指纹 + 该指纹下的具体条目组。
         * record 默认按引用比较数组，同一组多次入队会产生重复条目 → 双重 recycle；
         * 故重写 equals/hashCode 按内容比较，保证 evictable 中同组唯一。
         */
        private record EntryRef(long fp, int[] ids) {
            @Override
            public boolean equals(Object o) {
                return o instanceof EntryRef other
                        && fp == other.fp && Arrays.equals(ids, other.ids);
            }

            @Override
            public int hashCode() {
                return Long.hashCode(fp) * 31 + Arrays.hashCode(ids);
            }
        }

        /** 链式哈希：parent 为前一 block 的指纹（首 block 为 0） */
        long chainedHash(long parent, int[] tokens, int start, int len) {
            long h = parent ^ 0x9E3779B97F4A7C15L;
            for (int i = 0; i < len; i++) {
                h = h * 131L + tokens[start + i];
            }
            // 末段混合：让 parent 的位分布充分扩散，降低相邻 block 指纹相关性
            h ^= (h >>> 33);
            h *= 0xFF51AFD7ED558CCDL;
            h ^= (h >>> 33);
            return h;
        }

        void put(long fp, int[] blockIds) {
            Deque<int[]> entries = hashToBlocks.computeIfAbsent(fp, k -> new ArrayDeque<>());
            for (int[] existing : entries) {
                if (Arrays.equals(existing, blockIds)) {
                    return; // 内容去重：共享者重复注册同一组 block，追加会让 free 时
                    // 同一组两次进入可驱逐队列 → 双重 recycle → 同一 block 被分配给两个序列
                }
            }
            entries.addLast(blockIds);
            for (int id : blockIds) {
                blockToHash.put(id, fp);
            }
        }

        /**
         * 取可共享的条目组：优先仍被引用的活跃组（其 block 必在池中），
         * 其次取缓存态组（将重新激活）；无可共享时返回 null。
         */
        int[] getShareable(long fp, BlockPool pool) {
            Deque<int[]> entries = hashToBlocks.get(fp);
            if (entries == null) {
                return null;
            }
            int[] cachedFallback = null;
            for (int[] ids : entries) {
                if (anyReferenced(ids, pool)) {
                    return ids;
                }
                if (cachedFallback == null) {
                    cachedFallback = ids;
                }
            }
            return cachedFallback;
        }

        private boolean anyReferenced(int[] ids, BlockPool pool) {
            for (int id : ids) {
                if (pool.get(id).refCount() > 0) {
                    return true;
                }
            }
            return false;
        }

        Long hashOfBlock(int blockId) {
            return blockToHash.get(blockId);
        }

        /** 共享命中：含该 block 的条目组重新激活，移出可驱逐队列 */
        void activate(int blockId, BlockPool pool) {
            Long fp = blockToHash.get(blockId);
            if (fp == null) {
                return;
            }
            evictable.keySet().removeIf(ref -> ref.fp() == fp
                    && contains(ref.ids(), blockId)
                    && anyReferenced(ref.ids(), pool));
        }

        private static boolean contains(int[] ids, int blockId) {
            for (int id : ids) {
                if (id == blockId) {
                    return true;
                }
            }
            return false;
        }

        /** 某 block 引用归零时调用：含它的条目组若所有层 block 都无引用，转入可驱逐队列 */
        void markCachedIfFullyUnreferenced(int blockId, BlockPool pool) {
            Long fp = blockToHash.get(blockId);
            if (fp == null) {
                return;
            }
            Deque<int[]> entries = hashToBlocks.get(fp);
            if (entries == null) {
                return;
            }
            for (int[] ids : entries) {
                if (contains(ids, blockId) && !anyReferenced(ids, pool)) {
                    evictable.put(new EntryRef(fp, ids), Boolean.TRUE);
                }
            }
        }

        /** 驱逐最旧的可驱逐条目组，其 block 归还空闲池；@return false 表示无可驱逐 */
        boolean evictOldest(BlockPool pool) {
            Iterator<EntryRef> it = evictable.keySet().iterator();
            if (!it.hasNext()) {
                return false;
            }
            EntryRef ref = it.next();
            it.remove();
            Deque<int[]> entries = hashToBlocks.get(ref.fp());
            if (entries != null) {
                entries.removeIf(ids -> ids == ref.ids());
                if (entries.isEmpty()) {
                    hashToBlocks.remove(ref.fp());
                }
            }
            for (int id : ref.ids()) {
                blockToHash.remove(id);
                pool.recycle(id); // 驱逐后内容可被覆写
            }
            return true;
        }

        /** 缓存态 block 总数（各层分别计数） */
        int cachedBlockCount() {
            int n = 0;
            for (EntryRef ref : evictable.keySet()) {
                n += ref.ids().length;
            }
            return n;
        }
    }
}
