package io.leavesfly.minivllm.memory;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * KVCacheManager —— PagedAttention 内存管理总指挥。
 *
 * 职责：
 * 1. 按需分配 block（ensureCapacity）：token 数增长时才申请新 block，杜绝过度预留。
 * 2. 读写 KV：把每个 token 的 K/V 写入正确的物理 block 槽位，供 attention 读取。
 * 3. 释放：请求结束时按引用计数释放所有 block（共享 block 不会立即回收）。
 * 4. 前缀共享（PrefixCache）：多个请求若含相同 token 前缀，复用已缓存的 block，省算省存。
 *
 * 学习要点：这里没有一次性预留整段显存，而是“用到才给一块”，因此既无内部碎片
 * （block 等大，任意空闲块都能用），也几乎无外部碎片。这正是 vLLM 高吞吐的根基。
 */
public final class KVCacheManager {

    private final BlockPool pool;
    private final int blockSize;
    private final int dModel;

    /** 前缀缓存：block 内容指纹 -> 物理块 id，用于跨请求共享相同前缀 */
    private final PrefixCache prefixCache = new PrefixCache();

    public KVCacheManager(int numBlocks, int blockSize, int dModel) {
        if (numBlocks <= 0) throw new IllegalArgumentException("numBlocks 必须 > 0");
        if (blockSize <= 0) throw new IllegalArgumentException("blockSize 必须 > 0");
        if (dModel <= 0) throw new IllegalArgumentException("dModel 必须 > 0");
        this.pool = new BlockPool(numBlocks, blockSize, dModel);
        this.blockSize = blockSize;
        this.dModel = dModel;
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

    /** 剩余可用 block 数（便捷委托） */
    public int freeBlocks() {
        return pool.freeBlocks();
    }

    /**
     * 确保某请求的 BlockTable 至少能承载 requiredTokens 个 token。
     * 只在容量不足时申请新 block（按需分配）。
     * @return false 表示显存不足，调度器应阻塞新请求
     */
    public boolean ensureCapacity(BlockTable bt, int requiredTokens) {
        int needed = (requiredTokens + blockSize - 1) / blockSize;
        while (bt.numBlocks() < needed) {
            int id = pool.allocate();
            if (id < 0) {
                return false; // 池满
            }
            bt.append(id);
        }
        return true;
    }

    /**
     * 写入第 tokenIdx 个 token 的 K 和 V 到对应物理 block。
     * k/v 长度应等于 dModel。
     */
    public void writeKV(BlockTable bt, int tokenIdx, float[] k, float[] v) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        System.arraycopy(k, 0, blk.k, off, dModel);
        System.arraycopy(v, 0, blk.v, off, dModel);
    }

    /** 读取第 tokenIdx 个 token 的 K（返回拷贝，避免外部篡改） */
    public float[] readK(BlockTable bt, int tokenIdx) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        float[] r = new float[dModel];
        System.arraycopy(blk.k, off, r, 0, dModel);
        return r;
    }

    /** 读取第 tokenIdx 个 token 的 V（返回拷贝） */
    public float[] readV(BlockTable bt, int tokenIdx) {
        int logicalBlock = tokenIdx / blockSize;
        int slot = tokenIdx % blockSize;
        BlockPool.KVBlock blk = pool.get(bt.blockIdAt(logicalBlock));
        int off = slot * dModel;
        float[] r = new float[dModel];
        System.arraycopy(blk.v, off, r, 0, dModel);
        return r;
    }

    /**
     * 释放某请求的全部 block（按引用计数）。
     * 若 block 被其它请求共享，仅减引用，不真正回收。
     */
    public void free(BlockTable bt) {
        for (int id : bt.toArray()) {
            pool.release(id);
        }
        bt.clear();
    }

    // ─── 前缀共享 ───

    /**
     * 尝试为新请求复用已缓存的前缀 block。
     * @param tokens 新请求的完整 token 序列
     * @param bt     新请求的 BlockTable（将填入共享的 block id）
     * @return 命中共享的 token 数（这些 token 无需重算 prefill）
     */
    public int trySharePrefix(int[] tokens, BlockTable bt) {
        int shared = 0;
        int nFullBlocks = tokens.length / blockSize;
        for (int b = 0; b < nFullBlocks; b++) {
            long fp = prefixCache.fingerprint(tokens, b * blockSize, blockSize);
            Integer id = prefixCache.get(fp);
            if (id == null) {
                break; // 前缀不再匹配，停止共享
            }
            pool.retain(id); // 引用 +1
            bt.append(id);
            shared += blockSize;
        }
        return shared;
    }

    /**
     * 注册某请求已写入的 block，供后续请求共享前缀。
     * 应在 prefill 完成后调用。
     */
    public void registerPrefix(int[] tokens, BlockTable bt) {
        int nFullBlocks = Math.min(tokens.length / blockSize, bt.numBlocks());
        for (int b = 0; b < nFullBlocks; b++) {
            long fp = prefixCache.fingerprint(tokens, b * blockSize, blockSize);
            prefixCache.put(fp, bt.blockIdAt(b));
        }
    }

    // ─── 供 attention 层遍历的便捷接口 ───

    /** 第 logicalBlockIdx 个逻辑 block 的 K 数组（行优先 [blockSize, dModel]） */
    public float[] blockK(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).k;
    }

    /** 第 logicalBlockIdx 个逻辑 block 的 V 数组 */
    public float[] blockV(BlockTable bt, int logicalBlockIdx) {
        return pool.get(bt.blockIdAt(logicalBlockIdx)).v;
    }

    // ─── 简化版前缀缓存 ───
    // 真实 vLLM 用 RadixAttention 树做精确前缀匹配；这里用 block 级哈希映射，
    // 学习项目可接受极小概率哈希冲突（注释提醒：生产环境需校验内容）。

    private static final long HASH_MULTIPLIER = 131L;

    private static final class PrefixCache {
        private final Map<Long, Integer> map = new HashMap<>();

        long fingerprint(int[] tokens, int start, int len) {
            long h = 0L;
            for (int i = 0; i < len; i++) {
                h = h * HASH_MULTIPLIER + tokens[start + i];
            }
            return h;
        }

        Integer get(long fp) {
            return map.get(fp);
        }

        void put(long fp, int blockId) {
            map.putIfAbsent(fp, blockId);
        }
    }
}
