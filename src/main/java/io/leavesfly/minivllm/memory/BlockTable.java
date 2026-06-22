package io.leavesfly.minivllm.memory;

import java.util.ArrayList;
import java.util.List;

/**
 * BlockTable —— 单个请求的逻辑→物理 block 映射。
 *
 * 学习要点（对照 vLLM）：
 * 1. 每个请求拥有一张 BlockTable，记录其 KV cache 逻辑顺序对应的物理 block id 序列。
 * 2. block 在物理池中可以散落任意位置，BlockTable 把它们"串"成逻辑连续的 KV cache。
 * 3. 逻辑 token 下标 → 物理位置：
 *      logicalBlockIdx = tokenIdx / blockSize
 *      slotInBlock     = tokenIdx % blockSize
 *      physicalBlockId = blockTable.get(logicalBlockIdx)
 *    这正是操作系统中"页表"的思想：虚拟页号 → 物理页框号。
 */
public final class BlockTable {

    /** 按 KV cache 逻辑顺序排列的物理 block id */
    private final List<Integer> blockIds = new ArrayList<>();

    /** 追加一个物理 block id 到末尾 */
    public void append(int blockId) {
        blockIds.add(blockId);
    }

    /** 取第 logicalBlockIdx 个逻辑 block 对应的物理 block id */
    public int blockIdAt(int logicalBlockIdx) {
        return blockIds.get(logicalBlockIdx);
    }

    /** 已分配的逻辑 block 数 */
    public int numBlocks() {
        return blockIds.size();
    }

    /** 该 BlockTable 当前最多能承载的 token 数 */
    public int capacity(int blockSize) {
        return blockIds.size() * blockSize;
    }

    /** 返回所有 block id 的拷贝（用于释放或共享统计） */
    public int[] toArray() {
        int[] r = new int[blockIds.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = blockIds.get(i);
        }
        return r;
    }

    /** 清空（不释放物理 block，释放由 KVCacheManager 负责） */
    public void clear() {
        blockIds.clear();
    }

    @Override
    public String toString() {
        return "BlockTable" + blockIds;
    }
}
