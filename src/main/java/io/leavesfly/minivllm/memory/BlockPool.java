package io.leavesfly.minivllm.memory;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * BlockPool —— PagedAttention 的物理内存池。
 *
 * 学习要点（对照 vLLM PagedAttention）：
 * 1. GPU 显存被切分成固定大小的 block，每个 block 能装 {@code blockSize} 个 token 的 K 和 V。
 * 2. block 之间不需要连续，靠 BlockTable 记录逻辑顺序——这正是"分页"的精髓，
 *    类比操作系统把进程的虚拟页映射到散落的物理页框。
 * 3. 引用计数(refCount)实现共享：多个请求指向同一 block 时，只有最后一个释放才真正归还。
 *    这是前缀共享(prefix sharing)和 beam search 共享的底层基础。
 *
 * 本学习版用 Java 堆数组模拟"显存"，真实 vLLM 用的是 GPU 显存指针。
 */
public final class BlockPool {

    /** 单个 block 能容纳的 token 数（vLLM 默认 16） */
    private final int blockSize;
    /** 模型隐藏层维度（每个 token 的 K/V 向量长度） */
    private final int dModel;
    /** block 总数（= 模拟显存容量 / block 大小） */
    private final int numBlocks;

    private final KVBlock[] blocks;
    /** 空闲 block id 队列 */
    private final Deque<Integer> freeList;
    private int usedBlocks = 0;

    public BlockPool(int numBlocks, int blockSize, int dModel) {
        this.numBlocks = numBlocks;
        this.blockSize = blockSize;
        this.dModel = dModel;
        this.blocks = new KVBlock[numBlocks];
        this.freeList = new ArrayDeque<>(numBlocks);
        for (int i = 0; i < numBlocks; i++) {
            freeList.add(i); // block 数组懒分配：首次 allocate 时才创建，避免启动即占满堆
        }
    }

    public int blockSize() {
        return blockSize;
    }

    public int dModel() {
        return dModel;
    }

    public int numBlocks() {
        return numBlocks;
    }

    /**
     * 分配一个新 block（首次分配时才创建底层数组——懒分配）。
     * @return block id；若池满返回 -1（调度器应据此暂停接收新请求）
     */
    public int allocate() {
        Integer id = freeList.pollFirst();
        if (id == null) {
            return -1; // 显存不足，由调度器做 preemption/等待
        }
        if (blocks[id] == null) {
            blocks[id] = new KVBlock(id, blockSize, dModel);
        }
        blocks[id].refCount = 1;
        usedBlocks++;
        return id;
    }

    /** 增加引用（用于共享 block） */
    public void retain(int blockId) {
        blocks[blockId].refCount++;
    }

    /**
     * 释放一次引用；引用归零时归还到空闲池。
     */
    public void release(int blockId) {
        KVBlock b = blocks[blockId];
        b.refCount--;
        if (b.refCount == 0) {
            freeList.addFirst(blockId); // LIFO，利于缓存局部性
            usedBlocks--;
        } else if (b.refCount < 0) {
            throw new IllegalStateException("block " + blockId + " 引用计数下溢");
        }
    }

    /** 获取 block 对象（attention 层据此读写 K/V） */
    public KVBlock get(int blockId) {
        return blocks[blockId];
    }

    /** 当前已用 block 数 */
    public int usedBlocks() {
        return usedBlocks;
    }

    /** 剩余可用 block 数 */
    public int freeBlocks() {
        return freeList.size();
    }

    /**
     * 一个物理 block，承载 blockSize 个 token 的 K 与 V。
     * 数据布局：行优先 [blockSize, dModel]，token t 的 K 起始偏移 = t * dModel。
     *
     * 设计说明：k/v 数组故意暴露为 public final，因为 attention 层在 decode 热路径中
     * 需要直接读写这些数组（避免方法调用开销）。这是性能与封装的有意识权衡。
     */
    public static final class KVBlock {
        public final int id;
        public final float[] k; // [blockSize * dModel]
        public final float[] v; // [blockSize * dModel]
        private int refCount = 0;

        KVBlock(int id, int blockSize, int dModel) {
            this.id = id;
            this.k = new float[blockSize * dModel];
            this.v = new float[blockSize * dModel];
        }

        public int refCount() {
            return refCount;
        }
    }
}
