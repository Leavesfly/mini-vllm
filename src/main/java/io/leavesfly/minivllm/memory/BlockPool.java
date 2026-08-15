package io.leavesfly.minivllm.memory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
 * 本学习版用堆外内存（direct ByteBuffer）模拟"显存"：f32 KV 常驻堆外，
 * 不占 Java 堆（权重与激活留在堆上），大 KV 池不再挤压 -Xmx 预算，
 * 对应真实 vLLM 的 GPU 显存指针；INT8 量化路径数据量小，仍用堆上 byte[]。
 */
public final class BlockPool {

    /** 单个 block 能容纳的 token 数（vLLM 默认 16） */
    private final int blockSize;
    /** 模型隐藏层维度（每个 token 的 K/V 向量长度） */
    private final int dModel;
    /** block 总数（= 模拟显存容量 / block 大小） */
    private final int numBlocks;
    /** KV cache 是否 INT8 量化存储（对照 vLLM --kv-cache-dtype int8） */
    private final boolean int8;

    private final KVBlock[] blocks;
    /** 空闲 block id 队列 */
    private final Deque<Integer> freeList;
    private int usedBlocks = 0;

    public BlockPool(int numBlocks, int blockSize, int dModel) {
        this(numBlocks, blockSize, dModel, false);
    }

    /** @param int8 true 时 block 内 K/V 以 byte[] + per-token scale 存储，容量减为 f32 的 1/4 */
    public BlockPool(int numBlocks, int blockSize, int dModel, boolean int8) {
        this.numBlocks = numBlocks;
        this.blockSize = blockSize;
        this.dModel = dModel;
        this.int8 = int8;
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

    /** KV cache 是否 INT8 量化存储 */
    public boolean isInt8() {
        return int8;
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
            blocks[id] = new KVBlock(id, blockSize, dModel, int8);
        }
        blocks[id].refCount = 1;
        usedBlocks++;
        return id;
    }

    /** 增加引用（用于共享 block；缓存态 block 被命中时由此重新激活） */
    public void retain(int blockId) {
        KVBlock b = blocks[blockId];
        if (b.refCount == 0) {
            usedBlocks++; // 从缓存态复活，重新计为已用
        }
        b.refCount++;
    }

    /**
     * 释放一次引用；引用归零时归还到空闲池（普通路径）。
     * 前缀缓存场景请用 {@link #decRef}：引用归零的 block 可能要转入缓存态而非直接归还。
     */
    public void release(int blockId) {
        if (decRef(blockId)) {
            recycle(blockId);
        }
    }

    /**
     * 仅减引用，引用归零时不自动归还空闲池，由调用方决定去向
     * （归还空闲池 recycle，或转入前缀缓存态）。
     * @return true 表示引用计数归零
     */
    public boolean decRef(int blockId) {
        KVBlock b = blocks[blockId];
        b.refCount--;
        if (b.refCount == 0) {
            usedBlocks--;
            return true;
        } else if (b.refCount < 0) {
            throw new IllegalStateException("block " + blockId + " 引用计数下溢");
        }
        return false;
    }

    /** 把引用已归零的 block 归还空闲池（LIFO，利于缓存局部性） */
    public void recycle(int blockId) {
        freeList.addFirst(blockId);
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
     * 数据布局：行优先 [blockSize, dModel]，token t 的 K 起始字节偏移 = t * dModel * 4。
     *
     * 设计说明：存储缓冲故意暴露为 public final，因为 attention 层在 decode 热路径中
     * 需要直接读写（避免方法调用开销）。这是性能与封装的有意识权衡。
     *
     * 堆外内存（P5）：f32 模式的 K/V 各为 direct ByteBuffer——KV 池（可达 GB 级）
     * 不占 Java 堆，把 -Xmx 预算留给权重与激活；GC 也不感知堆外数据。
     * 分配在 block 首次使用时发生（懒分配），池内复用不释放（Cleaner 兜底回收）。
     * 读取走 Matmul 的堆外 SIMD 内核（FloatVector.fromByteBuffer），与堆上等速。
     *
     * INT8 量化存储（对照 vLLM kv_cache_dtype=int8）：K/V 各用 byte[] 存量化值，
     * kScale/vScale 为 per-token 对称量化缩放（scale = absmax/127）。
     * 容量从 f32 的 2×4B/token 降到 2×1B/token + 2 scale，带宽减半直接提速 decode。
     * f32 模式下 k8/v8/scale 为 null，反之亦然（互斥，避免双倍内存）。
     */
    public static final class KVBlock {
        public final int id;
        public final ByteBuffer k; // f32 模式：direct，[blockSize * dModel * 4] 字节；int8 模式为 null
        public final ByteBuffer v;
        public final byte[] k8; // int8 模式：[blockSize * dModel] 量化值；f32 模式为 null
        public final byte[] v8;
        public final float[] kScale; // int8 模式：per-token scale [blockSize]
        public final float[] vScale;
        private int refCount = 0;

        KVBlock(int id, int blockSize, int dModel, boolean int8) {
            this.id = id;
            if (int8) {
                this.k = null;
                this.v = null;
                this.k8 = new byte[blockSize * dModel];
                this.v8 = new byte[blockSize * dModel];
                this.kScale = new float[blockSize];
                this.vScale = new float[blockSize];
            } else {
                // direct + 本机字节序：FloatVector.fromByteBuffer 零换序加载的前提
                this.k = ByteBuffer.allocateDirect(blockSize * dModel * 4).order(ByteOrder.nativeOrder());
                this.v = ByteBuffer.allocateDirect(blockSize * dModel * 4).order(ByteOrder.nativeOrder());
                this.k8 = null;
                this.v8 = null;
                this.kScale = null;
                this.vScale = null;
            }
        }

        public int refCount() {
            return refCount;
        }
    }
}
