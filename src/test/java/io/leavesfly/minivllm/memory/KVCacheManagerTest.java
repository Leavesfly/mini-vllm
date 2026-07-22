package io.leavesfly.minivllm.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * KVCacheManager 单元测试 —— 验证 PagedAttention 的分配、读写与释放。
 */
class KVCacheManagerTest {

    private static final int BLOCK_SIZE = 4;
    private static final int D_MODEL = 8;
    private static final int NUM_BLOCKS = 16;

    private KVCacheManager createManager() {
        return new KVCacheManager(NUM_BLOCKS, BLOCK_SIZE, D_MODEL);
    }

    @Test
    void ensureCapacityAllocatesNeededBlocks() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        // 需要 5 个 token → ceil(5/4) = 2 blocks
        assertTrue(mgr.ensureCapacity(bt, 5));
        assertEquals(2, bt.numBlocks());
    }

    @Test
    void ensureCapacityNoExtraAllocation() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 4); // 恰好 1 block
        assertEquals(1, bt.numBlocks());
        // 再次请求同样容量不多分配
        mgr.ensureCapacity(bt, 4);
        assertEquals(1, bt.numBlocks());
    }

    @Test
    void ensureCapacityReturnsFalseWhenFull() {
        // 只有 2 个 block，每个装 4 token
        KVCacheManager mgr = new KVCacheManager(2, BLOCK_SIZE, D_MODEL);
        BlockTable bt = new BlockTable();
        // 请求 9 个 token → 需要 3 blocks，但只有 2
        assertFalse(mgr.ensureCapacity(bt, 9));
    }

    @Test
    void writeAndReadKV() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 3);

        // 写入 token 0 的 KV
        float[] k = new float[D_MODEL];
        float[] v = new float[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            k[i] = i + 1;
            v[i] = (i + 1) * 10;
        }
        mgr.writeKV(bt, 0, k, v);

        // 读取并验证
        float[] readK = mgr.readK(bt, 0);
        float[] readV = mgr.readV(bt, 0);
        assertArrayEquals(k, readK);
        assertArrayEquals(v, readV);
    }

    @Test
    void writeKVAcrossBlocks() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 6); // 需要 2 blocks (blockSize=4)

        // 写入 token 5（在第 2 个 block 的 slot 1）
        float[] k = new float[D_MODEL];
        float[] v = new float[D_MODEL];
        k[0] = 42f;
        v[0] = 99f;
        mgr.writeKV(bt, 5, k, v);

        float[] readK = mgr.readK(bt, 5);
        float[] readV = mgr.readV(bt, 5);
        assertEquals(42f, readK[0]);
        assertEquals(99f, readV[0]);
    }

    @Test
    void freeReleasesAllBlocks() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 8); // 2 blocks
        assertEquals(NUM_BLOCKS - 2, mgr.freeBlocks());

        mgr.free(bt);
        assertEquals(0, bt.numBlocks());
        assertEquals(NUM_BLOCKS, mgr.freeBlocks());
    }

    @Test
    void prefixSharingWorks() {
        KVCacheManager mgr = createManager();
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8}; // 2 full blocks

        // 第一个请求正常 prefill
        BlockTable bt1 = new BlockTable();
        mgr.ensureCapacity(bt1, 8);
        // 写入 KV（简化）
        for (int i = 0; i < 8; i++) {
            mgr.writeKV(bt1, i, new float[D_MODEL], new float[D_MODEL]);
        }
        // 注册前缀
        mgr.registerPrefix(tokens, bt1);

        // 第二个请求尝试共享
        BlockTable bt2 = new BlockTable();
        int shared = mgr.trySharePrefix(tokens, bt2);
        assertEquals(8, shared); // 共享了全部 8 个 token（2 blocks）
        assertEquals(2, bt2.numBlocks());

        // bt2 的 block 和 bt1 指向相同物理块
        assertEquals(bt1.blockIdAt(0), bt2.blockIdAt(0));
        assertEquals(bt1.blockIdAt(1), bt2.blockIdAt(1));
    }

    @Test
    void blockKAndBlockVAccess() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 4);

        float[] blockK = mgr.blockK(bt, 0);
        float[] blockV = mgr.blockV(bt, 0);
        assertEquals(BLOCK_SIZE * D_MODEL, blockK.length);
        assertEquals(BLOCK_SIZE * D_MODEL, blockV.length);
    }
}
