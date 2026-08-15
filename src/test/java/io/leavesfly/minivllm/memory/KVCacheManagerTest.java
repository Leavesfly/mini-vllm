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

        // 第二个请求尝试共享：至少保留 1 个 token 不共享（prefill 需要它算首 token logits），
        // 故 8 token / blockSize=4 最多共享 1 个完整 block
        BlockTable bt2 = new BlockTable();
        int shared = mgr.trySharePrefix(tokens, bt2);
        assertEquals(4, shared);
        assertEquals(1, bt2.numBlocks());

        // bt2 的 block 和 bt1 指向相同物理块
        assertEquals(bt1.blockIdAt(0), bt2.blockIdAt(0));
    }

    @Test
    void chainedHashPreventsSameContentDifferentContext() {
        KVCacheManager mgr = createManager();
        // 请求 A：[1,2,3,4 | 5,6,7,8]；请求 B：[9,9,9,9 | 5,6,7,8]
        // 第二个 block 内容相同但前缀不同，链式哈希下不应被共享
        int[] tokensA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        int[] tokensB = {9, 9, 9, 9, 5, 6, 7, 8, 9};

        BlockTable btA = new BlockTable();
        mgr.ensureCapacity(btA, 9);
        mgr.registerPrefix(tokensA, btA);

        BlockTable btB = new BlockTable();
        int shared = mgr.trySharePrefix(tokensB, btB);
        assertEquals(0, shared); // 首 block 不同，链式哈希下后续 block 也不可能命中
        assertEquals(0, btB.numBlocks());
    }

    @Test
    void freedRegisteredBlocksStayCachedNotReused() {
        KVCacheManager mgr = new KVCacheManager(4, BLOCK_SIZE, D_MODEL);
        int[] tokens = {1, 2, 3, 4, 5}; // 1 个完整 block + 尾部

        // 请求 1：分配 2 block、注册、释放
        BlockTable bt1 = new BlockTable();
        mgr.ensureCapacity(bt1, 5);
        mgr.registerPrefix(tokens, bt1);
        int cachedId = bt1.blockIdAt(0);
        mgr.free(bt1);

        // 已注册的 block 进入缓存态：不归还空闲池，也不计入 freeBlocks
        assertEquals(1, mgr.cachedBlocks());
        assertEquals(3, mgr.freeBlocks()); // 4 - 1 缓存

        // 新请求分配到的是别的 block，缓存块内容不会被覆写
        BlockTable bt2 = new BlockTable();
        mgr.ensureCapacity(bt2, 4);
        assertNotEquals(cachedId, bt2.blockIdAt(0));
    }

    @Test
    void lruEvictionReclaimsCachedBlocksWhenPoolFull() {
        KVCacheManager mgr = new KVCacheManager(2, BLOCK_SIZE, D_MODEL);
        int[] tokens = {1, 2, 3, 4, 5};

        // 请求 1：占 2 block，注册后释放 -> 1 缓存 + 1 空闲
        BlockTable bt1 = new BlockTable();
        mgr.ensureCapacity(bt1, 5);
        mgr.registerPrefix(tokens, bt1);
        mgr.free(bt1);
        assertEquals(1, mgr.freeBlocks());

        // 请求 2 需要 2 block：先拿空闲的 1 个，再驱逐缓存条目补齐
        BlockTable bt2 = new BlockTable();
        assertTrue(mgr.ensureCapacity(bt2, 5));
        assertEquals(2, bt2.numBlocks());
        assertEquals(0, mgr.cachedBlocks());
        assertEquals(0, mgr.freeBlocks());

        // 被驱逐的前缀不再可共享
        BlockTable bt3 = new BlockTable();
        assertEquals(0, mgr.trySharePrefix(tokens, bt3));
    }

    @Test
    void sharedBlockSurvivesFirstFreeUntilLastRefDropped() {
        KVCacheManager mgr = createManager();
        int[] tokens = {1, 2, 3, 4, 5};

        BlockTable bt1 = new BlockTable();
        mgr.ensureCapacity(bt1, 5);
        mgr.registerPrefix(tokens, bt1);

        // 请求 2 共享首 block
        BlockTable bt2 = new BlockTable();
        assertEquals(BLOCK_SIZE, mgr.trySharePrefix(tokens, bt2));

        // 请求 1 释放：共享块仍被 bt2 引用，不进缓存态也不回收
        mgr.free(bt1);
        assertEquals(1, mgr.pool().get(bt2.blockIdAt(0)).refCount());

        // 请求 2 也释放：条目全部无引用，转入可驱逐缓存
        mgr.free(bt2);
        assertEquals(1, mgr.cachedBlocks());

        // 请求 3 仍能命中共享
        BlockTable bt3 = new BlockTable();
        assertEquals(BLOCK_SIZE, mgr.trySharePrefix(tokens, bt3));
        assertEquals(0, mgr.cachedBlocks()); // 命中后重新激活
    }

    @Test
    void duplicateRegisterDoesNotDoubleRecycle() {
        // 回归：共享者 prefill 后重复注册同一组 block 必须被去重，
        // 否则同一组两次进入可驱逐队列 → 双重 recycle → 在用 block 被再分配
        KVCacheManager mgr = new KVCacheManager(3, BLOCK_SIZE, D_MODEL);
        int[] tokens = {1, 2, 3, 4, 5};

        // 请求 A：prefill + 注册
        BlockTable btA = new BlockTable();
        assertTrue(mgr.ensureCapacity(btA, 5));
        mgr.registerPrefix(tokens, btA);

        // 请求 B：共享 A 的首 block，prefill 后再次注册同一前缀
        BlockTable btB = new BlockTable();
        assertEquals(BLOCK_SIZE, mgr.trySharePrefix(tokens, btB));
        assertTrue(mgr.ensureCapacity(btB, 5));
        mgr.registerPrefix(tokens, btB);

        mgr.free(btA);
        mgr.free(btB);
        assertEquals(1, mgr.cachedBlocks()); // 缓存态条目只应有一组

        // 吃满整个池：2 空闲 + 1 驱逐
        BlockTable btC = new BlockTable();
        assertTrue(mgr.ensureCapacity(btC, BLOCK_SIZE * 3));
        assertEquals(0, mgr.freeBlocks());
        assertEquals(0, mgr.cachedBlocks());

        // 若存在双重 recycle，这里会错误地再分出一个在用 block
        BlockTable btD = new BlockTable();
        assertFalse(mgr.ensureCapacity(btD, BLOCK_SIZE));
        assertEquals(0, btD.numBlocks());
    }

    @Test
    void blockKAndBlockVAccess() {
        KVCacheManager mgr = createManager();
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 4);

        java.nio.ByteBuffer blockK = mgr.blockK(bt, 0);
        java.nio.ByteBuffer blockV = mgr.blockV(bt, 0);
        assertEquals(BLOCK_SIZE * D_MODEL * 4, blockK.capacity());
        assertEquals(BLOCK_SIZE * D_MODEL * 4, blockV.capacity());
        assertTrue(blockK.isDirect(), "f32 KV 应堆外存储");
    }

    // ─── INT8 量化存储 ───

    @Test
    void defaultDtypeIsF32() {
        assertFalse(createManager().isInt8());
    }

    @Test
    void int8QuantizeRoundTripErrorIsBounded() {
        // 对称量化往返误差：每元素误差 ≤ scale/2 = absmax/254
        KVCacheManager mgr = new KVCacheManager(NUM_BLOCKS, BLOCK_SIZE, D_MODEL, true);
        assertTrue(mgr.isInt8());
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 3);

        java.util.Random rnd = new java.util.Random(42L);
        float[] k = new float[D_MODEL];
        float[] v = new float[D_MODEL];
        float kMax = 0f, vMax = 0f;
        for (int i = 0; i < D_MODEL; i++) {
            k[i] = (float) rnd.nextGaussian();
            v[i] = (float) (rnd.nextGaussian() * 3.0);
            kMax = Math.max(kMax, Math.abs(k[i]));
            vMax = Math.max(vMax, Math.abs(v[i]));
        }
        mgr.writeKV(bt, 0, k, v);

        float[] rk = mgr.readK(bt, 0);
        float[] rv = mgr.readV(bt, 0);
        for (int i = 0; i < D_MODEL; i++) {
            assertEquals(k[i], rk[i], kMax / 254f + 1e-6f, "K 量化往返误差超限 i=" + i);
            assertEquals(v[i], rv[i], vMax / 254f + 1e-6f, "V 量化往返误差超限 i=" + i);
        }
    }

    @Test
    void int8ZeroAndOverwriteRowsReadBackCorrectly() {
        // 全零行（scale=0）与 block 复用覆写：读回必须精确为写入值
        KVCacheManager mgr = new KVCacheManager(NUM_BLOCKS, BLOCK_SIZE, D_MODEL, true);
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 1);

        // 先写非零行，再用全零行覆写同一 slot（验证陈旧字节被清零）
        float[] nz = new float[D_MODEL];
        java.util.Arrays.fill(nz, 1.5f);
        mgr.writeKV(bt, 0, nz, nz);
        float[] zero = new float[D_MODEL];
        mgr.writeKV(bt, 0, zero, zero);
        assertArrayEquals(zero, mgr.readK(bt, 0));
        assertArrayEquals(zero, mgr.readV(bt, 0));
    }

    @Test
    void int8QuantizedBlockArraysExposed() {
        // 量化域原语：blockK8/blockV8/blockKScale/blockVScale 供 attention 热路径直读
        KVCacheManager mgr = new KVCacheManager(NUM_BLOCKS, BLOCK_SIZE, D_MODEL, true);
        BlockTable bt = new BlockTable();
        mgr.ensureCapacity(bt, 2);

        float[] k = new float[D_MODEL];
        float[] v = new float[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            k[i] = i - D_MODEL / 2f;
            v[i] = (i - D_MODEL / 2f) * 0.5f;
        }
        mgr.writeKV(bt, 1, k, v); // slot 1

        byte[] k8 = mgr.blockK8(bt, 0);
        byte[] v8 = mgr.blockV8(bt, 0);
        float[] kScale = mgr.blockKScale(bt, 0);
        float[] vScale = mgr.blockVScale(bt, 0);
        assertEquals(BLOCK_SIZE * D_MODEL, k8.length);
        assertEquals(BLOCK_SIZE * D_MODEL, v8.length);
        assertEquals(BLOCK_SIZE, kScale.length);
        assertTrue(kScale[1] > 0f);
        // 量化域反量化与 readK 一致：x ≈ q * scale
        float[] rk = mgr.readK(bt, 1);
        for (int i = 0; i < D_MODEL; i++) {
            assertEquals(rk[i], k8[1 * D_MODEL + i] * kScale[1], 1e-6f);
            assertEquals(rk[i], k[i], kScale[1] / 2f + 1e-6f);
        }
        assertTrue(vScale[1] > 0f);
        assertEquals(v8[1 * D_MODEL] * vScale[1], mgr.readV(bt, 1)[0], 1e-6f);
    }
}
