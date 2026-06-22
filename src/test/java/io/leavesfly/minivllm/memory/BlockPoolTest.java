package io.leavesfly.minivllm.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * BlockPool 单元测试 —— 验证 PagedAttention 物理内存池的分配与释放。
 */
class BlockPoolTest {

    @Test
    void initialStateAllFree() {
        BlockPool pool = new BlockPool(10, 16, 64);
        assertEquals(10, pool.freeBlocks());
        assertEquals(0, pool.usedBlocks());
    }

    @Test
    void allocateReducesFreeCount() {
        BlockPool pool = new BlockPool(5, 16, 64);
        int id = pool.allocate();
        assertTrue(id >= 0);
        assertEquals(4, pool.freeBlocks());
        assertEquals(1, pool.usedBlocks());
    }

    @Test
    void allocateExhaustsPoolReturnsNegative() {
        BlockPool pool = new BlockPool(2, 16, 64);
        int id1 = pool.allocate();
        int id2 = pool.allocate();
        assertTrue(id1 >= 0);
        assertTrue(id2 >= 0);
        // 池满
        int id3 = pool.allocate();
        assertEquals(-1, id3);
    }

    @Test
    void releaseRestoresBlock() {
        BlockPool pool = new BlockPool(3, 16, 64);
        int id = pool.allocate();
        assertEquals(2, pool.freeBlocks());
        pool.release(id);
        assertEquals(3, pool.freeBlocks());
        assertEquals(0, pool.usedBlocks());
    }

    @Test
    void retainAndReleaseWithRefCount() {
        BlockPool pool = new BlockPool(5, 16, 64);
        int id = pool.allocate();
        // retain 增加引用
        pool.retain(id);
        assertEquals(2, pool.get(id).refCount());
        // 第一次 release 不回收
        pool.release(id);
        assertEquals(1, pool.get(id).refCount());
        assertEquals(4, pool.freeBlocks());
        // 第二次 release 才回收
        pool.release(id);
        assertEquals(5, pool.freeBlocks());
    }

    @Test
    void releaseUnderflowThrows() {
        BlockPool pool = new BlockPool(3, 16, 64);
        int id = pool.allocate();
        pool.release(id); // refCount 归 0
        assertThrows(IllegalStateException.class, () -> pool.release(id));
    }

    @Test
    void blockDataHasCorrectSize() {
        int blockSize = 16;
        int dModel = 64;
        BlockPool pool = new BlockPool(2, blockSize, dModel);
        int id = pool.allocate();
        BlockPool.KVBlock block = pool.get(id);
        assertEquals(blockSize * dModel, block.k.length);
        assertEquals(blockSize * dModel, block.v.length);
    }

    @Test
    void kOffsetCalculation() {
        int dModel = 64;
        BlockPool pool = new BlockPool(1, 16, dModel);
        int id = pool.allocate();
        BlockPool.KVBlock block = pool.get(id);
        assertEquals(0, block.kOffset(0, dModel));
        assertEquals(64, block.kOffset(1, dModel));
        assertEquals(128, block.kOffset(2, dModel));
    }
}
