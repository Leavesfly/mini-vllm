package io.leavesfly.minivllm.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * BlockTable 单元测试 —— 验证逻辑→物理 block 映射。
 */
class BlockTableTest {

    @Test
    void emptyTableHasZeroBlocks() {
        BlockTable bt = new BlockTable();
        assertEquals(0, bt.numBlocks());
    }

    @Test
    void appendAndAccess() {
        BlockTable bt = new BlockTable();
        bt.append(5);
        bt.append(3);
        bt.append(8);
        assertEquals(3, bt.numBlocks());
        assertEquals(5, bt.blockIdAt(0));
        assertEquals(3, bt.blockIdAt(1));
        assertEquals(8, bt.blockIdAt(2));
    }

    @Test
    void capacityCalculation() {
        BlockTable bt = new BlockTable();
        bt.append(0);
        bt.append(1);
        // 2 blocks * blockSize=16 = 32 token capacity
        assertEquals(32, bt.capacity(16));
    }

    @Test
    void toArrayReturnsCopy() {
        BlockTable bt = new BlockTable();
        bt.append(10);
        bt.append(20);
        int[] arr = bt.toArray();
        assertArrayEquals(new int[]{10, 20}, arr);
        // 修改返回数组不影响原始
        arr[0] = 999;
        assertEquals(10, bt.blockIdAt(0));
    }

    @Test
    void clearEmptiesTable() {
        BlockTable bt = new BlockTable();
        bt.append(1);
        bt.append(2);
        bt.clear();
        assertEquals(0, bt.numBlocks());
    }

    @Test
    void outOfBoundsAccessThrows() {
        BlockTable bt = new BlockTable();
        bt.append(1);
        assertThrows(IndexOutOfBoundsException.class, () -> bt.blockIdAt(1));
    }
}
