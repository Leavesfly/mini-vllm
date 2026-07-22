package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tensor 单元测试 —— 验证张量的创建、访问与基础操作。
 */
class TensorTest {

    @Test
    void createWithMatchingShapeAndData() {
        float[] data = {1, 2, 3, 4, 5, 6};
        Tensor t = new Tensor(data, 2, 3);
        assertEquals(6, t.size());
        assertEquals(2, t.dim());
        assertArrayEquals(new int[]{2, 3}, t.shape());
    }

    @Test
    void createWithMismatchedShapeThrows() {
        float[] data = {1, 2, 3};
        assertThrows(IllegalArgumentException.class, () -> new Tensor(data, 2, 3));
    }

    @Test
    void zerosCreatesAllZeroTensor() {
        Tensor t = Tensor.zeros(3, 4);
        assertEquals(12, t.size());
        for (float v : t.data()) {
            assertEquals(0f, v);
        }
    }

    @Test
    void get2dAndSet2d() {
        Tensor t = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(1f, t.get2d(0, 0));
        assertEquals(5f, t.get2d(1, 1));
        t.set2d(1, 2, 99f);
        assertEquals(99f, t.get2d(1, 2));
    }

    @Test
    void rowReturnsCopy() {
        Tensor t = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        float[] row0 = t.row(0);
        assertArrayEquals(new float[]{1, 2, 3}, row0);
        // 修改返回值不影响原始数据
        row0[0] = 999f;
        assertEquals(1f, t.get2d(0, 0));
    }

    @Test
    void get3dAndSet3d() {
        // shape [2, 3, 4] = 24 elements
        float[] data = new float[24];
        for (int i = 0; i < 24; i++) data[i] = i;
        Tensor t = new Tensor(data, 2, 3, 4);

        // [1,2,3] = index (1*3+2)*4+3 = 5*4+3 = 23
        assertEquals(23f, t.get3d(1, 2, 3));
        t.set3d(0, 0, 0, 100f);
        assertEquals(100f, t.get3d(0, 0, 0));
    }

    @Test
    void flatIndex() {
        Tensor t = Tensor.zeros(2, 3, 4);
        assertEquals(0, t.flatIndex(0, 0, 0));
        assertEquals(23, t.flatIndex(1, 2, 3));
        assertEquals(4, t.flatIndex(0, 1, 0));
    }
}
