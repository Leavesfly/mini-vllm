package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Matmul 单元测试 —— 验证矩阵乘法正确性。
 */
class MatmulTest {

    private static final float EPS = 1e-5f;

    @Test
    void matmul2dIdentity() {
        // A = [[1,0],[0,1]], B = [[3,4],[5,6]] -> C = B
        float[] a = {1, 0, 0, 1};
        float[] b = {3, 4, 5, 6};
        float[] c = Matmul.matmul(a, 2, 2, b, 2, 2);
        assertArrayEquals(new float[]{3, 4, 5, 6}, c, EPS);
    }

    @Test
    void matmul2dBasic() {
        // A[2,3] * B[3,2]
        // A = [[1,2,3],[4,5,6]]
        // B = [[7,8],[9,10],[11,12]]
        // C[0,0] = 1*7+2*9+3*11 = 58, C[0,1] = 1*8+2*10+3*12 = 64
        // C[1,0] = 4*7+5*9+6*11 = 139, C[1,1] = 4*8+5*10+6*12 = 154
        float[] a = {1, 2, 3, 4, 5, 6};
        float[] b = {7, 8, 9, 10, 11, 12};
        float[] c = Matmul.matmul(a, 2, 3, b, 3, 2);
        assertArrayEquals(new float[]{58, 64, 139, 154}, c, EPS);
    }

    @Test
    void matmulDimensionMismatchThrows() {
        float[] a = new float[6]; // 2x3
        float[] b = new float[8]; // 4x2 (k=4 != 3)
        assertThrows(IllegalArgumentException.class, () -> Matmul.matmul(a, 2, 3, b, 4, 2));
    }

    @Test
    void matmulTensorVersion() {
        Tensor a = new Tensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor b = new Tensor(new float[]{7, 8, 9, 10, 11, 12}, 3, 2);
        Tensor c = Matmul.matmul(a, b);
        assertArrayEquals(new int[]{2, 2}, c.shape());
        assertEquals(58f, c.get2d(0, 0), EPS);
        assertEquals(154f, c.get2d(1, 1), EPS);
    }

    @Test
    void vecMat() {
        // x = [1,2], W = [[1,0,1],[0,1,1]] (2x3)
        // y = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3]
        float[] x = {1, 2};
        float[] w = {1, 0, 1, 0, 1, 1};
        float[] y = Matmul.vecMat(x, w, 2, 3);
        assertArrayEquals(new float[]{1, 2, 3}, y, EPS);
    }

    @Test
    void matVec() {
        // W = [[1,2],[3,4],[5,6]] (3x2), x = [1,1]
        // y = [3, 7, 11]
        float[] w = {1, 2, 3, 4, 5, 6};
        float[] x = {1, 1};
        float[] y = Matmul.matVec(w, x, 3, 2);
        assertArrayEquals(new float[]{3, 7, 11}, y, EPS);
    }
}
