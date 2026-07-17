package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * RmsNorm / Silu 单元测试 —— 手算参考值对齐。
 */
class RmsNormTest {

    private static final float EPS = 1e-6f;

    @Test
    void rmsNormHandComputed() {
        // x = [1, 2, 3]，gamma = [1,1,1]，eps = 0
        // ss = (1+4+9)/3 = 14/3，inv = 1/sqrt(14/3)
        float[] x = {1f, 2f, 3f};
        RmsNorm norm = new RmsNorm(new float[]{1f, 1f, 1f}, 0f);
        norm.forwardInPlace(x);
        float inv = 1f / (float) Math.sqrt(14.0 / 3.0);
        assertArrayEquals(new float[]{inv, 2 * inv, 3 * inv}, x, 1e-6f);
    }

    @Test
    void rmsNormWithGamma() {
        // gamma 缩放：y = x/rms * gamma
        float[] x = {1f, -1f};
        RmsNorm norm = new RmsNorm(new float[]{2f, 0.5f}, 0f);
        norm.forwardInPlace(x);
        float inv = 1f / (float) Math.sqrt((1.0 + 1.0) / 2.0); // = 1
        assertArrayEquals(new float[]{2f * inv, -0.5f * inv}, x, 1e-6f);
    }

    @Test
    void rmsNormEpsPreventsDivByZero() {
        float[] x = {0f, 0f};
        RmsNorm norm = new RmsNorm(new float[]{1f, 1f}, 1e-6f);
        norm.forwardInPlace(x); // 不应产生 NaN
        assertEquals(0f, x[0], 1e-6f);
        assertEquals(0f, x[1], 1e-6f);
    }

    @Test
    void rmsNormOffset() {
        // QK-Norm 场景：大数组中按头区间归一化
        float[] x = {99f, 99f, 1f, 2f, 3f, 4f, 88f};
        RmsNorm norm = new RmsNorm(new float[]{1f, 1f, 1f, 1f}, 0f);
        norm.forwardInPlace(x, 2); // 只归一化 [2,6)
        assertEquals(99f, x[0], 1e-6f); // 区间外不动
        assertEquals(88f, x[6], 1e-6f);
        float ss = (1f + 4f + 9f + 16f) / 4f;
        float inv = 1f / (float) Math.sqrt(ss);
        assertEquals(inv, x[2], 1e-6f);
        assertEquals(4 * inv, x[5], 1e-6f);
    }

    @Test
    void rmsNormRows() {
        float[] x = {1f, 2f, 3f, 4f}; // 两行 [1,2] [3,4]
        RmsNorm norm = new RmsNorm(new float[]{1f, 1f}, 0f);
        norm.forwardRowsInPlace(x, 2);
        float inv1 = 1f / (float) Math.sqrt((1.0 + 4.0) / 2.0);
        float inv2 = 1f / (float) Math.sqrt((9.0 + 16.0) / 2.0);
        assertArrayEquals(new float[]{inv1, 2 * inv1, 3 * inv2, 4 * inv2}, x, 1e-6f);
    }

    @Test
    void siluValues() {
        assertEquals(0f, Silu.silu(0f), 1e-7f);
        // silu(1) = 1 * sigmoid(1) ≈ 0.7310586
        assertEquals(0.7310586f, Silu.silu(1f), 1e-6f);
        // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689414
        assertEquals(-0.2689414f, Silu.silu(-1f), 1e-6f);
        float[] arr = {0f, 1f};
        Silu.applyInPlace(arr);
        assertArrayEquals(new float[]{0f, 0.7310586f}, arr, 1e-6f);
    }
}
