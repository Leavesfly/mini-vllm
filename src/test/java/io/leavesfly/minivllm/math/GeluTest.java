package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Gelu 单元测试 —— 验证 GELU 激活函数的正确性。
 */
class GeluTest {

    private static final float EPS = 1e-3f;

    @Test
    void geluAtZeroIsZero() {
        assertEquals(0f, Gelu.gelu(0f), EPS);
    }

    @Test
    void geluPositiveInput() {
        // GELU(1.0) ≈ 0.8412
        float result = Gelu.gelu(1.0f);
        assertEquals(0.8412f, result, 0.01f);
    }

    @Test
    void geluNegativeInput() {
        // GELU(-1.0) ≈ -0.1588
        float result = Gelu.gelu(-1.0f);
        assertEquals(-0.1588f, result, 0.01f);
    }

    @Test
    void geluLargePositive() {
        // GELU(x) ≈ x for large positive x
        float x = 5.0f;
        float result = Gelu.gelu(x);
        assertEquals(x, result, 0.01f);
    }

    @Test
    void geluLargeNegative() {
        // GELU(x) ≈ 0 for large negative x
        float result = Gelu.gelu(-5.0f);
        assertEquals(0f, result, 0.01f);
    }

    @Test
    void geluExactCloseToApprox() {
        // 精确版和近似版应接近
        float[] testValues = {-2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f};
        for (float x : testValues) {
            float approx = Gelu.gelu(x);
            float exact = Gelu.geluExact(x);
            assertEquals(exact, approx, 0.02f, "x=" + x + " 精确与近似差距过大");
        }
    }

    @Test
    void applyInPlaceModifiesArray() {
        float[] x = {-1f, 0f, 1f, 2f};
        float[] original = x.clone();
        Gelu.applyInPlace(x);
        // 0 位置不变
        assertEquals(0f, x[1], EPS);
        // 其他位置应被修改
        assertNotEquals(original[0], x[0], EPS);
        assertNotEquals(original[2], x[2], EPS);
    }
}
