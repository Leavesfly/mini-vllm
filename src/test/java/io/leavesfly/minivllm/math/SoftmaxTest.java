package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Softmax 单元测试 —— 验证概率归一化与数值稳定性。
 */
class SoftmaxTest {

    private static final float EPS = 1e-5f;

    @Test
    void softmaxOutputSumsToOne() {
        float[] x = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] result = Softmax.softmaxInPlace(x);
        float sum = 0f;
        for (float v : result) sum += v;
        assertEquals(1.0f, sum, EPS);
    }

    @Test
    void softmaxPreservesOrder() {
        float[] x = {1.0f, 3.0f, 2.0f};
        Softmax.softmaxInPlace(x);
        // 最大输入应有最大概率
        assertTrue(x[1] > x[2]);
        assertTrue(x[2] > x[0]);
    }

    @Test
    void softmaxNumericalStability() {
        // 大数值不应导致 NaN 或 Infinity
        float[] x = {1000f, 1001f, 1002f};
        float[] result = Softmax.softmaxInPlace(x);
        for (float v : result) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }
        float sum = 0f;
        for (float v : result) sum += v;
        assertEquals(1.0f, sum, EPS);
    }

    @Test
    void softmaxUniformInput() {
        float[] x = {1.0f, 1.0f, 1.0f, 1.0f};
        Softmax.softmaxInPlace(x);
        for (float v : x) {
            assertEquals(0.25f, v, EPS);
        }
    }

    @Test
    void softmaxRowsInPlace() {
        // 2 rows, 3 cols
        float[] x = {1f, 2f, 3f, 1f, 1f, 1f};
        Softmax.softmaxRowsInPlace(x, 2, 3);
        // 第一行和为 1
        float sum0 = x[0] + x[1] + x[2];
        assertEquals(1.0f, sum0, EPS);
        // 第二行均匀
        assertEquals(x[3], x[4], EPS);
        assertEquals(x[4], x[5], EPS);
    }

    @Test
    void softmaxWithTempLow() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        // 低温度使分布更尖锐
        float[] low = Softmax.softmaxWithTemp(logits, 0.1f);
        float[] high = Softmax.softmaxWithTemp(logits, 2.0f);
        // 低温度下最大值的概率应更接近 1
        assertTrue(low[2] > high[2]);
    }

    @Test
    void softmaxWithTempDoesNotModifyInput() {
        float[] logits = {1.0f, 2.0f, 3.0f};
        float[] copy = logits.clone();
        Softmax.softmaxWithTemp(logits, 0.5f);
        assertArrayEquals(copy, logits);
    }
}
