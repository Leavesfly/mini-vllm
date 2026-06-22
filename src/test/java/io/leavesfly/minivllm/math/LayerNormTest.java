package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * LayerNorm 单元测试 —— 验证层归一化的正确性。
 */
class LayerNormTest {

    private static final float EPS = 1e-4f;

    @Test
    void forwardWithIdentityParams() {
        // gamma=1, beta=0 → 标准化后均值≈0, 方差≈1
        int dim = 4;
        float[] gamma = {1, 1, 1, 1};
        float[] beta = {0, 0, 0, 0};
        LayerNorm ln = new LayerNorm(gamma, beta, 1e-5f);

        float[] x = {1f, 2f, 3f, 4f};
        ln.forwardInPlace(x);

        // 归一化后均值应接近 0
        float mean = 0f;
        for (float v : x) mean += v;
        mean /= dim;
        assertEquals(0f, mean, EPS);

        // 方差应接近 1
        float var = 0f;
        for (float v : x) var += (v - mean) * (v - mean);
        var /= dim;
        assertEquals(1f, var, 0.01f);
    }

    @Test
    void forwardWithGammaAndBeta() {
        // gamma=2, beta=1 → 线性变换后结果
        int dim = 3;
        float[] gamma = {2, 2, 2};
        float[] beta = {1, 1, 1};
        LayerNorm ln = new LayerNorm(gamma, beta, 1e-5f);

        float[] x = {0f, 0f, 0f};
        ln.forwardInPlace(x);
        // 均匀输入归一化后 = 0, 再 *2 + 1 = 1
        for (float v : x) {
            assertEquals(1f, v, EPS);
        }
    }

    @Test
    void forwardRowsInPlace() {
        int dim = 3;
        float[] gamma = {1, 1, 1};
        float[] beta = {0, 0, 0};
        LayerNorm ln = new LayerNorm(gamma, beta, 1e-5f);

        // 2 rows, each dim=3
        float[] x = {1f, 2f, 3f, 10f, 20f, 30f};
        ln.forwardRowsInPlace(x, 2);

        // 每行归一化后均值都应接近 0
        float mean0 = (x[0] + x[1] + x[2]) / 3;
        float mean1 = (x[3] + x[4] + x[5]) / 3;
        assertEquals(0f, mean0, EPS);
        assertEquals(0f, mean1, EPS);
    }

    @Test
    void forwardConstantInputGivesZero() {
        // 常量输入归一化后为 0（再 * gamma + beta）
        float[] gamma = {1, 1, 1, 1};
        float[] beta = {0, 0, 0, 0};
        LayerNorm ln = new LayerNorm(gamma, beta, 1e-5f);

        float[] x = {5f, 5f, 5f, 5f};
        ln.forwardInPlace(x);
        for (float v : x) {
            assertEquals(0f, v, EPS);
        }
    }
}
