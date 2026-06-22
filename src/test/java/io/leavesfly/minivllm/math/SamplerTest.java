package io.leavesfly.minivllm.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sampler 单元测试 —— 验证采样器各策略的正确性。
 */
class SamplerTest {

    @Test
    void argmaxReturnsMaxIndex() {
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};
        assertEquals(1, Sampler.argmax(logits));
    }

    @Test
    void argmaxWithNegatives() {
        float[] logits = {-3f, -1f, -5f, -2f};
        assertEquals(1, Sampler.argmax(logits));
    }

    @Test
    void sampleWithVeryLowTempIsGreedy() {
        Sampler sampler = new Sampler(42L);
        sampler.temperature = 0.001f;
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};
        // 低温度退化为 greedy
        int result = sampler.sample(logits);
        assertEquals(1, result);
    }

    @Test
    void sampleWithTopK1IsGreedy() {
        Sampler sampler = new Sampler(42L);
        sampler.topK = 1;
        sampler.temperature = 1.0f;
        float[] logits = {1.0f, 5.0f, 3.0f, 2.0f};
        assertEquals(1, sampler.sample(logits));
    }

    @Test
    void sampleReturnsValidIndex() {
        Sampler sampler = new Sampler(123L);
        sampler.temperature = 1.0f;
        float[] logits = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        for (int i = 0; i < 100; i++) {
            int token = sampler.sample(logits);
            assertTrue(token >= 0 && token < logits.length,
                    "采样结果应在有效范围内: " + token);
        }
    }

    @Test
    void sampleWithTopPLimitsDistribution() {
        Sampler sampler = new Sampler(42L);
        sampler.temperature = 1.0f;
        sampler.topP = 0.5f;
        // 给一个明显不均匀的分布，topP=0.5 应限制在概率大的区域
        float[] logits = {0f, 0f, 0f, 0f, 10f};
        int[] counts = new int[5];
        for (int i = 0; i < 100; i++) {
            counts[sampler.sample(logits.clone())]++;
        }
        // index 4 应占绝大多数
        assertTrue(counts[4] > 90, "topP 应限制采样集中在高概率 token");
    }

    @Test
    void sampleDeterministicWithSameSeed() {
        float[] logits = {1f, 2f, 3f, 4f, 5f};
        Sampler s1 = new Sampler(999L);
        s1.temperature = 0.8f;
        Sampler s2 = new Sampler(999L);
        s2.temperature = 0.8f;
        // 相同种子应产生相同序列
        for (int i = 0; i < 20; i++) {
            assertEquals(s1.sample(logits.clone()), s2.sample(logits.clone()));
        }
    }
}
