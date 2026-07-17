package io.leavesfly.minivllm.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * RotaryEmbedding 单元测试 —— 手算参考值对齐（half-split / GPT-NeoX 风格）。
 */
class RotaryEmbeddingTest {

    @Test
    void positionZeroIsIdentity() {
        // pos=0 时 angle=0，cos=1 sin=0，旋转为恒等
        RotaryEmbedding rope = new RotaryEmbedding(4, 16, 10000f);
        float[] x = {1f, 2f, 3f, 4f};
        rope.applyInPlace(x, 0, 0);
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f}, x, 1e-6f);
    }

    @Test
    void handComputedRotation() {
        // headDim=4, theta=10000：half=2
        // invFreq[0] = 10000^0 = 1，invFreq[1] = 10000^(-2/4) = 0.01
        // pos=1：angle0=1.0，angle1=0.01
        // half-split 配对：(x0,x2) 转 angle0，(x1,x3) 转 angle1
        RotaryEmbedding rope = new RotaryEmbedding(4, 16, 10000f);
        float[] x = {1f, 0f, 0f, 1f};
        rope.applyInPlace(x, 0, 1);
        double c0 = Math.cos(1.0), s0 = Math.sin(1.0);
        double c1 = Math.cos(0.01), s1 = Math.sin(0.01);
        assertEquals((float) c0, x[0], 1e-6f);              // x0' = 1*cos(1) - 0*sin(1)
        assertEquals((float) (-s1), x[1], 1e-6f);           // x1' = 0*cos(.01) - 1*sin(.01)
        assertEquals((float) s0, x[2], 1e-6f);              // x2' = 0*cos(1) + 1*sin(1)
        assertEquals((float) c1, x[3], 1e-6f);              // x3' = 1*cos(.01) + 0*sin(.01)
    }

    @Test
    void rotationPreservesNorm() {
        // 旋转是正交变换，L2 范数不变
        RotaryEmbedding rope = new RotaryEmbedding(8, 64, 1000000f);
        float[] x = new float[8];
        for (int i = 0; i < 8; i++) {
            x[i] = i * 0.7f - 2f;
        }
        float before = norm(x);
        rope.applyInPlace(x, 0, 37);
        assertEquals(before, norm(x), 1e-5f);
    }

    @Test
    void cosSinTableValues() {
        // invFreq[i] = theta^(-2i/headDim)
        RotaryEmbedding rope = new RotaryEmbedding(4, 16, 100f);
        // pos=2, i=1：angle = 2 * 100^(-2/4) = 2 * 0.1 = 0.2
        assertEquals((float) Math.cos(0.2), rope.cosAt(2, 1), 1e-6f);
        assertEquals((float) Math.sin(0.2), rope.sinAt(2, 1), 1e-6f);
    }

    private static float norm(float[] x) {
        float s = 0f;
        for (float v : x) {
            s += v * v;
        }
        return (float) Math.sqrt(s);
    }
}
