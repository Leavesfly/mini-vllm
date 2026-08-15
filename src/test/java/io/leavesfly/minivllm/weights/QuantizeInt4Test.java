package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.model.Linear;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * QuantizeInt4Test —— INT4 per-group 量化（Q4_0 风格打包）与 dotInt4 内核的单元测试。
 *
 * 覆盖：
 * 1. 打包布局：组内前半元素在低 nibble、后半在高 nibble（byte j = pack(q[j], q[j+G/2])）
 * 2. 内核正确性：dotInt4 结果与「反量化后 double 精度点积」一致（浮点顺序差内容差）
 * 3. 量化损失：int4 反量化权重与原 bf16 权重的点积相对误差在可接受范围
 * 4. matVecInt4 / Linear.ofInt4 前向与逐行 dotInt4 一致
 * 5. 维度约束：cols 不能整除 groupSize 时拒绝量化
 */
class QuantizeInt4Test {

    /** f32 -> bf16 位（截断，与 SafetensorsLoader 的 f32 路径一致） */
    private static short bf16(float f) {
        return (short) (Float.floatToIntBits(f) >>> 16);
    }

    /** 从 Int4Weight 反量化出 f32 权重（测试参考用） */
    private static float[] dequantize(Quantize.Int4Weight q) {
        float[] w = new float[q.rows * q.cols];
        int half = q.groupSize / 2;
        for (int r = 0; r < q.rows; r++) {
            for (int g = 0; g < q.groupsPerRow(); g++) {
                float s = q.scale[r * q.groupsPerRow() + g];
                int elemBase = r * q.cols + g * q.groupSize;
                int packBase = r * (q.cols / 2) + g * half;
                for (int j = 0; j < half; j++) {
                    int b = q.data[packBase + j] & 0xFF;
                    w[elemBase + j] = ((b & 0xF) - 8) * s;
                    w[elemBase + half + j] = ((b >>> 4) - 8) * s;
                }
            }
        }
        return w;
    }

    @Test
    void packingLayoutIsQ40Style() {
        // 单组 G=4：absmax=7 → scale=1，q = [7, -7, 3, 0]（bf16 精确值）
        short[] w = {bf16(7f), bf16(-7f), bf16(3f), bf16(0f)};
        Quantize.Int4Weight q = Quantize.quantizeBf16ToInt4(w, 1, 4, 4);

        assertEquals(1, q.scale.length);
        assertEquals(1f, q.scale[0], 1e-6);
        assertEquals(2, q.data.length);
        // byte0 = pack(q0=7, q2=3) = (7+8) | ((3+8)<<4)；byte1 = pack(q1=-7, q3=0)
        assertEquals((byte) ((7 + 8) | ((3 + 8) << 4)), q.data[0]);
        assertEquals((byte) ((-7 + 8) | ((0 + 8) << 4)), q.data[1]);
    }

    @Test
    void dotInt4MatchesDequantizedReference() {
        int cols = 256, group = 64;
        Random rnd = new Random(42);
        short[] wBf16 = new short[cols];
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            wBf16[i] = bf16((rnd.nextFloat() - 0.5f) * 4f);
            x[i] = (rnd.nextFloat() - 0.5f) * 2f;
        }
        Quantize.Int4Weight q = Quantize.quantizeBf16ToInt4(wBf16, 1, cols, group);
        float[] w = dequantize(q);

        double ref = 0;
        for (int i = 0; i < cols; i++) {
            ref += (double) w[i] * x[i];
        }
        float got = Matmul.dotInt4(q.data, 0, x, 0, cols, q.scale, 0, group);

        float denom = Math.max(1e-3f, Math.abs((float) ref));
        assertEquals((float) ref, got, denom * 1e-4f, "dotInt4 与反量化参考不一致");
    }

    @Test
    void int4QuantizationErrorIsBounded() {
        // int4 量化损失：反量化权重与原 bf16 权重点积的相对误差
        int cols = 1024, group = 64;
        Random rnd = new Random(7);
        short[] wBf16 = new short[cols];
        float[] wF32 = new float[cols];
        float[] x = new float[cols];
        for (int i = 0; i < cols; i++) {
            // 正态分布更接近真实权重（absmax 离群值会拉大量化步长）
            wF32[i] = (float) rnd.nextGaussian() * 0.05f;
            wBf16[i] = bf16(wF32[i]);
            x[i] = (float) rnd.nextGaussian();
        }
        Quantize.Int4Weight q = Quantize.quantizeBf16ToInt4(wBf16, 1, cols, group);
        float[] wDeq = dequantize(q);

        double ref = 0, got = 0;
        for (int i = 0; i < cols; i++) {
            ref += (double) wF32[i] * x[i];
            got += (double) wDeq[i] * x[i];
        }
        double relErr = Math.abs(got - ref) / Math.max(1e-6, Math.abs(ref));
        assertTrue(relErr < 0.05, "int4 量化点积相对误差过大: " + relErr);
    }

    @Test
    void matVecInt4MatchesRowDots() {
        int rows = 8, cols = 128, group = 64;
        Random rnd = new Random(3);
        short[] wBf16 = new short[rows * cols];
        float[] x = new float[cols];
        for (int i = 0; i < wBf16.length; i++) {
            wBf16[i] = bf16((rnd.nextFloat() - 0.5f) * 2f);
        }
        for (int i = 0; i < cols; i++) {
            x[i] = rnd.nextFloat() - 0.5f;
        }
        Quantize.Int4Weight q = Quantize.quantizeBf16ToInt4(wBf16, rows, cols, group);

        float[] y = Matmul.matVecInt4(q.data, q.scale, x, rows, cols, group);
        assertEquals(rows, y.length);
        for (int r = 0; r < rows; r++) {
            float rowDot = Matmul.dotInt4(q.data, r * (cols / 2), x, 0, cols,
                    q.scale, r * q.groupsPerRow(), group);
            assertEquals(rowDot, y[r], 1e-4f, "第 " + r + " 行不一致");
        }
    }

    @Test
    void linearInt4ForwardAndDotRow() {
        int in = 128, out = 4, group = 64;
        Random rnd = new Random(11);
        short[] wBf16 = new short[out * in];
        for (int i = 0; i < wBf16.length; i++) {
            wBf16[i] = bf16((rnd.nextFloat() - 0.5f) * 0.2f);
        }
        float[] x = new float[in];
        for (int i = 0; i < in; i++) {
            x[i] = rnd.nextFloat();
        }
        Quantize.Int4Weight q = Quantize.quantizeBf16ToInt4(wBf16, out, in, group);
        Linear linear = Linear.ofInt4(q.data, q.scale, group, in, out);

        assertTrue(linear.isInt4());
        assertEquals((long) in * out, linear.numParameters());

        float[] y = linear.forward(x);
        assertEquals(out, y.length);
        for (int o = 0; o < out; o++) {
            assertEquals(y[o], linear.dotRow(x, 0, o), 1e-4f, "dotRow 与 forward 第 " + o + " 行不一致");
        }
    }

    @Test
    void nonDivisibleColsAreRejected() {
        short[] w = new short[3 * 10];
        assertThrows(IllegalArgumentException.class,
                () -> Quantize.quantizeBf16ToInt4(w, 3, 10, 64));
    }
}
