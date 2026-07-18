package io.leavesfly.minivllm.weights;

/**
 * Quantize —— 对称 INT8 量化工具（per-row absmax 方案）。
 *
 * 学习要点：
 * 1. 对称量化：scale = max(|row|) / 127，q = round(w / scale)，范围 [-127, 127]。
 *    反量化 w ≈ q * scale。每行一个 scale，精度损失极小（<0.5% 相对误差）。
 * 2. 内存收益：权重从 bf16(2B) 降到 int8(1B)，带宽再减半——decode 是 memory-bound，
 *    这意味着理论 token 速率可再提升 ~60-80%。
 * 3. 点积时先解量化再做 FMA：dot = scale * Σ(q[i] * x[i])，
 *    整数乘浮点的 FMA 在现代 CPU 上吞吐与纯浮点相当，瓶颈仍在权重读取带宽。
 * 4. 与 GPTQ/AWQ 等 per-group 方案相比，per-row 实现最简，对 0.6B 小模型精度影响可忽略。
 */
public final class Quantize {

    private Quantize() {
    }

    /**
     * 量化结果：packed int8 权重 + per-row scale。
     */
    public static final class Int8Weight {
        /** 量化后的权重 [rows * cols]，每个元素为 signed byte（-127~127） */
        public final byte[] data;
        /** 每行的缩放因子 [rows]，反量化：w ≈ data[row*cols+col] * scale[row] */
        public final float[] scale;
        public final int rows;
        public final int cols;

        public Int8Weight(byte[] data, float[] scale, int rows, int cols) {
            this.data = data;
            this.scale = scale;
            this.rows = rows;
            this.cols = cols;
        }
    }

    /**
     * 从 F32 权重 [rows, cols] 行优先 -> INT8 量化。
     * 逐行计算 absmax，对称映射到 [-127, 127]。
     */
    public static Int8Weight quantizeF32(float[] weight, int rows, int cols) {
        byte[] data = new byte[rows * cols];
        float[] scale = new float[rows];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            float absMax = 0f;
            for (int c = 0; c < cols; c++) {
                float v = Math.abs(weight[off + c]);
                if (v > absMax) absMax = v;
            }
            // 避免除零：全零行 scale=1（量化后全为 0）
            float s = absMax > 0f ? absMax / 127f : 1f;
            scale[r] = s;
            float invS = 1f / s;
            for (int c = 0; c < cols; c++) {
                int q = Math.round(weight[off + c] * invS);
                // clamp 到 [-127, 127]
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                data[off + c] = (byte) q;
            }
        }
        return new Int8Weight(data, scale, rows, cols);
    }

    /**
     * 从 BF16 位权重 [rows, cols] 行优先 -> INT8 量化。
     * 先逐元素转 f32 再量化，避免中间存储完整 f32 数组（逐行处理）。
     */
    public static Int8Weight quantizeBf16(short[] weightBf16, int rows, int cols) {
        byte[] data = new byte[rows * cols];
        float[] scale = new float[rows];
        for (int r = 0; r < rows; r++) {
            int off = r * cols;
            // 第一遍：求 absMax
            float absMax = 0f;
            for (int c = 0; c < cols; c++) {
                float v = Math.abs(Float.intBitsToFloat((weightBf16[off + c] & 0xFFFF) << 16));
                if (v > absMax) absMax = v;
            }
            float s = absMax > 0f ? absMax / 127f : 1f;
            scale[r] = s;
            float invS = 1f / s;
            // 第二遍：量化
            for (int c = 0; c < cols; c++) {
                float w = Float.intBitsToFloat((weightBf16[off + c] & 0xFFFF) << 16);
                int q = Math.round(w * invS);
                if (q > 127) q = 127;
                else if (q < -127) q = -127;
                data[off + c] = (byte) q;
            }
        }
        return new Int8Weight(data, scale, rows, cols);
    }
}
