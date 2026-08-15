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
 * 5. INT4 per-group 量化（{@link Int4Weight}）：per-row 只有一个 scale 粒度太粗，
 *    4-bit 下误差不可接受，因此按 group（默认 64 元素）细分，每组一个 scale。
 *    打包布局参照 llama.cpp Q4_0：组内前半元素放低 nibble、后半放高 nibble，
 *    即 byte j = pack(q[j], q[j + G/2])——解包后两段各自连续，SIMD 内核友好。
 *    实测（Qwen3-0.6B / Apple Silicon）：0.6B 未打满带宽瓶颈，量化路径反而更慢，
 *    int4 的用武之地是 4B+ 大模型的内存可行为与带宽受限平台。
 */
public final class Quantize {

    /** INT4 默认分组大小（元素数）：2 的幂，主流 hidden 维度均可整除 */
    public static final int DEFAULT_INT4_GROUP = 64;

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
     * INT4 per-group 对称量化结果：packed 4-bit 权重 + per-group scale。
     *
     * 布局（行优先，每行 cols/2 字节）：行内按 groupSize 元素分组，组内前半 G/2 个
     * 元素存低 nibble、后半 G/2 个存高 nibble（Q4_0 风格）：
     *   data[row][g*G/2 + j] = (q[g*G + j] + 8) | ((q[g*G + j + G/2] + 8) << 4)
     * q ∈ [-7, 7]，存为 q+8 ∈ [1, 15]；解包 (nibble - 8) 还原。
     */
    public static final class Int4Weight {
        /** packed 权重 [rows * cols/2]，每字节两个 4-bit 元素 */
        public final byte[] data;
        /** per-group 缩放因子 [rows * (cols/groupSize)]，反量化：w ≈ (nibble-8) * scale[group] */
        public final float[] scale;
        public final int rows;
        public final int cols;
        /** 每组元素数（一个 scale 覆盖的范围） */
        public final int groupSize;

        public Int4Weight(byte[] data, float[] scale, int rows, int cols, int groupSize) {
            this.data = data;
            this.scale = scale;
            this.rows = rows;
            this.cols = cols;
            this.groupSize = groupSize;
        }

        /** 每行的分组数 */
        public int groupsPerRow() {
            return cols / groupSize;
        }
    }

    /**
     * 从 BF16 位权重 [rows, cols] 行优先 -> INT4 per-group 对称量化。
     * scale = 组内 absmax / 7（4-bit 有符号对称区间 [-7, 7]）；要求 cols % groupSize == 0。
     * 内存为 bf16 的 1/4（0.5 字节/元素 + scale 摊销），decode 带宽再减半。
     */
    public static Int4Weight quantizeBf16ToInt4(short[] weightBf16, int rows, int cols, int groupSize) {
        if (groupSize <= 0 || (groupSize & 1) != 0) {
            throw new IllegalArgumentException("groupSize 必须为正偶数: " + groupSize);
        }
        if (cols % groupSize != 0) {
            throw new IllegalArgumentException("cols=" + cols + " 不能整除 groupSize=" + groupSize);
        }
        int groupsPerRow = cols / groupSize;
        int half = groupSize / 2;
        byte[] data = new byte[rows * (cols / 2)];
        float[] scale = new float[rows * groupsPerRow];
        for (int r = 0; r < rows; r++) {
            int rowOff = r * cols;
            int packRow = r * (cols / 2);
            for (int g = 0; g < groupsPerRow; g++) {
                int base = rowOff + g * groupSize;
                float absMax = 0f;
                for (int c = 0; c < groupSize; c++) {
                    float v = Math.abs(Float.intBitsToFloat((weightBf16[base + c] & 0xFFFF) << 16));
                    if (v > absMax) absMax = v;
                }
                float s = absMax > 0f ? absMax / 7f : 1f; // 全零组 scale=1（量化后全为 0）
                scale[r * groupsPerRow + g] = s;
                float invS = 1f / s;
                int packBase = packRow + g * half;
                for (int j = 0; j < half; j++) {
                    int q0 = clamp7(Math.round(Float.intBitsToFloat((weightBf16[base + j] & 0xFFFF) << 16) * invS));
                    int q1 = clamp7(Math.round(Float.intBitsToFloat((weightBf16[base + half + j] & 0xFFFF) << 16) * invS));
                    data[packBase + j] = (byte) ((q0 + 8) | ((q1 + 8) << 4));
                }
            }
        }
        return new Int4Weight(data, scale, rows, cols, groupSize);
    }

    /** clamp 到 4-bit 对称区间 [-7, 7] */
    private static int clamp7(int q) {
        return q > 7 ? 7 : Math.max(q, -7);
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
