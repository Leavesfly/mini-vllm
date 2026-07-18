package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;

/**
 * 线性层 y = x·Wᵀ + b —— TransformerModel 中最常见的算子（Q/K/V/O 投影、FFN、lm_head 都是它）。
 *
 * 学习要点：
 * 1. 权重布局与 PyTorch nn.Linear 一致：W 形状 [outFeatures, inFeatures]（行优先），
 *    计算 y = x·Wᵀ + b，即 y[i] = Σ_p x[p] * W[i*in+p] + b[i]。
 * 2. 零依赖下没有 cuBLAS，矩阵乘退化为 Matmul.matVec 的逐行点积。
 * 3. 权重支持三种常驻精度（互斥，其余为 null）：
 *    - weight(F32)：默认，随机初始化 / GPT-2 路径使用；
 *    - weightBf16(BF16 位)：Qwen3 真实权重可直接以 bf16 常驻，内存/带宽减半；
 *    - weightInt8(INT8 量化)：per-row 对称量化，带宽为 bf16 的一半，decode 理论再提速 60-80%。
 */
public final class Linear {

    public final float[] weight;        // [outFeatures, inFeatures] 行优先；bf16/int8 模式为 null
    public final short[] weightBf16;    // bf16 位版权重；f32/int8 模式为 null
    public final byte[] weightInt8;     // int8 量化权重；f32/bf16 模式为 null
    public final float[] scaleInt8;     // int8 per-row 缩放因子 [outFeatures]；非 int8 模式为 null
    public final float[] bias;          // [outFeatures]，可为 null
    public final int inFeatures;
    public final int outFeatures;

    public Linear(float[] weight, float[] bias, int inFeatures, int outFeatures) {
        this(weight, null, null, null, bias, inFeatures, outFeatures);
    }

    private Linear(float[] weight, short[] weightBf16, byte[] weightInt8, float[] scaleInt8,
                   float[] bias, int inFeatures, int outFeatures) {
        this.weight = weight;
        this.weightBf16 = weightBf16;
        this.weightInt8 = weightInt8;
        this.scaleInt8 = scaleInt8;
        this.bias = bias;
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }

    /** 是否为 bf16 常驻权重 */
    public boolean isBf16() {
        return weightBf16 != null;
    }

    /** 是否为 int8 量化权重 */
    public boolean isInt8() {
        return weightInt8 != null;
    }

    /** 单向量前向：x[inFeatures] -> y[outFeatures] */
    public float[] forward(float[] x) {
        float[] y;
        if (weightInt8 != null) {
            y = Matmul.matVecInt8(weightInt8, scaleInt8, x, outFeatures, inFeatures);
        } else if (weightBf16 != null) {
            y = Matmul.matVecBf16(weightBf16, x, outFeatures, inFeatures);
        } else {
            y = Matmul.matVec(weight, x, outFeatures, inFeatures);
        }
        if (bias != null) {
            for (int i = 0; i < outFeatures; i++) {
                y[i] += bias[i];
            }
        }
        return y;
    }

    /**
     * 批量前向：x[m, inFeatures] -> y[m, outFeatures]
     * prefill 阶段对整段 prompt 一次性投影时使用。
     *
     * 性能：按输出通道 o 单次并行（而非逐行 m 次 matVec 分发），
     * 每个权重行 weight[o] 只读一次并复用到全部 m 个输入行，缓存友好、降线程调度开销。
     * 每个 y[i][o] 与逐行 matVec 一致，结果 bitwise 相同。
     */
    public float[] forwardBatch(float[] x, int m) {
        float[] y = new float[m * outFeatures];
        boolean int8 = weightInt8 != null;
        boolean bf16 = weightBf16 != null;
        Matmul.parallelRows(outFeatures, o -> {
            int wOff = o * inFeatures;
            float b = bias != null ? bias[o] : 0f;
            for (int i = 0; i < m; i++) {
                float dot;
                if (int8) {
                    dot = Matmul.dotInt8(weightInt8, wOff, x, i * inFeatures, inFeatures, scaleInt8[o]);
                } else if (bf16) {
                    dot = Matmul.dotBf16(weightBf16, wOff, x, i * inFeatures, inFeatures);
                } else {
                    dot = Matmul.dot(weight, wOff, x, i * inFeatures, inFeatures);
                }
                y[i * outFeatures + o] = dot + b;
            }
        });
        return y;
    }

    /** 无偏置线性层便捷构造（F32 权重） */
    public static Linear of(float[] weight, int inFeatures, int outFeatures) {
        return new Linear(weight, null, null, null, null, inFeatures, outFeatures);
    }

    /** 无偏置线性层便捷构造（BF16 位权重常驻） */
    public static Linear ofBf16(short[] weightBf16, int inFeatures, int outFeatures) {
        return new Linear(null, weightBf16, null, null, null, inFeatures, outFeatures);
    }

    /** 无偏置线性层便捷构造（INT8 量化权重 + per-row scale） */
    public static Linear ofInt8(byte[] weightInt8, float[] scaleInt8, int inFeatures, int outFeatures) {
        return new Linear(null, null, weightInt8, scaleInt8, null, inFeatures, outFeatures);
    }

    /** 参数量（PyTorch: weight.numel() + bias.numel()） */
    public long numParameters() {
        long w = weight != null ? weight.length : (weightBf16 != null ? weightBf16.length : weightInt8.length);
        return w + (bias == null ? 0L : bias.length);
    }
}
