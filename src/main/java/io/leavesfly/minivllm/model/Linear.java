package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;

/**
 * 线性层 y = x·Wᵀ + b —— Transformer 中最常见的算子（Q/K/V/O 投影、FFN、lm_head 都是它）。
 *
 * 学习要点：
 * 1. 权重布局与 PyTorch nn.Linear 一致：W 形状 [outFeatures, inFeatures]（行优先），
 *    计算 y = x·Wᵀ + b，即 y[i] = Σ_p x[p] * W[i*in+p] + b[i]。
 * 2. 零依赖下没有 cuBLAS，矩阵乘退化为 Matmul.matVec 的逐行点积。
 */
public final class Linear {

    public final float[] weight; // [outFeatures, inFeatures] 行优先
    public final float[] bias;   // [outFeatures]，可为 null
    public final int inFeatures;
    public final int outFeatures;

    public Linear(float[] weight, float[] bias, int inFeatures, int outFeatures) {
        this.weight = weight;
        this.bias = bias;
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }

    /** 单向量前向：x[inFeatures] -> y[outFeatures] */
    public float[] forward(float[] x) {
        float[] y = Matmul.matVec(weight, x, outFeatures, inFeatures);
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
     */
    public float[] forwardBatch(float[] x, int m) {
        float[] y = new float[m * outFeatures];
        float[] xr = new float[inFeatures];
        for (int i = 0; i < m; i++) {
            System.arraycopy(x, i * inFeatures, xr, 0, inFeatures);
            float[] yr = Matmul.matVec(weight, xr, outFeatures, inFeatures);
            if (bias != null) {
                for (int o = 0; o < outFeatures; o++) {
                    yr[o] += bias[o];
                }
            }
            System.arraycopy(yr, 0, y, i * outFeatures, outFeatures);
        }
        return y;
    }

    /** 无偏置线性层便捷构造 */
    public static Linear of(float[] weight, int inFeatures, int outFeatures) {
        return new Linear(weight, null, inFeatures, outFeatures);
    }
}
