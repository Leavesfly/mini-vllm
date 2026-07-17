package io.leavesfly.minivllm.math;

/**
 * RmsNorm —— 均方根归一化，LLaMA / Qwen 系列模型的标准归一化层。
 *
 * 学习要点：
 * 1. 与 LayerNorm 的区别：不减均值、无 bias（beta），只做缩放：
 *      y = x / sqrt(mean(x²) + eps) * gamma
 *    计算更简单，实践中效果与 LayerNorm 相当，是现代 LLM 的主流选择。
 * 2. Qwen3 还在注意力的 Q/K 上按头做 RMSNorm（QK-Norm），此时 dim = headDim，
 *    需要对大数组中的每个头区间分别归一化，故提供带 offset 的版本。
 */
public final class RmsNorm {

    private final float[] weight; // gamma，[dim]
    private final float eps;
    private final int dim;

    public RmsNorm(float[] weight, float eps) {
        this.weight = weight;
        this.eps = eps;
        this.dim = weight.length;
    }

    /** 对完整向量就地做 RmsNorm（x.length 必须等于 dim） */
    public float[] forwardInPlace(float[] x) {
        forwardInPlace(x, 0);
        return x;
    }

    /**
     * 对大数组中 [offset, offset+dim) 区间就地做 RmsNorm。
     * 用于 QK-Norm：Q/K 投影结果按头切分后逐头归一化。
     */
    public void forwardInPlace(float[] x, int offset) {
        float ss = 0f;
        for (int i = 0; i < dim; i++) {
            float v = x[offset + i];
            ss += v * v;
        }
        float inv = 1f / (float) Math.sqrt(ss / dim + eps);
        for (int i = 0; i < dim; i++) {
            x[offset + i] = x[offset + i] * inv * weight[i];
        }
    }

    /** 对二维矩阵每一行做 RmsNorm（行优先存储，行长必须等于 dim） */
    public void forwardRowsInPlace(float[] x, int rows) {
        for (int r = 0; r < rows; r++) {
            forwardInPlace(x, r * dim);
        }
    }

    /** 参数量（仅 gamma） */
    public long numParameters() {
        return weight.length;
    }
}
