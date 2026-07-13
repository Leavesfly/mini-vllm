package io.leavesfly.minivllm.math;

/**
 * LayerNorm —— 层归一化，TransformerModel 每个 sublayer 前后都会用到。
 *
 * 学习要点：
 * 1. 对最后一维做归一化：减均值、除标准差，再用可学习参数 gamma(缩放) beta(偏移) 仿射变换。
 * 2. 作用：稳定训练与推理的数值分布，让深层网络可训练。
 * 3. eps 防止除零。这里实现的是标准 LayerNorm（非 RMSNorm）；GPT-2 用 LayerNorm，LLaMA 用 RMSNorm。
 */
public final class LayerNorm {

    private final float[] gamma;
    private final float[] beta;
    private final float eps;
    private final int dim;

    public LayerNorm(float[] gamma, float[] beta, float eps) {
        this.gamma = gamma;
        this.beta = beta;
        this.eps = eps;
        this.dim = gamma.length;
    }

    /**
     * 对一个向量就地做 LayerNorm（输入会被改写为输出）。
     */
    public float[] forwardInPlace(float[] x) {
        // 1. 求均值
        float mean = 0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= dim;
        // 2. 求方差
        float var = 0f;
        for (int i = 0; i < dim; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= dim;
        // 3. 归一化 + 仿射
        float invStd = 1f / (float) Math.sqrt(var + eps);
        for (int i = 0; i < dim; i++) {
            x[i] = (x[i] - mean) * invStd * gamma[i] + beta[i];
        }
        return x;
    }

    /**
     * 对二维矩阵每一行做 LayerNorm（行优先存储）。
     */
    public void forwardRowsInPlace(float[] x, int rows) {
        for (int r = 0; r < rows; r++) {
            int off = r * dim;
            float mean = 0f;
            for (int i = 0; i < dim; i++) mean += x[off + i];
            mean /= dim;
            float var = 0f;
            for (int i = 0; i < dim; i++) {
                float d = x[off + i] - mean;
                var += d * d;
            }
            var /= dim;
            float invStd = 1f / (float) Math.sqrt(var + eps);
            for (int i = 0; i < dim; i++) {
                x[off + i] = (x[off + i] - mean) * invStd * gamma[i] + beta[i];
            }
        }
    }

    /** 参数量（gamma + beta） */
    public long numParameters() {
        return (long) gamma.length + beta.length;
    }
}
