package io.leavesfly.minivllm.math;

/**
 * Softmax —— 将任意实数向量归一化为概率分布，是注意力与采样的核心。
 *
 * 学习要点：
 * 1. 数值稳定：先减去最大值再 exp，避免 exp(大数) 溢出为 NaN。这是工程必备技巧。
 * 2. attention 中的 softmax 沿 key 维度做；采样中的 softmax 沿词表维度做。
 */
public final class Softmax {

    private Softmax() {
    }

    /**
     * 就地对一维向量做 softmax，返回自身。
     */
    public static float[] softmaxInPlace(float[] x) {
        float max = ArrayUtil.max(x);
        float sum = 0f;
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max); // 减最大值稳定
            sum += x[i];
        }
        float inv = 1f / sum;
        for (int i = 0; i < x.length; i++) {
            x[i] *= inv;
        }
        return x;
    }

    /**
     * 对二维矩阵每一行做 softmax（行优先存储）。
     * 常用于 attention：每行是一个 query 对所有 key 的分数。
     */
    public static void softmaxRowsInPlace(float[] x, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            float max = x[off];
            for (int j = 1; j < cols; j++) {
                if (x[off + j] > max) max = x[off + j];
            }
            float sum = 0f;
            for (int j = 0; j < cols; j++) {
                x[off + j] = (float) Math.exp(x[off + j] - max);
                sum += x[off + j];
            }
            float inv = 1f / sum;
            for (int j = 0; j < cols; j++) {
                x[off + j] *= inv;
            }
        }
    }

    /**
     * 带温度的 softmax：先除以 temperature 再 softmax。
     * temperature 越小分布越尖锐（更确定），越大越平缓（更随机）。
     */
    public static float[] softmaxWithTemp(float[] logits, float temperature) {
        float[] x = logits.clone();
        float invT = 1f / temperature;
        for (int i = 0; i < x.length; i++) {
            x[i] *= invT;
        }
        return softmaxInPlace(x);
    }
}
