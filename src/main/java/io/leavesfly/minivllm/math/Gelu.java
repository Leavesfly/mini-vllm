package io.leavesfly.minivllm.math;

/**
 * GELU 激活函数 —— GPT-2 的 FFN 中间激活。
 *
 * 学习要点：
 * 1. GELU(x) = x · Φ(x)，Φ 是标准正态分布的累积分布函数。
 * 2. 精确版用 erf，近似版用 tanh 多项式。两者数值接近，近似版计算更快。
 * 3. 相比 ReLU，GELU 在 0 附近更平滑，是现代 Transformer 的标配。
 */
public final class Gelu {

    private Gelu() {
    }

    /** 近似 GELU：0.5x(1 + tanh(√(2/π)(x + 0.044715x³))) */
    public static float gelu(float x) {
        float c = (float) Math.sqrt(2.0 / Math.PI);
        float inner = c * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1f + (float) Math.tanh(inner));
    }

    /** 精确 GELU：0.5x(1 + erf(x/√2)) */
    public static float geluExact(float x) {
        return 0.5f * x * (1f + erf(x / (float) Math.sqrt(2.0)));
    }

    /**
     * 就地对向量应用近似 GELU。
     */
    public static void applyInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = gelu(x[i]);
        }
    }

    // ---- erf 近似（Abramowitz & Stegun 7.1.26），精度约 1e-7 ----
    private static float erf(float x) {
        float t = 1f / (1f + 0.3275911f * Math.abs(x));
        float poly = t * (0.254829592f
                + t * (-0.284496736f
                + t * (1.421413741f
                + t * (-1.453152027f
                + t * 1.061405429f))));
        return x >= 0 ? 1f - poly * (float) Math.exp(-x * x)
                      : poly * (float) Math.exp(-x * x) - 1f;
    }
}
