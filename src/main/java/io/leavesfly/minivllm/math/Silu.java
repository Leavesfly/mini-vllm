package io.leavesfly.minivllm.math;

/**
 * SiLU（Sigmoid Linear Unit）—— SwiGLU 的激活函数。
 *
 * 学习要点：
 * 1. silu(x) = x * sigmoid(x) = x / (1 + e^-x)，也叫 swish。
 * 2. Qwen3 的 FFN 是 SwiGLU：down( silu(gate(x)) * up(x) )，
 *    相比 GPT-2 的 GELU 两层结构，多一路门控投影，表达能力更强。
 */
public final class Silu {

    private Silu() {
    }

    /** 单点 silu */
    public static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    /** 对整个数组就地施加 silu */
    public static void applyInPlace(float[] x) {
        for (int i = 0; i < x.length; i++) {
            x[i] = silu(x[i]);
        }
    }
}
