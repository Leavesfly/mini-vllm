package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Gelu;

/**
 * 前馈网络 (FFN / MLP) —— 两层线性 + GELU 激活。
 *
 * 学习要点：
 * 1. GPT-2 的 FFN：fc1(dModel→dFfn) → GELU → fc2(dFfn→dModel)。
 * 2. dFfn 通常 = 4·dModel，是 TransformerModel 参数量的主要来源。
 * 3. FFN 对每个 token 独立作用（不同 token 间无交互，那是 attention 的职责）。
 */
public final class Ffn {

    private final Linear fc1; // dModel -> dFfn
    private final Linear fc2; // dFfn -> dModel

    public Ffn(Linear fc1, Linear fc2) {
        this.fc1 = fc1;
        this.fc2 = fc2;
    }

    /** 单 token 前向：x[dModel] -> y[dModel] */
    public float[] forward(float[] x) {
        float[] h = fc1.forward(x);
        Gelu.applyInPlace(h);
        return fc2.forward(h);
    }

    /** 批量前向：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] forwardBatch(float[] x, int seqLen) {
        float[] h = fc1.forwardBatch(x, seqLen); // [seqLen, dFfn]
        Gelu.applyInPlace(h);                    // 逐元素激活
        return fc2.forwardBatch(h, seqLen);      // [seqLen, dModel]
    }

    /** 参数量（fc1 + fc2） */
    public long numParameters() {
        return fc1.numParameters() + fc2.numParameters();
    }
}
