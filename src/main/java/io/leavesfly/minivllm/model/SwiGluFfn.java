package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Silu;

/**
 * SwiGluFfn —— SwiGLU 前馈网络，Qwen / LLaMA 系列的 FFN 结构。
 *
 * 学习要点：
 * 1. 结构：down( silu(gate(x)) * up(x) )，三个无 bias 线性层。
 *    相比 GPT-2 的 fc1->GELU->fc2，多一路 gate 投影做门控，同参数量下表达更强。
 * 2. Qwen3-0.6B：dModel=1024，intermediate=3072（约 3×dModel，而非 GPT-2 的 4×）。
 * 3. FFN 对每个 token 独立作用，与 attention 的跨 token 交互互补。
 */
public final class SwiGluFfn {

    private final Linear gate; // dModel -> dFfn
    private final Linear up;   // dModel -> dFfn
    private final Linear down; // dFfn -> dModel

    public SwiGluFfn(Linear gate, Linear up, Linear down) {
        this.gate = gate;
        this.up = up;
        this.down = down;
    }

    /** 单 token 前向：x[dModel] -> y[dModel] */
    public float[] forward(float[] x) {
        float[] g = gate.forward(x);
        float[] u = up.forward(x);
        for (int i = 0; i < g.length; i++) {
            g[i] = Silu.silu(g[i]) * u[i];
        }
        return down.forward(g);
    }

    /** 批量前向：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] forwardBatch(float[] x, int seqLen) {
        float[] g = gate.forwardBatch(x, seqLen); // [seqLen, dFfn]
        float[] u = up.forwardBatch(x, seqLen);
        for (int i = 0; i < g.length; i++) {
            g[i] = Silu.silu(g[i]) * u[i];
        }
        return down.forwardBatch(g, seqLen);
    }

    /** 参数量（gate + up + down） */
    public long numParameters() {
        return gate.numParameters() + up.numParameters() + down.numParameters();
    }
}
