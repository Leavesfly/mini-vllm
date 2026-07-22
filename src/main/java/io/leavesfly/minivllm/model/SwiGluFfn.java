package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.math.Silu;

/**
 * SwiGluFfn —— SwiGLU 前馈网络，Qwen / LLaMA 系列的 FFN 结构。
 *
 * 学习要点：
 * 1. 结构：down( silu(gate(x)) * up(x) )，三个无 bias 线性层。
 *    相比 GPT-2 的 fc1->GELU->fc2，多一路 gate 投影做门控，同参数量下表达更强。
 * 2. Qwen3-0.6B：dModel=1024，intermediate=3072（约 3×dModel，而非 GPT-2 的 4×）。
 * 3. FFN 对每个 token 独立作用，与 attention 的跨 token 交互互补。
 * 4. 性能优化：gate 和 up 读同一个输入 x，融合为单次 parallelRows 调度，
 *    x 只从缓存读一次、线程调度开销减半。
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

    /**
     * 单 token 前向：x[dModel] -> y[dModel]。
     * gate+up 融合：单次 parallelRows 同时计算两路投影，x 只读一次、线程调度减半。
     */
    public float[] forward(float[] x) {
        int dFfn = gate.outFeatures();
        float[] gu = new float[dFfn]; // 复用为 gate 结果，然后就地 silu*up

        // 融合 gate+up：每个输出维度 i 同时算 gate[i] 和 up[i]
        Matmul.parallelRows(dFfn, i -> {
            float g = gate.dotRow(x, 0, i);
            float u = up.dotRow(x, 0, i);
            gu[i] = Silu.silu(g) * u;
        });
        return down.forward(gu);
    }

    /**
     * 批量前向：x[seqLen, dModel] -> y[seqLen, dModel]。
     * gate+up 融合：按输出通道并行，每个权重行只读一次并复用到全部 m 个输入行。
     */
    public float[] forwardBatch(float[] x, int seqLen) {
        int dFfn = gate.outFeatures();
        int in = gate.inFeatures();
        float[] g = new float[seqLen * dFfn];

        Matmul.parallelRows(dFfn, o -> {
            for (int i = 0; i < seqLen; i++) {
                int xOff = i * in;
                float gv = gate.dotRow(x, xOff, o);
                float uv = up.dotRow(x, xOff, o);
                g[i * dFfn + o] = Silu.silu(gv) * uv;
            }
        });
        return down.forwardBatch(g, seqLen);
    }

    /** 参数量（gate + up + down） */
    public long numParameters() {
        return gate.numParameters() + up.numParameters() + down.numParameters();
    }
}
