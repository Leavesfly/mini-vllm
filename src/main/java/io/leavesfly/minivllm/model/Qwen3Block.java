package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.ArrayUtil;
import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

import java.util.function.UnaryOperator;

/**
 * Qwen3Block —— Qwen3 的 Transformer 层（pre-norm 残差结构）。
 *
 * 学习要点：
 *   x = x + Attention(RmsNorm(x))      // input_layernorm -> GQA -> 残差
 *   x = x + SwiGLU(RmsNorm(x))         // post_attention_layernorm -> MLP -> 残差
 * 与 GPT-2 的 TransformerBlock 结构同形，差异在归一化（RMSNorm vs LayerNorm）
 * 与子层内部（GQA+RoPE+QK-Norm vs MHA+wpe，SwiGLU vs GELU-MLP）。
 */
public final class Qwen3Block {

    private final RmsNorm inputNorm;      // input_layernorm
    private final RmsNorm postAttnNorm;   // post_attention_layernorm
    private final Qwen3Attention attn;
    private final SwiGluFfn ffn;
    private final int dModel;

    /**
     * 预分配工作缓冲（decode 单 token 路径复用，避免每 token 56 次 clone 的 GC 压力）。
     * 引擎 step 为单线程顺序调用，无并发安全问题。
     */
    private final float[] hBuf;
    private final float[] h2Buf;

    public Qwen3Block(RmsNorm inputNorm, RmsNorm postAttnNorm, Qwen3Attention attn, SwiGluFfn ffn, int dModel) {
        this.inputNorm = inputNorm;
        this.postAttnNorm = postAttnNorm;
        this.attn = attn;
        this.ffn = ffn;
        this.dModel = dModel;
        this.hBuf = new float[dModel];
        this.h2Buf = new float[dModel];
    }

    /** Prefill：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] prefill(float[] x, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
        applyResidual(x, seqLen, inputNorm, h -> attn.prefill(h, seqLen, kvMgr, bt, startIdx));
        applyResidual(x, seqLen, postAttnNorm, h -> ffn.forwardBatch(h, seqLen));
        return x;
    }

    /** Decode：x[dModel] -> y[dModel]（单 token，复用预分配缓冲避免 GC） */
    public float[] decode(float[] x, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        System.arraycopy(x, 0, hBuf, 0, dModel);
        inputNorm.forwardInPlace(hBuf);
        float[] a = attn.decodePaged(hBuf, curIdx, kvMgr, bt);
        ArrayUtil.addInPlace(x, a);

        System.arraycopy(x, 0, h2Buf, 0, dModel);
        postAttnNorm.forwardInPlace(h2Buf);
        float[] f = ffn.forward(h2Buf);
        ArrayUtil.addInPlace(x, f);
        return x;
    }

    /**
     * 批量 Decode：x[B, dModel] -> y[B, dModel]（B 个序列各一个新 token）。
     * attention 按序列独立走 KV cache；FFN 与投影按 [B, dModel] 批量（权重跨 B 行复用）。
     */
    public float[] decodeBatch(float[] x, int batch, int[] curIdxs,
                               KVCacheManager kvMgr, BlockTable[] bts) {
        applyResidual(x, batch, inputNorm, h -> attn.decodeBatch(h, batch, curIdxs, kvMgr, bts));
        applyResidual(x, batch, postAttnNorm, h -> ffn.forwardBatch(h, batch));
        return x;
    }

    /** 纯前向（无 KV cache）：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] forward(float[] x, int seqLen) {
        applyResidual(x, seqLen, inputNorm, h -> attn.forwardDense(h, seqLen));
        applyResidual(x, seqLen, postAttnNorm, h -> ffn.forwardBatch(h, seqLen));
        return x;
    }

    /** 残差子步骤：clone → norm → sublayer → 累加回 x */
    private void applyResidual(float[] x, int rows, RmsNorm norm, UnaryOperator<float[]> sublayer) {
        float[] h = x.clone();
        norm.forwardRowsInPlace(h, rows);
        ArrayUtil.addInPlace(x, sublayer.apply(h));
    }

    /** 参数量（inputNorm + postAttnNorm + attn + ffn） */
    public long numParameters() {
        return inputNorm.numParameters() + postAttnNorm.numParameters()
                + attn.numParameters() + ffn.numParameters();
    }
}
