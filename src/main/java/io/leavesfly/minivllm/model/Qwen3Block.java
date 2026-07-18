package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

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

    private final RmsNorm ln1;      // input_layernorm
    private final RmsNorm ln2;      // post_attention_layernorm
    private final Qwen3Attention attn;
    private final SwiGluFfn ffn;
    private final int dModel;

    /**
     * 预分配工作缓冲（decode 单 token 路径复用，避免每 token 56 次 clone 的 GC 压力）。
     * 引擎 step 为单线程顺序调用，无并发安全问题。
     */
    private final float[] hBuf;
    private final float[] h2Buf;

    public Qwen3Block(RmsNorm ln1, RmsNorm ln2, Qwen3Attention attn, SwiGluFfn ffn, int dModel) {
        this.ln1 = ln1;
        this.ln2 = ln2;
        this.attn = attn;
        this.ffn = ffn;
        this.dModel = dModel;
        this.hBuf = new float[dModel];
        this.h2Buf = new float[dModel];
    }

    /** Prefill：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] prefill(float[] x, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
        float[] h = x.clone();
        ln1.forwardRowsInPlace(h, seqLen);
        float[] a = attn.prefill(h, seqLen, kvMgr, bt, startIdx);
        for (int i = 0; i < x.length; i++) x[i] += a[i];

        float[] h2 = x.clone();
        ln2.forwardRowsInPlace(h2, seqLen);
        float[] f = ffn.forwardBatch(h2, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += f[i];
        return x;
    }

    /** Decode：x[dModel] -> y[dModel]（单 token，复用预分配缓冲避免 GC） */
    public float[] decode(float[] x, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        System.arraycopy(x, 0, hBuf, 0, dModel);
        ln1.forwardInPlace(hBuf);
        float[] a = attn.decodePaged(hBuf, curIdx, kvMgr, bt);
        for (int i = 0; i < dModel; i++) x[i] += a[i];

        System.arraycopy(x, 0, h2Buf, 0, dModel);
        ln2.forwardInPlace(h2Buf);
        float[] f = ffn.forward(h2Buf);
        for (int i = 0; i < dModel; i++) x[i] += f[i];
        return x;
    }

    /**
     * 批量 Decode：x[B, dModel] -> y[B, dModel]（B 个序列各一个新 token）。
     * attention 按序列独立走 KV cache；FFN 与投影按 [B, dModel] 批量（权重跨 B 行复用）。
     */
    public float[] decodeBatch(float[] x, int batch, int[] curIdxs,
                               KVCacheManager kvMgr, BlockTable[] bts) {
        float[] h = x.clone();
        ln1.forwardRowsInPlace(h, batch);
        float[] a = attn.decodeBatch(h, batch, curIdxs, kvMgr, bts); // [B, dModel]
        for (int i = 0; i < x.length; i++) x[i] += a[i];

        float[] h2 = x.clone();
        ln2.forwardRowsInPlace(h2, batch);
        float[] f = ffn.forwardBatch(h2, batch);
        for (int i = 0; i < x.length; i++) x[i] += f[i];
        return x;
    }

    /** 纯前向（无 KV cache）：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] forward(float[] x, int seqLen) {
        float[] h = x.clone();
        ln1.forwardRowsInPlace(h, seqLen);
        float[] a = attn.forwardDense(h, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += a[i];

        float[] h2 = x.clone();
        ln2.forwardRowsInPlace(h2, seqLen);
        float[] f = ffn.forwardBatch(h2, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += f[i];
        return x;
    }

    /** 参数量（ln1 + ln2 + attn + ffn） */
    public long numParameters() {
        return ln1.numParameters() + ln2.numParameters()
                + attn.numParameters() + ffn.numParameters();
    }
}
