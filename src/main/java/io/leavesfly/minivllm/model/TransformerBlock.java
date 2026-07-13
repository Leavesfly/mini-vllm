package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.LayerNorm;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * TransformerModel Block —— 一个完整的残差子层组合。
 *
 * 学习要点（GPT-2 的 pre-LayerNorm 结构）：
 *   x = x + Attention(LayerNorm(x))      // 第一个残差：自注意力
 *   x = x + FFN(LayerNorm(x))            // 第二个残差：前馈网络
 * 1. pre-LN：先归一化再进子层，残差路径不经 LN，梯度流更稳，是现代 TransformerModel 主流。
 * 2. 残差连接：缓解深层网络退化，使信息能直达输出。
 * 3. 每个 block 拥有自己的一份 KV cache（写在传入的 BlockTable 里）。
 */
public final class TransformerBlock {

    private final LayerNorm ln1;
    private final LayerNorm ln2;
    private final Attention attn;
    private final Ffn ffn;
    private final int dModel;

    public TransformerBlock(LayerNorm ln1, LayerNorm ln2, Attention attn, Ffn ffn, int dModel) {
        this.ln1 = ln1;
        this.ln2 = ln2;
        this.attn = attn;
        this.ffn = ffn;
        this.dModel = dModel;
    }

    /** Prefill：x[seqLen, dModel] -> y[seqLen, dModel] */
    public float[] prefill(float[] x, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
        // 1. 自注意力残差块
        float[] h = x.clone();
        ln1.forwardRowsInPlace(h, seqLen);
        float[] a = attn.prefill(h, seqLen, kvMgr, bt, startIdx);
        for (int i = 0; i < x.length; i++) x[i] += a[i];

        // 2. FFN 残差块
        float[] h2 = x.clone();
        ln2.forwardRowsInPlace(h2, seqLen);
        float[] f = ffn.forwardBatch(h2, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += f[i];
        return x;
    }

    /** Decode：x[dModel] -> y[dModel]（单 token） */
    public float[] decode(float[] x, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        // 1. 自注意力残差块
        float[] h = x.clone();
        ln1.forwardInPlace(h);
        float[] a = attn.decodePaged(h, curIdx, kvMgr, bt);
        for (int i = 0; i < dModel; i++) x[i] += a[i];

        // 2. FFN 残差块
        float[] h2 = x.clone();
        ln2.forwardInPlace(h2);
        float[] f = ffn.forward(h2);
        for (int i = 0; i < dModel; i++) x[i] += f[i];
        return x;
    }

    /**
     * 纯前向（无 KV cache）—— 供 PyTorch 风格 TransformerModel.forward 使用。
     * 结构与 prefill 一致（pre-LN + 两个残差），但注意力走 forwardDense，不读写 KV cache。
     * x[seqLen, dModel] -> y[seqLen, dModel]
     */
    public float[] forward(float[] x, int seqLen) {
        // 1. 自注意力残差块
        float[] h = x.clone();
        ln1.forwardRowsInPlace(h, seqLen);
        float[] a = attn.forwardDense(h, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += a[i];

        // 2. FFN 残差块
        float[] h2 = x.clone();
        ln2.forwardRowsInPlace(h2, seqLen);
        float[] f = ffn.forwardBatch(h2, seqLen);
        for (int i = 0; i < x.length; i++) x[i] += f[i];
        return x;
    }

    /** 该 block 是否使用稀疏注意力 */
    public boolean isSparseAttention() {
        return attn.isSparse();
    }

    /** 参数量（ln1 + ln2 + attn + ffn） */
    public long numParameters() {
        return ln1.numParameters() + ln2.numParameters()
                + attn.numParameters() + ffn.numParameters();
    }
}
