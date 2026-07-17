package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.weights.Bf16;

/**
 * 嵌入层 —— 把 token id 映射为稠密向量，是 TransformerModel 的输入入口。
 *
 * 学习要点：
 * 1. 本质是一张查表：weight 形状 [vocabSize, dModel]，第 id 行就是 token id 的向量。
 * 2. GPT-2 中词嵌入与 lm_head 共享权重（tieWordEmbeddings），即用同一张表做输入查表和输出投影。
 * 3. 支持 F32 / BF16 两种常驻精度（二选一）。tied 下这张表既是最大的权重（vocab×d），
 *    又兼作 lm_head——bf16 常驻可显著降内存与 lm_head 的带宽。
 */
public final class Embedding {

    public final float[] weight;      // [vocabSize, dModel] 行优先；bf16 模式为 null
    public final short[] weightBf16;  // bf16 位版；f32 模式为 null
    public final int vocabSize;
    public final int dModel;

    public Embedding(float[] weight, int vocabSize, int dModel) {
        this(weight, null, vocabSize, dModel);
    }

    private Embedding(float[] weight, short[] weightBf16, int vocabSize, int dModel) {
        this.weight = weight;
        this.weightBf16 = weightBf16;
        this.vocabSize = vocabSize;
        this.dModel = dModel;
    }

    /** BF16 位权重常驻构造 */
    public static Embedding ofBf16(short[] weightBf16, int vocabSize, int dModel) {
        return new Embedding(null, weightBf16, vocabSize, dModel);
    }

    /** 是否为 bf16 常驻权重 */
    public boolean isBf16() {
        return weightBf16 != null;
    }

    /** 查单个 token 的向量（返回拷贝） */
    public float[] lookup(int id) {
        float[] r = new float[dModel];
        if (weightBf16 != null) {
            int off = id * dModel;
            for (int d = 0; d < dModel; d++) {
                r[d] = Bf16.bf16ToFloat(weightBf16[off + d] & 0xFFFF);
            }
        } else {
            System.arraycopy(weight, id * dModel, r, 0, dModel);
        }
        return r;
    }

    /** 批量查表：ids[seqLen] -> [seqLen, dModel] */
    public float[] lookupBatch(int[] ids) {
        float[] r = new float[ids.length * dModel];
        for (int i = 0; i < ids.length; i++) {
            if (weightBf16 != null) {
                int off = ids[i] * dModel;
                int dst = i * dModel;
                for (int d = 0; d < dModel; d++) {
                    r[dst + d] = Bf16.bf16ToFloat(weightBf16[off + d] & 0xFFFF);
                }
            } else {
                System.arraycopy(weight, ids[i] * dModel, r, i * dModel, dModel);
            }
        }
        return r;
    }

    /**
     * 反向用：用 hidden 向量与权重表点积，得到每个 token 的 logit。
     * lm_head 与 embedding 共享权重时用此方法：logit[i] = hidden · weight[i]。
     *
     * 性能：weight 布局 [vocabSize, dModel] 与 {@link Matmul#matVec} 的 [out, in] 完全一致，
     * 直接复用向量化 + 行并行内核。lm_head 是全模型最大的一次 matVec（vocabSize≈15 万行），
     * 此前的朴素标量单线程实现是 decode 的头号热点，改走 matVec 后与其余 Linear 层同享 SIMD+多核。
     */
    public float[] projectToVocab(float[] hidden) {
        return weightBf16 != null
                ? Matmul.matVecBf16(weightBf16, hidden, vocabSize, dModel)
                : Matmul.matVec(weight, hidden, vocabSize, dModel);
    }

    /**
     * 批量反向投影：hidden[rows, dModel] -> logits[rows, vocabSize]。
     * PyTorch 风格整段 forward 时用此方法一次性算出每个位置的词表分布。
     * 逐行复用 {@link #projectToVocab}（SIMD + 行并行）。
     */
    public float[] projectToVocabBatch(float[] hidden, int rows) {
        float[] logits = new float[rows * vocabSize];
        float[] row = new float[dModel];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(hidden, r * dModel, row, 0, dModel);
            float[] rowLogits = projectToVocab(row);
            System.arraycopy(rowLogits, 0, logits, r * vocabSize, vocabSize);
        }
        return logits;
    }

    /** 参数量（vocabSize * dModel） */
    public long numParameters() {
        return weight != null ? weight.length : weightBf16.length;
    }
}
