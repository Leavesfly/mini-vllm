package io.leavesfly.minivllm.model;

/**
 * 嵌入层 —— 把 token id 映射为稠密向量，是 Transformer 的输入入口。
 *
 * 学习要点：
 * 1. 本质是一张查表：weight 形状 [vocabSize, dModel]，第 id 行就是 token id 的向量。
 * 2. GPT-2 中词嵌入与 lm_head 共享权重（tieWordEmbeddings），即用同一张表做输入查表和输出投影。
 */
public final class Embedding {

    public final float[] weight; // [vocabSize, dModel] 行优先
    public final int vocabSize;
    public final int dModel;

    public Embedding(float[] weight, int vocabSize, int dModel) {
        this.weight = weight;
        this.vocabSize = vocabSize;
        this.dModel = dModel;
    }

    /** 查单个 token 的向量（返回拷贝） */
    public float[] lookup(int id) {
        float[] r = new float[dModel];
        System.arraycopy(weight, id * dModel, r, 0, dModel);
        return r;
    }

    /** 批量查表：ids[seqLen] -> [seqLen, dModel] */
    public float[] lookupBatch(int[] ids) {
        float[] r = new float[ids.length * dModel];
        for (int i = 0; i < ids.length; i++) {
            System.arraycopy(weight, ids[i] * dModel, r, i * dModel, dModel);
        }
        return r;
    }

    /**
     * 反向用：用 hidden 向量与权重表点积，得到每个 token 的 logit。
     * lm_head 与 embedding 共享权重时用此方法：logit[i] = hidden · weight[i]。
     */
    public float[] projectToVocab(float[] hidden) {
        float[] logits = new float[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            int off = i * dModel;
            float s = 0f;
            for (int d = 0; d < dModel; d++) {
                s += hidden[d] * weight[off + d];
            }
            logits[i] = s;
        }
        return logits;
    }
}
