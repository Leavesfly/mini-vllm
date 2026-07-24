package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Bf16;
import io.leavesfly.minivllm.math.Matmul;

/**
 * 嵌入层 —— 把 token id 映射为稠密向量，是 TransformerModel 的输入入口。
 *
 * 学习要点：
 * 1. 本质是一张查表：weight 形状 [vocabSize, dModel]，第 id 行就是 token id 的向量。
 * 2. GPT-2 中词嵌入与 lm_head 共享权重（tieWordEmbeddings），即用同一张表做输入查表和输出投影。
 * 3. 支持 F32 / BF16 / INT8 三种常驻精度。tied 下这张表既是最大的权重（vocab×d），
 *    又兼作 lm_head——int8 量化可使 lm_head 带宽降至 bf16 的一半。
 */
public final class Embedding {

    private final float[] weight;        // [vocabSize, dModel] 行优先；bf16/int8 模式为 null
    private final short[] weightBf16;    // bf16 位版；f32/int8 模式为 null
    private final byte[] weightInt8;     // int8 量化版；f32/bf16 模式为 null
    private final float[] scaleInt8;     // int8 per-row scale [vocabSize]；非 int8 为 null
    private final int vocabSize;
    private final int dModel;

    public Embedding(float[] weight, int vocabSize, int dModel) {
        this(weight, null, null, null, vocabSize, dModel);
    }

    private Embedding(float[] weight, short[] weightBf16, byte[] weightInt8, float[] scaleInt8,
                      int vocabSize, int dModel) {
        this.weight = weight;
        this.weightBf16 = weightBf16;
        this.weightInt8 = weightInt8;
        this.scaleInt8 = scaleInt8;
        this.vocabSize = vocabSize;
        this.dModel = dModel;
    }

    /** BF16 位权重常驻构造 */
    public static Embedding ofBf16(short[] weightBf16, int vocabSize, int dModel) {
        return new Embedding(null, weightBf16, null, null, vocabSize, dModel);
    }

    /** INT8 量化权重构造 */
    public static Embedding ofInt8(byte[] weightInt8, float[] scaleInt8, int vocabSize, int dModel) {
        return new Embedding(null, null, weightInt8, scaleInt8, vocabSize, dModel);
    }

    // ─── 访问器（与 Linear 风格对齐） ───

    public float[] weight() { return weight; }
    public short[] weightBf16() { return weightBf16; }
    public byte[] weightInt8() { return weightInt8; }
    public float[] scaleInt8() { return scaleInt8; }
    public int vocabSize() { return vocabSize; }
    public int dModel() { return dModel; }

    /** 是否为 bf16 常驻权重 */
    public boolean isBf16() {
        return weightBf16 != null;
    }

    /** 是否为 int8 量化权重 */
    public boolean isInt8() {
        return weightInt8 != null;
    }

    /** 查单个 token 的向量（返回拷贝） */
    public float[] lookup(int id) {
        float[] r = new float[dModel];
        if (weightInt8 != null) {
            int off = id * dModel;
            float s = scaleInt8[id];
            for (int d = 0; d < dModel; d++) {
                r[d] = weightInt8[off + d] * s;
            }
        } else if (weightBf16 != null) {
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
            if (weightInt8 != null) {
                int off = ids[i] * dModel;
                int dst = i * dModel;
                float s = scaleInt8[ids[i]];
                for (int d = 0; d < dModel; d++) {
                    r[dst + d] = weightInt8[off + d] * s;
                }
            } else if (weightBf16 != null) {
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
        if (weightInt8 != null) {
            return Matmul.matVecInt8(weightInt8, scaleInt8, hidden, vocabSize, dModel);
        } else if (weightBf16 != null) {
            return Matmul.matVecBf16(weightBf16, hidden, vocabSize, dModel);
        } else {
            return Matmul.matVec(weight, hidden, vocabSize, dModel);
        }
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

    /**
     * 融合批量 lm_head（性能优化）：hidden[B, dModel] -> float[B][vocabSize]。
     * 一次 parallelRows 按词表行并行，每行同时算 B 个序列的 logit，
     * 权重行只读一遍即复用到全部 B 个 hidden，且仅 1 次 fork-join（而非 B 次）。
     * 用于 decodeLogitsBatch 的 lm_head 阶段，B≥2 时显著降低调度开销与带宽。
     */
    public float[][] projectToVocabFused(float[] hidden, int batch) {
        float[][] out = new float[batch][vocabSize];
        boolean int8 = weightInt8 != null;
        boolean bf16 = weightBf16 != null;
        Matmul.parallelRows(vocabSize, v -> {
            int wOff = v * dModel;
            for (int b = 0; b < batch; b++) {
                float dot;
                if (int8) {
                    dot = Matmul.dotInt8(weightInt8, wOff, hidden, b * dModel, dModel, scaleInt8[v]);
                } else if (bf16) {
                    dot = Matmul.dotBf16(weightBf16, wOff, hidden, b * dModel, dModel);
                } else {
                    dot = Matmul.dot(weight, wOff, hidden, b * dModel, dModel);
                }
                out[b][v] = dot;
            }
        });
        return out;
    }

    /** 参数量（vocabSize * dModel） */
    public long numParameters() {
        return weight != null ? weight.length : (weightBf16 != null ? weightBf16.length : weightInt8.length);
    }
}
