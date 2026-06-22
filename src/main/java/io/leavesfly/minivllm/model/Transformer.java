package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.LayerNorm;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * Transformer —— 完整的 GPT-2 风格 Decoder-Only 模型。
 *
 * 前向流程：
 *   x = wte(tokens) + wpe(positions)        // 词嵌入 + 位置嵌入
 *   for block in blocks: x = block(x)       // N 层 Transformer Block
 *   x = LayerNorm(x)                         // 最终归一化
 *   logits = lm_head(x)                      // 投影到词表（tied 时复用 wte）
 *
 * 学习要点：
 * 1. prefill 一次性处理整段 prompt，返回每个位置的隐状态；生成只需最后一个位置。
 * 2. decode 逐 token 推进，每步复用 KV cache（散落在物理 block 中，经 PagedAttention 读取）。
 * 3. tied embeddings：lm_head 与 wte 共享权重，省一半参数（GPT-2 默认开启）。
 */
public final class Transformer {

    private final ModelConfig cfg;
    private final Embedding wte; // 词嵌入 [vocab, d]
    private final Embedding wpe; // 位置嵌入 [maxSeqLen, d]
    private final TransformerBlock[] blocks;
    private final LayerNorm lnF;
    private final int dModel;

    public Transformer(ModelConfig cfg, Embedding wte, Embedding wpe,
                       TransformerBlock[] blocks, LayerNorm lnF) {
        this.cfg = cfg;
        this.wte = wte;
        this.wpe = wpe;
        this.blocks = blocks;
        this.lnF = lnF;
        this.dModel = cfg.dModel;
    }

    public int dModel() {
        return dModel;
    }

    public int vocabSize() {
        return cfg.vocabSize;
    }

    public ModelConfig config() {
        return cfg;
    }

    /**
     * Prefill：处理整段 prompt。
     *
     * @param tokenIds prompt 的 token id 序列
     * @param startIdx 这些 token 在序列中的起始全局下标（前缀共享时 >0）
     * @return [seqLen, dModel] 所有位置的最终隐状态
     */
    public float[] prefill(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
        int seqLen = tokenIds.length;
        // 词嵌入 + 位置嵌入
        float[] x = wte.lookupBatch(tokenIds); // [seqLen, d]
        for (int t = 0; t < seqLen; t++) {
            int pos = startIdx + t;
            int pOff = pos * dModel;
            int xOff = t * dModel;
            for (int d = 0; d < dModel; d++) {
                x[xOff + d] += wpe.weight[pOff + d];
            }
        }
        // 逐层 Transformer Block（每层用各自的 BlockTable，因为每层有独立 KV cache）
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].prefill(x, seqLen, kvMgr, bts[i], startIdx);
        }
        // 最终 LayerNorm
        lnF.forwardRowsInPlace(x, seqLen);
        return x; // [seqLen, d]
    }

    /**
     * Decode：处理单个新 token。
     *
     * @param tokenId 当前 token id
     * @param curIdx  当前 token 的全局下标
     * @return [dModel] 当前位置的最终隐状态
     */
    public float[] decode(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
        float[] x = wte.lookup(tokenId); // [d]
        int pOff = curIdx * dModel;
        for (int d = 0; d < dModel; d++) {
            x[d] += wpe.weight[pOff + d];
        }
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].decode(x, curIdx, kvMgr, bts[i]);
        }
        lnF.forwardInPlace(x);
        return x; // [d]
    }

    /** 从完整隐状态中取最后一行（用于生成下一个 token） */
    public float[] lastRow(float[] allHidden, int seqLen) {
        float[] r = new float[dModel];
        System.arraycopy(allHidden, (seqLen - 1) * dModel, r, 0, dModel);
        return r;
    }

    /**
     * 隐状态 -> 词表 logits。
     * tied embeddings 时复用 wte 权重做投影。
     */
    public float[] logits(float[] hidden) {
        return wte.projectToVocab(hidden);
    }
}
