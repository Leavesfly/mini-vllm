package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.math.Tensor;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

import java.util.function.BiConsumer;

/**
 * Qwen3Model —— 完整的 Qwen3 Decoder-Only 模型（与 TransformerModel 平行的第二套实现）。
 *
 * 前向流程：
 *   x = embed_tokens(tokens)              // 仅词嵌入，无位置嵌入（位置由 RoPE 提供）
 *   for block in blocks: x = block(x)     // N 层 Qwen3Block（GQA + QK-Norm + RoPE + SwiGLU）
 *   x = RmsNorm(x)                        // model.norm
 *   logits = x · embed_tokensᵀ            // tied lm_head（Qwen3-0.6B tie_word_embeddings=true）
 *
 * 两套对外接口与 {@link TransformerModel} 对齐：
 * A. PyTorch 风格：Tensor logits = model.forward(inputIds)   // [seqLen, vocabSize]
 * B. 推理引擎路径（PagedAttention KV cache）：prefillLogits / decodeLogits。
 *
 * 学习要点：
 * 1. 无 wpe：位置信息完全由注意力内部的 RoPE 提供，嵌入层只做查表。
 * 2. tied embeddings：lm_head 复用 embed_tokens，logits = hidden · embedᵀ。
 * 3. layerTrace 调试钩子：逐层 dump hidden states，用于与 HF 参考输出逐层对齐。
 */
public final class Qwen3Model implements LlmModel {

    private final ModelConfig cfg;
    private final Embedding wte; // embed_tokens [vocab, dModel]，tied 时兼作 lm_head
    private final Qwen3Block[] blocks;
    private final RmsNorm lnF;   // model.norm
    private final int dModel;

    /** debug：非 null 时，每层 block 输出后回调（layerIdx, hidden 拷贝） */
    public volatile BiConsumer<Integer, float[]> layerTrace = null;

    public Qwen3Model(ModelConfig cfg, Embedding wte, Qwen3Block[] blocks, RmsNorm lnF) {
        this.cfg = cfg;
        this.wte = wte;
        this.blocks = blocks;
        this.lnF = lnF;
        this.dModel = cfg.dModel;
    }

    @Override
    public ModelConfig config() {
        return cfg;
    }

    public int dModel() {
        return dModel;
    }

    public int vocabSize() {
        return cfg.vocabSize;
    }

    public int numLayers() {
        return blocks.length;
    }

    // ===================== 推理引擎接口（PagedAttention KV cache） =====================

    @Override
    public float[] prefillLogits(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
        float[] hidden = prefill(tokenIds, kvMgr, bts, startIdx);
        return logits(lastRow(hidden, tokenIds.length));
    }

    @Override
    public float[] decodeLogits(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
        return logits(decode(tokenId, curIdx, kvMgr, bts));
    }

    @Override
    public float[][] decodeLogitsBatch(int[] tokenIds, int[] curIdxs,
                                       KVCacheManager kvMgr, BlockTable[][] bts) {
        int batch = tokenIds.length;
        if (batch == 1) {
            // 单序列走原路径，避免批量开销
            return new float[][]{decodeLogits(tokenIds[0], curIdxs[0], kvMgr, bts[0])};
        }
        // 1. 批量词嵌入：[B, dModel]
        float[] x = new float[batch * dModel];
        for (int b = 0; b < batch; b++) {
            float[] row = wte.lookup(tokenIds[b]);
            System.arraycopy(row, 0, x, b * dModel, dModel);
        }
        // 2. 逐层批量前向（每层收集各序列的本层 BlockTable）
        BlockTable[] layerBts = new BlockTable[batch];
        for (int i = 0; i < blocks.length; i++) {
            for (int b = 0; b < batch; b++) {
                layerBts[b] = bts[b][i];
            }
            x = blocks[i].decodeBatch(x, batch, curIdxs, kvMgr, layerBts);
            trace(i, x);
        }
        lnF.forwardRowsInPlace(x, batch);
        // 3. 逐序列 lm_head 投影
        float[][] out = new float[batch][];
        float[] row = new float[dModel];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(x, b * dModel, row, 0, dModel);
            out[b] = logits(row);
        }
        return out;
    }

    // ===================== 内部前向 =====================

    private float[] prefill(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
        int seqLen = tokenIds.length;
        float[] x = wte.lookupBatch(tokenIds); // [seqLen, d]，无位置嵌入
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].prefill(x, seqLen, kvMgr, bts[i], startIdx);
            trace(i, x);
        }
        lnF.forwardRowsInPlace(x, seqLen);
        return x;
    }

    private float[] decode(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
        float[] x = wte.lookup(tokenId); // [d]
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].decode(x, curIdx, kvMgr, bts[i]);
            trace(i, x);
        }
        lnF.forwardInPlace(x);
        return x;
    }

    private float[] lastRow(float[] allHidden, int seqLen) {
        float[] r = new float[dModel];
        System.arraycopy(allHidden, (seqLen - 1) * dModel, r, 0, dModel);
        return r;
    }

    /** tied lm_head：logits = hidden · embed_tokensᵀ */
    private float[] logits(float[] hidden) {
        return wte.projectToVocab(hidden);
    }

    private void trace(int layer, float[] x) {
        BiConsumer<Integer, float[]> t = layerTrace;
        if (t != null) {
            t.accept(layer, x.clone());
        }
    }

    // ===================== PyTorch 风格接口 =====================

    /**
     * PyTorch 风格前向：输入 token id 序列，返回每个位置的词表 logits。
     * 等价于 HuggingFace `model(input_ids).logits`（无状态、无 KV cache）。
     */
    public Tensor forward(int[] inputIds) {
        int seqLen = inputIds.length;
        float[] x = wte.lookupBatch(inputIds);
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].forward(x, seqLen);
            trace(i, x);
        }
        lnF.forwardRowsInPlace(x, seqLen);
        float[] logits = wte.projectToVocabBatch(x, seqLen);
        return new Tensor(logits, seqLen, cfg.vocabSize);
    }

    /** 便捷：整段前向后仅取最后一个位置的 logits */
    public float[] forwardLastLogits(int[] inputIds) {
        Tensor l = forward(inputIds);
        int seqLen = inputIds.length;
        float[] last = new float[cfg.vocabSize];
        System.arraycopy(l.data, (seqLen - 1) * cfg.vocabSize, last, 0, cfg.vocabSize);
        return last;
    }

    /**
     * 便捷：整段前向后仅取最后一个位置、final norm 之后的 hidden（hs[-1]）。
     * 与 {@link #forward} 同路径，但止于 model.norm 之前的 lm_head 投影，用于逐层对齐验证。
     */
    public float[] forwardLastHidden(int[] inputIds) {
        int seqLen = inputIds.length;
        float[] x = wte.lookupBatch(inputIds);
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].forward(x, seqLen);
            trace(i, x);
        }
        lnF.forwardRowsInPlace(x, seqLen);
        float[] last = new float[dModel];
        System.arraycopy(x, (seqLen - 1) * dModel, last, 0, dModel);
        return last;
    }

    /** 参数总量（tied 下 lm_head 不重复计入） */
    public long numParameters() {
        long n = wte.numParameters();
        for (Qwen3Block b : blocks) {
            n += b.numParameters();
        }
        n += lnF.numParameters();
        return n;
    }
}
