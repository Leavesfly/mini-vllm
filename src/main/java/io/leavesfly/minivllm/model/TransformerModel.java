package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.LayerNorm;
import io.leavesfly.minivllm.math.Tensor;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * TransformerModel —— 完整的 GPT-2 / GPT-3 风格 Decoder-Only 模型。
 *
 * 前向流程：
 *   x = wte(tokens) + wpe(positions)        // 词嵌入 + 位置嵌入
 *   for block in blocks: x = block(x)       // N 层 TransformerBlock（GPT-3 交替 dense/sparse 注意力）
 *   x = LayerNorm(x)                         // 最终归一化
 *   logits = lm_head(x)                      // 投影到词表（tied 时复用 wte）
 *
 * 两套对外接口：
 * A. PyTorch 风格（无状态，无 KV cache）：
 *      Tensor logits = model.forward(inputIds);   // [seqLen, vocabSize]
 *    等价于 HuggingFace `model(input_ids).logits`，便于整段前向、与 PyTorch 结果对拍、教学。
 * B. 推理引擎路径（PagedAttention KV cache）：
 *      prefillLogits(...) 处理整段 prompt 并返回最后位置 logits；
 *      decodeLogits(...) 逐 token 推进并复用 KV cache，返回当前位置 logits。
 *    二者在 dense 层数值一致（forward 与 prefill 共用同一注意力核心）。
 *    内部的 prefill/decode/lastRow/logits 为实现细节，不对外暴露。
 *
 * 学习要点：
 * 1. prefill 一次性处理整段 prompt，返回每个位置的隐状态；生成只需最后一个位置。
 * 2. decode 逐 token 推进，每步复用 KV cache（散落在物理 block 中，经 PagedAttention 读取）。
 * 3. tied embeddings：lm_head 与 wte 共享权重，省一半参数（GPT-2/GPT-3 默认开启）。
 * 4. GPT-3 相对 GPT-2 的唯一架构差异是交替的 dense / locally banded sparse 注意力，
 *    由 ModelConfig.useSparseAttention 控制，逐层在 ModelLoader 里装配。
 */
public final class TransformerModel implements LlmModel {

    private final ModelConfig cfg;
    private final Embedding wte; // 词嵌入 [vocab, d]
    private final Embedding wpe; // 位置嵌入 [maxSeqLen, d]
    private final TransformerBlock[] blocks;
    private final LayerNorm lnF;
    private final int dModel;

    public TransformerModel(ModelConfig cfg, Embedding wte, Embedding wpe,
                            TransformerBlock[] blocks, LayerNorm lnF) {
        this.cfg = cfg;
        this.wte = wte;
        this.wpe = wpe;
        this.blocks = blocks;
        this.lnF = lnF;
        this.dModel = cfg.dModel();
    }

    public int dModel() {
        return dModel;
    }

    public int vocabSize() {
        return cfg.vocabSize();
    }

    @Override
    public ModelConfig config() {
        return cfg;
    }

    // ─── 推理引擎对外接口（PagedAttention KV cache） ───

    /**
     * Prefill（引擎接口）：处理整段 prompt，写入 KV cache，返回最后一个位置的词表 logits。
     *
     * 内部完成 词嵌入 → N 层 TransformerBlock → 最终 LayerNorm → 取最后位置隐状态 → 投影词表，
     * 正是引擎接纳新请求时的一次性前向。
     *
     * @param tokenIds prompt 的 token id 序列
     * @param startIdx 这些 token 在序列中的起始全局下标（前缀共享时 >0）
     * @return [vocabSize] 最后一个位置的 logits（下一个 token 的分布）
     */
    @Override
    public float[] prefillLogits(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
        float[] hidden = prefill(tokenIds, kvMgr, bts, startIdx);
        return logits(lastRow(hidden, tokenIds.length));
    }

    /**
     * Decode（引擎接口）：处理单个新 token，复用 KV cache，返回该位置的词表 logits。
     *
     * @param tokenId 当前 token id
     * @param curIdx  当前 token 的全局下标
     * @return [vocabSize] 当前位置的 logits（下一个 token 的分布）
     */
    @Override
    public float[] decodeLogits(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
        return logits(decode(tokenId, curIdx, kvMgr, bts));
    }

    // ─── 内部前向步骤（实现细节，不对外暴露） ───

    /**
     * Prefill：处理整段 prompt。
     *
     * @param tokenIds prompt 的 token id 序列
     * @param startIdx 这些 token 在序列中的起始全局下标（前缀共享时 >0）
     * @return [seqLen, dModel] 所有位置的最终隐状态
     */
    private float[] prefill(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx) {
        int seqLen = tokenIds.length;
        // 词嵌入 + 位置嵌入
        float[] x = embed(tokenIds, startIdx); // [seqLen, d]
        // 逐层 TransformerBlock（每层用各自的 BlockTable，因为每层有独立 KV cache）
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
    private float[] decode(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts) {
        float[] x = wte.lookup(tokenId); // [d]
        int pOff = curIdx * dModel;
        for (int d = 0; d < dModel; d++) {
            x[d] += wpe.weight()[pOff + d];
        }
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].decode(x, curIdx, kvMgr, bts[i]);
        }
        lnF.forwardInPlace(x);
        return x; // [d]
    }

    /** 从完整隐状态中取最后一行（用于生成下一个 token） */
    private float[] lastRow(float[] allHidden, int seqLen) {
        float[] r = new float[dModel];
        System.arraycopy(allHidden, (seqLen - 1) * dModel, r, 0, dModel);
        return r;
    }

    /**
     * 隐状态 -> 词表 logits。
     * tied embeddings 时复用 wte 权重做投影。
     */
    private float[] logits(float[] hidden) {
        return wte.projectToVocab(hidden);
    }

    // ─── PyTorch 风格对外接口 ───

    /**
     * PyTorch 风格前向：输入 token id 序列，返回每个位置的词表 logits。
     * 等价于 HuggingFace `logits = model(input_ids).logits`。
     *
     * 这是一次「无状态、无 KV cache」的稠密前向，内部走标准 causal（含 GPT-3 稀疏层）注意力，
     * 适合整段前向、与 PyTorch 结果对拍、教学演示；在线推理请用 prefill/decode。
     *
     * @param inputIds [seqLen] token id
     * @return Tensor，shape=[seqLen, vocabSize]
     */
    public Tensor forward(int[] inputIds) {
        int seqLen = inputIds.length;
        float[] x = embed(inputIds, 0);
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].forward(x, seqLen);
        }
        lnF.forwardRowsInPlace(x, seqLen);
        float[] logits = wte.projectToVocabBatch(x, seqLen);
        return new Tensor(logits, seqLen, cfg.vocabSize());
    }

    /**
     * PyTorch 风格前向（Tensor 版）：inputIds 为 [seqLen]（token id，以 float 存储），
     * 返回 [seqLen, vocabSize] logits。
     */
    public Tensor forward(Tensor inputIds) {
        int n = inputIds.size();
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = (int) inputIds.data()[i];
        }
        return forward(ids);
    }

    /** 便捷：整段前向后仅取最后一个位置的 logits（下一个 token 的分布） */
    public float[] forwardLastLogits(int[] inputIds) {
        Tensor l = forward(inputIds);
        int seqLen = inputIds.length;
        float[] last = new float[cfg.vocabSize()];
        System.arraycopy(l.data(), (seqLen - 1) * cfg.vocabSize(), last, 0, cfg.vocabSize());
        return last;
    }

    /**
     * 参数总量（PyTorch: sum(p.numel())）。
     * tied embeddings 下 lm_head 复用 wte，不重复计入。
     */
    public long numParameters() {
        long n = wte.numParameters() + wpe.numParameters();
        for (TransformerBlock b : blocks) {
            n += b.numParameters();
        }
        n += lnF.numParameters();
        return n;
    }

    /** 层数 */
    public int numLayers() {
        return blocks.length;
    }

    // ─── 内部工具 ───

    /** 词嵌入 + 位置嵌入：ids[seqLen] -> x[seqLen, dModel]，位置从 startIdx 起 */
    private float[] embed(int[] ids, int startIdx) {
        int seqLen = ids.length;
        float[] x = wte.lookupBatch(ids); // [seqLen, d]
        for (int t = 0; t < seqLen; t++) {
            int pOff = (startIdx + t) * dModel;
            int xOff = t * dModel;
            for (int d = 0; d < dModel; d++) {
                x[xOff + d] += wpe.weight()[pOff + d];
            }
        }
        return x;
    }
}
