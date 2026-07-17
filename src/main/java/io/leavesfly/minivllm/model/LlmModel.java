package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * LlmModel —— 推理引擎视角的模型接口。
 *
 * 学习要点：
 * 1. 引擎（LLMEngine）只关心三件事：prefill 整段 prompt、decode 单个 token、读取配置。
 *    至于内部是 GPT-2 风格（TransformerModel）还是 Qwen3 风格（Qwen3Model），引擎无感。
 * 2. 两条路径都复用同一套 PagedAttention KV cache 语义：
 *    prefill 写入各层 BlockTable，decode 逐 block 读取累加。
 */
public interface LlmModel {

    /**
     * Prefill：处理整段 prompt，写入 KV cache，返回最后一个位置的词表 logits。
     *
     * @param tokenIds prompt 的 token id 序列
     * @param kvMgr    KV cache 管理器
     * @param bts      每层的 BlockTable
     * @param startIdx 这些 token 在序列中的起始全局下标（前缀共享时 >0）
     * @return [vocabSize] 最后一个位置的 logits
     */
    float[] prefillLogits(int[] tokenIds, KVCacheManager kvMgr, BlockTable[] bts, int startIdx);

    /**
     * Decode：处理单个新 token，复用 KV cache，返回该位置的词表 logits。
     *
     * @param tokenId 当前 token id
     * @param curIdx  当前 token 的全局下标
     * @param kvMgr   KV cache 管理器
     * @param bts     每层的 BlockTable
     * @return [vocabSize] 当前位置的 logits
     */
    float[] decodeLogits(int tokenId, int curIdx, KVCacheManager kvMgr, BlockTable[] bts);

    /**
     * 批量 Decode：一次处理 B 个并发序列各自的新 token，返回每个序列的 logits。
     *
     * 这是 vLLM 提升吞吐的核心——把 B 个序列的隐状态堆成 [B, dModel] 矩阵，
     * 所有 Linear 投影按 GEMM 计算：权重只从内存读一次、跨 B 行复用，
     * 既摊薄权重读取带宽，又摊薄单 token decode 的 fork-join 调度开销。
     * 注意力仍按序列独立（各有 KV cache 与位置）。
     *
     * 默认实现逐序列回退到 {@link #decodeLogits}（行为等价），
     * 支持批处理的模型（如 Qwen3）可覆盖为真正的批量前向。
     *
     * @param tokenIds 每个序列的当前 token id，长度 B
     * @param curIdxs  每个序列当前 token 的全局下标，长度 B
     * @param kvMgr    KV cache 管理器
     * @param bts      每个序列各层的 BlockTable，形状 [B][nLayer]
     * @return [B][vocabSize] 每个序列的 logits
     */
    default float[][] decodeLogitsBatch(int[] tokenIds, int[] curIdxs,
                                        KVCacheManager kvMgr, BlockTable[][] bts) {
        float[][] out = new float[tokenIds.length][];
        for (int i = 0; i < tokenIds.length; i++) {
            out[i] = decodeLogits(tokenIds[i], curIdxs[i], kvMgr, bts[i]);
        }
        return out;
    }

    /** 模型配置 */
    ModelConfig config();

    /** 参数总量（tied embeddings 时 lm_head 不重复计入） */
    long numParameters();
}
