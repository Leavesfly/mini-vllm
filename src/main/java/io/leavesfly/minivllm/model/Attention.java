package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.math.Softmax;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * 多头自注意力 —— Transformer 的核心，也是 PagedAttention 嵌入的位置。
 *
 * 学习要点：
 * 1. 每个头独立做 attention：Q·Kᵀ/√d → softmax → ·V。多头结果拼接后过输出投影。
 * 2. prefill 阶段一次性处理整段 prompt，此时 K/V 在手边是连续的，用标准 causal attention，
 *    同时把每个 token 的 K/V 写入物理 block，供后续 decode 复用。
 * 3. decode 阶段只算 1 个新 token 的 Q，历史 K/V 全在 BlockTable 指向的散落 block 里。
 *    这里用两种方式实现，对比学习 PagedAttention 的本质：
 *      - decodePaged：逐 block 读取 K/V 并累加 attention（无拷贝，真正 PagedAttention 语义）
 *      - decodeGather：先把散落 block 的 K/V 收集成连续数组再做标准 attention（有拷贝，便于理解）
 *    真实 vLLM 的 PagedAttention CUDA kernel 走的是 decodePaged 思路。
 */
public final class Attention {

    private final ModelConfig cfg;
    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear oProj;

    private final int nHead;
    private final int headDim;
    private final int dModel;
    private final int blockSize;

    public Attention(ModelConfig cfg, Linear qProj, Linear kProj, Linear vProj, Linear oProj) {
        this.cfg = cfg;
        this.qProj = qProj;
        this.kProj = kProj;
        this.vProj = vProj;
        this.oProj = oProj;
        this.nHead = cfg.nHead;
        this.headDim = cfg.headDim();
        this.dModel = cfg.dModel;
        this.blockSize = cfg.blockSize;
    }

    /**
     * Prefill：处理一段 prompt token。
     *
     * @param input   [seqLen, dModel] 嵌入后的输入
     * @param seqLen  prompt 长度
     * @param kvMgr   KV cache 管理器
     * @param bt      本请求的 BlockTable
     * @param startIdx 这些 token 在序列中的起始全局下标（前缀共享时 >0）
     * @return [seqLen, dModel] attention 输出
     */
    public float[] prefill(float[] input, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
        // 1. 投影 Q/K/V
        float[] q = qProj.forwardBatch(input, seqLen); // [seqLen, dModel]
        float[] k = kProj.forwardBatch(input, seqLen);
        float[] v = vProj.forwardBatch(input, seqLen);

        // 2. 写入 KV cache（每个 token 的 K/V 落到物理 block）
        float[] kt = new float[dModel];
        float[] vt = new float[dModel];
        for (int t = 0; t < seqLen; t++) {
            System.arraycopy(k, t * dModel, kt, 0, dModel);
            System.arraycopy(v, t * dModel, vt, 0, dModel);
            kvMgr.writeKV(bt, startIdx + t, kt, vt);
        }

        // 3. causal self-attention（此时 K/V 连续在手，直接算最高效）
        float[] out = new float[seqLen * dModel];
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            for (int i = 0; i < seqLen; i++) {
                int qOff = i * dModel + hOff;
                float[] scores = new float[i + 1]; // 只到 i（causal）
                for (int j = 0; j <= i; j++) {
                    int kOff = j * dModel + hOff;
                    float s = 0f;
                    for (int d = 0; d < headDim; d++) {
                        s += q[qOff + d] * k[kOff + d];
                    }
                    scores[j] = s * invSqrt;
                }
                Softmax.softmaxInPlace(scores);
                int oOff = i * dModel + hOff;
                for (int d = 0; d < headDim; d++) out[oOff + d] = 0f;
                for (int j = 0; j <= i; j++) {
                    int vOff = j * dModel + hOff;
                    float w = scores[j];
                    for (int d = 0; d < headDim; d++) {
                        out[oOff + d] += w * v[vOff + d];
                    }
                }
            }
        }
        // 4. 输出投影
        return oProj.forwardBatch(out, seqLen);
    }

    /**
     * Decode（PagedAttention block-wise 累加）：处理单个新 token。
     *
     * @param hidden  [dModel] 当前 token 的隐状态
     * @param curIdx  当前 token 的全局下标
     * @return [dModel] attention 输出
     *
     * 关键：历史 K/V 散落在 BlockTable 指向的多个物理 block 中，
     * 这里逐 block 读取并累加 attention 分数与输出，不做全局拷贝——
     * 这正是 PagedAttention 的核心语义。
     */
    public float[] decodePaged(float[] hidden, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        float[] q = qProj.forward(hidden); // [dModel]
        float[] k = kProj.forward(hidden);
        float[] v = vProj.forward(hidden);
        kvMgr.writeKV(bt, curIdx, k, v); // 当前 token 的 K/V 入 block

        int totalTokens = curIdx + 1; // 0..curIdx
        int nBlocks = bt.numBlocks();
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        float[] out = new float[dModel];

        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            // 第一遍：逐 block 累加所有历史 token 的 attention scores
            float[] scores = new float[totalTokens];
            int idx = 0;
            for (int b = 0; b < nBlocks; b++) {
                float[] kBlk = kvMgr.blockK(bt, b); // [blockSize, dModel]
                int remain = totalTokens - b * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int kOff = s * dModel + hOff;
                    float sc = 0f;
                    for (int d = 0; d < headDim; d++) {
                        sc += q[hOff + d] * kBlk[kOff + d];
                    }
                    scores[idx++] = sc * invSqrt;
                }
            }
            Softmax.softmaxInPlace(scores);
            // 第二遍：逐 block 加权 V 累加输出
            idx = 0;
            for (int b = 0; b < nBlocks; b++) {
                float[] vBlk = kvMgr.blockV(bt, b);
                int remain = totalTokens - b * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int vOff = s * dModel + hOff;
                    float w = scores[idx++];
                    for (int d = 0; d < headDim; d++) {
                        out[hOff + d] += w * vBlk[vOff + d];
                    }
                }
            }
        }
        return oProj.forward(out);
    }

    /**
     * Decode（gather 版）：先把散落 block 的 K/V 收集成连续数组，再做标准 attention。
     *
     * 对比 decodePaged：逻辑更直观，但有 O(totalTokens·dModel) 的拷贝开销；
     * decodePaged 无拷贝、更接近真实 PagedAttention kernel。两者数学结果一致。
     * 学习时可用断言对比两者输出验证正确性。
     */
    public float[] decodeGather(float[] hidden, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        float[] q = qProj.forward(hidden);
        float[] k = kProj.forward(hidden);
        float[] v = vProj.forward(hidden);
        kvMgr.writeKV(bt, curIdx, k, v);

        int totalTokens = curIdx + 1;
        // gather 散落 block 的 K/V 成连续数组
        float[] kAll = new float[totalTokens * dModel];
        float[] vAll = new float[totalTokens * dModel];
        int idx = 0;
        int nBlocks = bt.numBlocks();
        for (int b = 0; b < nBlocks; b++) {
            float[] kBlk = kvMgr.blockK(bt, b);
            float[] vBlk = kvMgr.blockV(bt, b);
            int remain = totalTokens - b * blockSize;
            int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
            for (int s = 0; s < tokensInBlock; s++) {
                System.arraycopy(kBlk, s * dModel, kAll, idx * dModel, dModel);
                System.arraycopy(vBlk, s * dModel, vAll, idx * dModel, dModel);
                idx++;
            }
        }
        // 标准 attention（decode 时当前 token 可见全部历史）
        float[] out = new float[dModel];
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            float[] scores = new float[totalTokens];
            for (int j = 0; j < totalTokens; j++) {
                int kOff = j * dModel + hOff;
                float s = 0f;
                for (int d = 0; d < headDim; d++) {
                    s += q[hOff + d] * kAll[kOff + d];
                }
                scores[j] = s * invSqrt;
            }
            Softmax.softmaxInPlace(scores);
            for (int j = 0; j < totalTokens; j++) {
                int vOff = j * dModel + hOff;
                float w = scores[j];
                for (int d = 0; d < headDim; d++) {
                    out[hOff + d] += w * vAll[vOff + d];
                }
            }
        }
        return oProj.forward(out);
    }
}
