package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.math.Softmax;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * 多头自注意力 —— TransformerModel 的核心，也是 PagedAttention 嵌入的位置。
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
 * 4. GPT-3 支持：sparse 层（局部带状）下，每个 query 仅关注最近 window 个 token，
 *    prefill / decodePaged / decodeGather 三条路径都遵守同一窗口约束。
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
    private final boolean sparse; // 是否为 GPT-3 局部带状稀疏注意力层
    private final int window;     // 稀疏窗口：每个 query 仅关注最近 window 个 token

    /** dense 注意力（GPT-2 风格）便捷构造 */
    public Attention(ModelConfig cfg, Linear qProj, Linear kProj, Linear vProj, Linear oProj) {
        this(cfg, qProj, kProj, vProj, oProj, false, Integer.MAX_VALUE);
    }

    /**
     * @param sparse 是否为局部带状稀疏注意力层（GPT-3 交替模式下的奇数层）
     * @param window 稀疏窗口大小（dense 层传 Integer.MAX_VALUE 即可）
     */
    public Attention(ModelConfig cfg, Linear qProj, Linear kProj, Linear vProj, Linear oProj,
                     boolean sparse, int window) {
        this.cfg = cfg;
        this.qProj = qProj;
        this.kProj = kProj;
        this.vProj = vProj;
        this.oProj = oProj;
        this.nHead = cfg.nHead();
        this.headDim = cfg.headDim();
        this.dModel = cfg.dModel();
        this.blockSize = cfg.blockSize();
        this.sparse = sparse;
        this.window = window <= 0 ? Integer.MAX_VALUE : window;
    }

    /** 该层是否为局部带状稀疏注意力 */
    public boolean isSparse() {
        return sparse;
    }

    /** 参数量（q/k/v/o 四个投影） */
    public long numParameters() {
        return qProj.numParameters() + kProj.numParameters()
                + vProj.numParameters() + oProj.numParameters();
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

        // 3. causal（可选局部带状 sparse）self-attention：startIdx>0（前缀共享/分块 prefill）时，
        //    query 还需注意 BlockTable 中已缓存的历史 KV，走融合路径
        float[] out = startIdx > 0
                ? causalAttentionWithPrefix(q, k, v, seqLen, startIdx, kvMgr, bt)
                : causalAttention(q, k, v, seqLen);
        // 4. 输出投影
        return oProj.forwardBatch(out, seqLen);
    }

    /**
     * causal（可选局部带状 sparse）多头注意力核心。
     * q/k/v: [seqLen, dModel] -> 返回 [seqLen, dModel]。
     * sparse 层：query i 仅关注 [i-window+1, i] 的 key（局部带状）；dense 层：关注 [0, i]。
     * prefill 与 PyTorch 风格 forward 共用此核心，保证两条路径数值一致。
     */
    private float[] causalAttention(float[] q, float[] k, float[] v, int seqLen) {
        float[] out = new float[seqLen * dModel];
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            for (int i = 0; i < seqLen; i++) {
                int lo = sparse ? Math.max(0, i - window + 1) : 0; // 稀疏层只看最近 window 个 token
                int qOff = i * dModel + hOff;
                float[] scores = new float[i - lo + 1];
                for (int j = lo; j <= i; j++) {
                    int kOff = j * dModel + hOff;
                    scores[j - lo] = Matmul.dot(q, qOff, k, kOff, headDim) * invSqrt;
                }
                Softmax.softmaxInPlace(scores);
                int oOff = i * dModel + hOff;
                for (int j = lo; j <= i; j++) {
                    int vOff = j * dModel + hOff;
                    float w = scores[j - lo];
                    for (int d = 0; d < headDim; d++) {
                        out[oOff + d] += w * v[vOff + d];
                    }
                }
            }
        }
        return out;
    }

    /**
     * 带缓存前缀的 causal 多头注意力（前缀共享 / 分块 prefill 专用）。
     *
     * query i（全局位置 startIdx+i）需注意两部分 KV：
     *   1. 已缓存前缀：BlockTable 中位置 0..startIdx-1 的 KV（逐 block 分页读取）
     *   2. chunk 内 causal：本 chunk 位置 0..i 的连续 KV
     * 两部分用 Online Softmax 融合为单遍扫描；sparse 层的窗口约束与
     * {@link #causalAttention} 一致（query 仅看最近 window 个 token）。
     */
    private float[] causalAttentionWithPrefix(float[] q, float[] k, float[] v, int seqLen,
                                              int startIdx, KVCacheManager kvMgr, BlockTable bt) {
        float[] out = new float[seqLen * dModel];
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        final boolean int8 = kvMgr.isInt8();
        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            for (int i = 0; i < seqLen; i++) {
                int globalPos = startIdx + i;
                int lo = sparse ? Math.max(0, globalPos - window + 1) : 0;
                int qOff = i * dModel + hOff;

                // Online Softmax 状态
                float maxScore = Float.NEGATIVE_INFINITY;
                float sumExp = 0f;
                float[] acc = new float[headDim];

                // 1. 已缓存前缀（0..startIdx-1）：逐 block、逐 token 扫描，窗口外跳过
                int nBlocks = startIdx / blockSize + (startIdx % blockSize == 0 ? 0 : 1);
                for (int blk = 0; blk < nBlocks; blk++) {
                    java.nio.ByteBuffer kBuf = int8 ? null : kvMgr.blockK(bt, blk);
                    java.nio.ByteBuffer vBuf = int8 ? null : kvMgr.blockV(bt, blk);
                    byte[] k8 = int8 ? kvMgr.blockK8(bt, blk) : null;
                    byte[] v8 = int8 ? kvMgr.blockV8(bt, blk) : null;
                    float[] kScale = int8 ? kvMgr.blockKScale(bt, blk) : null;
                    float[] vScale = int8 ? kvMgr.blockVScale(bt, blk) : null;
                    int slots = Math.min(blockSize, startIdx - blk * blockSize);
                    for (int s = 0; s < slots; s++) {
                        int gid = blk * blockSize + s;
                        if (gid < lo) continue;
                        int off = s * dModel + hOff;
                        // INT8 模式：量化域点积×scale，与反量化后点积等价（up to 量化误差）
                        float score = int8
                                ? Matmul.dotInt8(k8, off, q, qOff, headDim, kScale[s]) * invSqrt
                                : Matmul.dot(q, qOff, kBuf, off * 4, headDim) * invSqrt;
                        if (score > maxScore) {
                            float correction = (float) Math.exp(maxScore - score);
                            sumExp *= correction;
                            for (int d = 0; d < headDim; d++) {
                                acc[d] *= correction;
                            }
                            maxScore = score;
                        }
                        float w = (float) Math.exp(score - maxScore);
                        sumExp += w;
                        if (int8) {
                            Matmul.axpyInt8(w, v8, off, acc, 0, headDim, vScale[s]);
                        } else {
                            Matmul.axpy(w, vBuf, off * 4, acc, 0, headDim);
                        }
                    }
                }
                // 2. chunk 内 causal（本地连续 KV）
                int jFrom = Math.max(lo - startIdx, 0);
                for (int j = jFrom; j <= i; j++) {
                    int kj = j * dModel + hOff;
                    float score = Matmul.dot(q, qOff, k, kj, headDim) * invSqrt;
                    if (score > maxScore) {
                        float correction = (float) Math.exp(maxScore - score);
                        sumExp *= correction;
                        for (int d = 0; d < headDim; d++) {
                            acc[d] *= correction;
                        }
                        maxScore = score;
                    }
                    float w = (float) Math.exp(score - maxScore);
                    sumExp += w;
                    Matmul.axpy(w, v, kj, acc, 0, headDim);
                }
                // 归一化写出
                float invSum = 1f / sumExp;
                int oOff = i * dModel + hOff;
                for (int d = 0; d < headDim; d++) {
                    out[oOff + d] = acc[d] * invSum;
                }
            }
        }
        return out;
    }

    /**
     * 纯前向（无 KV cache）—— 供 PyTorch 风格 TransformerModel.forward 使用。
     * 一次性对整段序列做（可选稀疏的）causal self-attention，不读写任何 KV block。
     *
     * @param input  [seqLen, dModel]
     * @param seqLen 序列长度
     * @return [seqLen, dModel]
     */
    public float[] forwardDense(float[] input, int seqLen) {
        float[] q = qProj.forwardBatch(input, seqLen);
        float[] k = kProj.forwardBatch(input, seqLen);
        float[] v = vProj.forwardBatch(input, seqLen);
        float[] out = causalAttention(q, k, v, seqLen);
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
        // 局部带状稀疏：query(curIdx) 仅关注最近 window 个 token，其余整体跳过
        int lo = sparse ? Math.max(0, totalTokens - window) : 0;
        int nValid = totalTokens - lo;
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        float[] out = new float[dModel];
        final boolean int8 = kvMgr.isInt8();

        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            // 第一遍：逐 block 累加窗口内历史 token 的 attention scores
            float[] scores = new float[nValid];
            for (int b = 0; b < nBlocks; b++) {
                java.nio.ByteBuffer kBuf = int8 ? null : kvMgr.blockK(bt, b); // f32：堆外 [blockSize, dModel]
                byte[] k8 = int8 ? kvMgr.blockK8(bt, b) : null;   // int8：量化值
                float[] kScale = int8 ? kvMgr.blockKScale(bt, b) : null;
                int remain = totalTokens - b * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int gid = b * blockSize + s; // 该 token 的全局下标
                    if (gid < lo) continue;      // 稀疏窗口外，跳过
                    int kOff = s * dModel + hOff;
                    scores[gid - lo] = int8
                            ? Matmul.dotInt8(k8, kOff, q, hOff, headDim, kScale[s]) * invSqrt
                            : Matmul.dot(q, hOff, kBuf, kOff * 4, headDim) * invSqrt;
                }
            }
            Softmax.softmaxInPlace(scores);
            // 第二遍：逐 block 加权 V 累加输出（int8 模式在量化域直接累加）
            for (int b = 0; b < nBlocks; b++) {
                java.nio.ByteBuffer vBuf = int8 ? null : kvMgr.blockV(bt, b);
                byte[] v8 = int8 ? kvMgr.blockV8(bt, b) : null;
                float[] vScale = int8 ? kvMgr.blockVScale(bt, b) : null;
                int remain = totalTokens - b * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int gid = b * blockSize + s;
                    if (gid < lo) continue;
                    int vOff = s * dModel + hOff;
                    float w = scores[gid - lo];
                    if (int8) {
                        Matmul.axpyInt8(w, v8, vOff, out, hOff, headDim, vScale[s]);
                    } else {
                        Matmul.axpy(w, vBuf, vOff * 4, out, hOff, headDim);
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
        // gather 散落 block 的 K/V 成连续数组（INT8 模式逐行反量化，学习对照路径）
        float[] kAll = new float[totalTokens * dModel];
        float[] vAll = new float[totalTokens * dModel];
        int idx = 0;
        int nBlocks = bt.numBlocks();
        final boolean int8 = kvMgr.isInt8();
        for (int b = 0; b < nBlocks; b++) {
            int remain = totalTokens - b * blockSize;
            int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
            if (int8) {
                byte[] k8 = kvMgr.blockK8(bt, b);
                byte[] v8 = kvMgr.blockV8(bt, b);
                float[] kScale = kvMgr.blockKScale(bt, b);
                float[] vScale = kvMgr.blockVScale(bt, b);
                for (int s = 0; s < tokensInBlock; s++) {
                    for (int d = 0; d < dModel; d++) {
                        kAll[idx * dModel + d] = k8[s * dModel + d] * kScale[s];
                        vAll[idx * dModel + d] = v8[s * dModel + d] * vScale[s];
                    }
                    idx++;
                }
            } else {
                java.nio.ByteBuffer kBuf = kvMgr.blockK(bt, b);
                java.nio.ByteBuffer vBuf = kvMgr.blockV(bt, b);
                for (int s = 0; s < tokensInBlock; s++) {
                    int byteOff = s * dModel * 4;
                    for (int d = 0; d < dModel; d++) {
                        kAll[idx * dModel + d] = kBuf.getFloat(byteOff + d * 4);
                        vAll[idx * dModel + d] = vBuf.getFloat(byteOff + d * 4);
                    }
                    idx++;
                }
            }
        }
        // 标准 attention（decode 时当前 token 可见全部历史；sparse 层限制在最近 window 个）
        float[] out = new float[dModel];
        int lo = sparse ? Math.max(0, totalTokens - window) : 0;
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        for (int h = 0; h < nHead; h++) {
            int hOff = h * headDim;
            float[] scores = new float[totalTokens - lo];
            for (int j = lo; j < totalTokens; j++) {
                int kOff = j * dModel + hOff;
                scores[j - lo] = Matmul.dot(q, hOff, kAll, kOff, headDim) * invSqrt;
            }
            Softmax.softmaxInPlace(scores);
            for (int j = lo; j < totalTokens; j++) {
                int vOff = j * dModel + hOff;
                float w = scores[j - lo];
                for (int d = 0; d < headDim; d++) {
                    out[hOff + d] += w * vAll[vOff + d];
                }
            }
        }
        return oProj.forward(out);
    }
}
