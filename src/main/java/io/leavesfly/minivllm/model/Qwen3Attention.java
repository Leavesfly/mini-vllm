package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.math.Softmax;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

/**
 * Qwen3Attention —— Qwen3 的自注意力层：GQA + QK-Norm + RoPE。
 *
 * 学习要点（与 GPT-2 风格 {@link Attention} 的差异）：
 * 1. GQA（Grouped Query Attention）：nHead 个 Q 头共享 nKVHead 个 K/V 头，
 *    Q 头 h 读取 KV 头 h/group（group = nHead/nKVHead）。KV cache 每 token 只存
 *    kvDim = nKVHead*headDim（而非 dModel），显存与带宽都省 group 倍。
 * 2. QK-Norm：Q/K 投影后先按头做 RMSNorm（Qwen3 特有，稳定注意力 logits），
 *    再施加 RoPE。顺序不能反：先 Norm 后 RoPE。
 * 3. RoPE：位置编码作用在 Q/K 上（half-split 旋转，见 {@link RotaryEmbedding}），
 *    替代 GPT-2 的学习式位置嵌入 wpe；V 不做旋转。
 * 4. 维度关系（Qwen3-0.6B）：dModel=1024，headDim=128，nHead=16 -> qDim=2048，
 *    nKVHead=8 -> kvDim=1024。注意 qDim ≠ dModel（headDim 是独立超参）。
 * 5. PagedAttention 语义与 GPT-2 路径一致：prefill 写 KV、decode 逐 block 累加，
 *    区别仅在于 block 内行 stride 为 kvDim，头偏移按 KV 头计算。
 */
public final class Qwen3Attention {

    private final Linear qProj;
    private final Linear kProj;
    private final Linear vProj;
    private final Linear oProj;
    private final RmsNorm qNorm; // [headDim]，层内各 Q 头共享
    private final RmsNorm kNorm; // [headDim]，层内各 KV 头共享
    private final RotaryEmbedding rope;

    private final int nHead;
    private final int nKVHead;
    private final int group;   // nHead / nKVHead
    private final int headDim;
    private final int qDim;    // nHead * headDim
    private final int kvDim;   // nKVHead * headDim
    private final int blockSize;

    /**
     * 融合 QKV 投影权重（性能优化）：把 q/k/v 三个 Linear 的权重拼接为一个
     * [qDim+2*kvDim, dModel] 的大矩阵。decode 时一次 matVec 完成全部投影，
     * 将 3 次 fork-join 线程调度合并为 1 次，减少每层 ~2 次调度开销。
     */
    private final Linear fusedQKV;
    private final int qkvDim; // qDim + 2*kvDim

    public Qwen3Attention(ModelConfig cfg, Linear qProj, Linear kProj, Linear vProj, Linear oProj,
                          RmsNorm qNorm, RmsNorm kNorm, RotaryEmbedding rope) {
        this.qProj = qProj;
        this.kProj = kProj;
        this.vProj = vProj;
        this.oProj = oProj;
        this.qNorm = qNorm;
        this.kNorm = kNorm;
        this.rope = rope;
        this.nHead = cfg.nHead;
        this.nKVHead = cfg.kvHeads();
        this.group = nHead / nKVHead;
        this.headDim = cfg.headDim();
        this.qDim = cfg.qDim();
        this.kvDim = cfg.kvDim();
        this.blockSize = cfg.blockSize;
        this.qkvDim = qDim + 2 * kvDim;
        this.fusedQKV = buildFusedQKV();
    }

    /** 拼接 q/k/v 权重为单个 Linear（支持 bf16/int8/f32 三种格式） */
    private Linear buildFusedQKV() {
        int in = qProj.inFeatures;
        if (qProj.isBf16()) {
            short[] w = new short[qkvDim * in];
            System.arraycopy(qProj.weightBf16, 0, w, 0, qDim * in);
            System.arraycopy(kProj.weightBf16, 0, w, qDim * in, kvDim * in);
            System.arraycopy(vProj.weightBf16, 0, w, (qDim + kvDim) * in, kvDim * in);
            return Linear.ofBf16(w, in, qkvDim);
        } else if (qProj.isInt8()) {
            byte[] w = new byte[qkvDim * in];
            float[] s = new float[qkvDim];
            System.arraycopy(qProj.weightInt8, 0, w, 0, qDim * in);
            System.arraycopy(qProj.scaleInt8, 0, s, 0, qDim);
            System.arraycopy(kProj.weightInt8, 0, w, qDim * in, kvDim * in);
            System.arraycopy(kProj.scaleInt8, 0, s, qDim, kvDim);
            System.arraycopy(vProj.weightInt8, 0, w, (qDim + kvDim) * in, kvDim * in);
            System.arraycopy(vProj.scaleInt8, 0, s, qDim + kvDim, kvDim);
            return Linear.ofInt8(w, s, in, qkvDim);
        } else {
            float[] w = new float[qkvDim * in];
            System.arraycopy(qProj.weight, 0, w, 0, qDim * in);
            System.arraycopy(kProj.weight, 0, w, qDim * in, kvDim * in);
            System.arraycopy(vProj.weight, 0, w, (qDim + kvDim) * in, kvDim * in);
            return Linear.of(w, in, qkvDim);
        }
    }

    /** 参数量（q/k/v/o 投影 + qk norm） */
    public long numParameters() {
        return qProj.numParameters() + kProj.numParameters()
                + vProj.numParameters() + oProj.numParameters()
                + qNorm.numParameters() + kNorm.numParameters();
    }

    /**
     * Prefill：处理一段 prompt token，写入 KV cache。
     *
     * @param input   [seqLen, dModel] 归一化后的输入
     * @param seqLen  prompt 长度
     * @param kvMgr   KV cache 管理器
     * @param bt      本层本请求的 BlockTable
     * @param startIdx 这些 token 在序列中的起始全局下标
     * @return [seqLen, dModel] attention 输出
     */
    public float[] prefill(float[] input, int seqLen, KVCacheManager kvMgr, BlockTable bt, int startIdx) {
        // 1. 投影 Q/K/V
        float[] q = qProj.forwardBatch(input, seqLen); // [seqLen, qDim]
        float[] k = kProj.forwardBatch(input, seqLen); // [seqLen, kvDim]
        float[] v = vProj.forwardBatch(input, seqLen); // [seqLen, kvDim]

        // 2. QK-Norm（按头）+ RoPE（按位置），再写 KV cache
        float[] kt = new float[kvDim];
        float[] vt = new float[kvDim];
        for (int t = 0; t < seqLen; t++) {
            int pos = startIdx + t;
            applyQkNormAndRope(q, t * qDim, k, t * kvDim, pos);
            System.arraycopy(k, t * kvDim, kt, 0, kvDim);
            System.arraycopy(v, t * kvDim, vt, 0, kvDim);
            kvMgr.writeKV(bt, pos, kt, vt);
        }

        // 3. causal GQA attention（K/V 连续在手，直接算）
        float[] out = causalAttention(q, k, v, seqLen);
        // 4. 输出投影 [seqLen, qDim] -> [seqLen, dModel]
        return oProj.forwardBatch(out, seqLen);
    }

    /**
     * 纯前向（无 KV cache）—— 供 PyTorch 风格 Qwen3Model.forward 使用。
     */
    public float[] forwardDense(float[] input, int seqLen) {
        float[] q = qProj.forwardBatch(input, seqLen);
        float[] k = kProj.forwardBatch(input, seqLen);
        float[] v = vProj.forwardBatch(input, seqLen);
        for (int t = 0; t < seqLen; t++) {
            applyQkNormAndRope(q, t * qDim, k, t * kvDim, t);
        }
        float[] out = causalAttention(q, k, v, seqLen);
        return oProj.forwardBatch(out, seqLen);
    }

    /**
     * Decode（PagedAttention block-wise 累加）：处理单个新 token。
     * 性能优化：Q/K/V 融合投影，3 次 fork-join 合并为 1 次。
     *
     * @param hidden [dModel] 当前 token 归一化后的隐状态
     * @param curIdx 当前 token 的全局下标
     * @return [dModel] attention 输出
     */
    public float[] decodePaged(float[] hidden, int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        // 融合 QKV 投影：一次 matVec 得到 [qDim+2*kvDim]，切分出 q/k/v
        float[] qkv = fusedQKV.forward(hidden);
        float[] q = new float[qDim];
        float[] k = new float[kvDim];
        float[] v = new float[kvDim];
        System.arraycopy(qkv, 0, q, 0, qDim);
        System.arraycopy(qkv, qDim, k, 0, kvDim);
        System.arraycopy(qkv, qDim + kvDim, v, 0, kvDim);

        applyQkNormAndRope(q, 0, k, 0, curIdx);
        kvMgr.writeKV(bt, curIdx, k, v);

        float[] out = new float[qDim];
        pagedAttention(q, 0, out, 0, curIdx, kvMgr, bt);
        return oProj.forward(out);
    }

    /**
     * 批量 Decode：一次处理 B 个序列各自的新 token。
     *
     * @param hidden  [B, dModel] B 个序列归一化后的隐状态
     * @param batch   批大小 B
     * @param curIdxs 每个序列当前 token 的全局下标
     * @param bts     每个序列本层的 BlockTable
     * @return [B, dModel] attention 输出
     *
     * q/k/v/o 投影走 forwardBatch（权重跨 B 行复用）；QK-Norm/RoPE/writeKV 与注意力按序列独立。
     */
    public float[] decodeBatch(float[] hidden, int batch, int[] curIdxs,
                               KVCacheManager kvMgr, BlockTable[] bts) {
        // 融合 QKV 投影：一次 forwardBatch 得到 [B, qkvDim]，切分出 q/k/v
        float[] qkvAll = fusedQKV.forwardBatch(hidden, batch); // [B, qkvDim]
        float[] q = new float[batch * qDim];
        float[] k = new float[batch * kvDim];
        float[] v = new float[batch * kvDim];
        for (int b = 0; b < batch; b++) {
            int src = b * qkvDim;
            System.arraycopy(qkvAll, src, q, b * qDim, qDim);
            System.arraycopy(qkvAll, src + qDim, k, b * kvDim, kvDim);
            System.arraycopy(qkvAll, src + qDim + kvDim, v, b * kvDim, kvDim);
        }
        float[] kt = new float[kvDim];
        float[] vt = new float[kvDim];
        for (int b = 0; b < batch; b++) {
            applyQkNormAndRope(q, b * qDim, k, b * kvDim, curIdxs[b]);
            System.arraycopy(k, b * kvDim, kt, 0, kvDim);
            System.arraycopy(v, b * kvDim, vt, 0, kvDim);
            kvMgr.writeKV(bts[b], curIdxs[b], kt, vt);
        }
        float[] out = new float[batch * qDim];
        for (int b = 0; b < batch; b++) {
            pagedAttention(q, b * qDim, out, b * qDim, curIdxs[b], kvMgr, bts[b]);
        }
        return oProj.forwardBatch(out, batch); // [B, dModel]
    }

    /**
     * PagedAttention block-wise 累加：用已 QK-Norm+RoPE 的 q（位于 qBase）与 bt 中历史 KV
     * 计算 GQA 注意力，写入 out 的 outBase 处。单次/批量 decode 共用此核心。
     *
     * 性能：多头并行（各头写 out 的不同区间，无数据竞争），
     * 长上下文时 attention 计算量大，阈值降为 2 即并行。
     */
    private void pagedAttention(float[] q, int qBase, float[] out, int outBase,
                               int curIdx, KVCacheManager kvMgr, BlockTable bt) {
        int totalTokens = curIdx + 1;
        int nBlocks = bt.numBlocks();
        float invSqrt = 1f / (float) Math.sqrt(headDim);

        // 多头并行：各头写入 out 的不同 qOff 区间，无数据竞争
        Matmul.parallelRows(nHead, 2, h -> {
            int qOff = qBase + h * headDim;
            int kvOff = (h / group) * headDim; // GQA：Q 头 h 读 KV 头 h/group
            // 第一遍：逐 block 累加 attention scores
            float[] scores = new float[totalTokens];
            for (int blk = 0; blk < nBlocks; blk++) {
                float[] kBlk = kvMgr.blockK(bt, blk); // [blockSize, kvDim]
                int remain = totalTokens - blk * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int kOff = s * kvDim + kvOff;
                    scores[blk * blockSize + s] = Matmul.dot(q, qOff, kBlk, kOff, headDim) * invSqrt;
                }
            }
            Softmax.softmaxInPlace(scores);
            // 第二遍：逐 block 加权 V 累加输出（SIMD axpy）
            int oOff = outBase + h * headDim;
            for (int blk = 0; blk < nBlocks; blk++) {
                float[] vBlk = kvMgr.blockV(bt, blk);
                int remain = totalTokens - blk * blockSize;
                int tokensInBlock = Math.min(blockSize, Math.max(0, remain));
                for (int s = 0; s < tokensInBlock; s++) {
                    int vOff = s * kvDim + kvOff;
                    float w = scores[blk * blockSize + s];
                    Matmul.axpy(w, vBlk, vOff, out, oOff, headDim);
                }
            }
        });
    }

    // ===================== 内部工具 =====================

    /** QK-Norm（按头）后接 RoPE（按位置 pos），q/k 就地修改 */
    private void applyQkNormAndRope(float[] q, int qBase, float[] k, int kBase, int pos) {
        for (int h = 0; h < nHead; h++) {
            qNorm.forwardInPlace(q, qBase + h * headDim);
        }
        for (int kh = 0; kh < nKVHead; kh++) {
            kNorm.forwardInPlace(k, kBase + kh * headDim);
        }
        for (int h = 0; h < nHead; h++) {
            rope.applyInPlace(q, qBase + h * headDim, pos);
        }
        for (int kh = 0; kh < nKVHead; kh++) {
            rope.applyInPlace(k, kBase + kh * headDim, pos);
        }
    }

    /**
     * causal GQA 多头注意力核心。
     * q: [seqLen, qDim]，k/v: [seqLen, kvDim] -> 返回 [seqLen, qDim]。
     */
    private float[] causalAttention(float[] q, float[] k, float[] v, int seqLen) {
        float[] out = new float[seqLen * qDim];
        float invSqrt = 1f / (float) Math.sqrt(headDim);
        // 多头并行：各头写入 out 的不同 qOff 区间，无数据竞争；prefill 计算量大，阈值降到 2 即并行
        Matmul.parallelRows(nHead, 2, h -> {
            int qOff = h * headDim;
            int kvOff = (h / group) * headDim;
            for (int i = 0; i < seqLen; i++) {
                int qi = i * qDim + qOff;
                float[] scores = new float[i + 1];
                for (int j = 0; j <= i; j++) {
                    int kj = j * kvDim + kvOff;
                    scores[j] = Matmul.dot(q, qi, k, kj, headDim) * invSqrt;
                }
                Softmax.softmaxInPlace(scores);
                int oi = i * qDim + qOff;
                for (int j = 0; j <= i; j++) {
                    int vj = j * kvDim + kvOff;
                    float w = scores[j];
                    Matmul.axpy(w, v, vj, out, oi, headDim);
                }
            }
        });
        return out;
    }
}
