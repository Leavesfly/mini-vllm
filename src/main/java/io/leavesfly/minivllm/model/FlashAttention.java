package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * FlashAttention —— 分块 causal prefill 内核（对照 FlashAttention-2 的 tiling 思想）。
 *
 * 学习要点：
 * 1. 朴素 prefill 对每个 query 物化 scores[i+1] 并两遍扫描 K/V（先 softmax 再加权），
 *    K/V 行被每个 query 重读一次——长 prompt 时 L2 流量为 O(n²·headDim)。
 * 2. 本内核按 Flash2 的"外层 Q 块、内层 KV"两级分块：Q 块（BQ=64 行）的累加器
 *    acc[BQ][headDim] 常驻 L1，KV 顺序单遍扫过（online softmax 融合，零 scores 物化），
 *    K/V 重读次数降为 n/BQ——L2 流量降一个数量级，且不改变数学结果（仅浮点求和顺序）。
 * 3. Online Softmax：每行维护 running max m 与归一化分母 l；遇到更大 score 时用
 *    exp(oldMax-newMax) 修正已有累加。与 decode 路径的 pagedAttention 同一套数学。
 * 4. 前缀共享/分块 prefill：chunk 之前已缓存的 KV（paged，可能堆外/INT8 量化）作为
 *    无因果约束的前缀段先扫，再扫 chunk 内带因果约束的连续 K/V，两段共享同一组
 *    online 状态——这就是"一条 kernel 统一纯 prefill 与带前缀 prefill"。
 * 5. GQA：Q 头 h 读 KV 头 h/group；并行粒度为 Q 头（各头写 out 不相交区间）。
 */
final class FlashAttention {

    /** Q 块行数：acc[BQ][headDim] 约 32KB（headDim=128 时），刚好常驻 L1 */
    private static final int BQ = 64;

    private FlashAttention() {
    }

    /**
     * 分块 causal prefill 注意力。
     *
     * @param q        [seqLen, nHead*headDim] 本 chunk 的 Q（已 QK-Norm+RoPE）
     * @param k        [seqLen, nKVHead*headDim] 本 chunk 的 K（同上；f32 连续数组）
     * @param v        [seqLen, nKVHead*headDim] 本 chunk 的 V
     * @param seqLen   chunk 长度
     * @param startIdx chunk 在序列中的全局起始下标（>0 时从 kvMgr 读已缓存前缀）
     * @param nHead    Q 头数
     * @param nKVHead  KV 头数（GQA；与 nHead 相等即 MHA）
     * @param headDim  头维
     * @param kvMgr    KV cache（startIdx>0 时非 null；f32 堆外 / INT8 量化均可）
     * @param bt       本层 BlockTable（读前缀用）
     * @param out      [seqLen, nHead*headDim] 输出（本方法全权写入，无需预清零）
     */
    static void causalPrefill(float[] q, float[] k, float[] v, int seqLen, int startIdx,
                              int nHead, int nKVHead, int headDim,
                              KVCacheManager kvMgr, BlockTable bt, float[] out) {
        final int qDim = nHead * headDim;
        final int kvDim = nKVHead * headDim;
        final int group = nHead / nKVHead;
        final float invSqrt = 1f / (float) Math.sqrt(headDim);
        final int blockSize = startIdx > 0 ? kvMgr.blockSize() : 0;
        final boolean int8 = startIdx > 0 && kvMgr.isInt8();

        Matmul.parallelRows(nHead, 2, h -> {
            int qOff = h * headDim;
            int kvOff = (h / group) * headDim;
            // 外层 Q 块：acc 常驻 L1，内层单遍扫 KV
            for (int i0 = 0; i0 < seqLen; i0 += BQ) {
                int rows = Math.min(BQ, seqLen - i0);
                float[] acc = new float[rows * headDim];
                float[] m = new float[rows];
                float[] l = new float[rows];
                Arrays.fill(m, Float.NEGATIVE_INFINITY);

                // 1. paged 前缀段（无因果约束：块内所有 query 可见全部前缀 token）
                for (int g = 0; g < startIdx; g++) {
                    int blk = g / blockSize;
                    int s = g % blockSize;
                    if (int8) {
                        byte[] k8 = kvMgr.blockK8(bt, blk);
                        byte[] v8 = kvMgr.blockV8(bt, blk);
                        float sc = kvMgr.blockKScale(bt, blk)[s];
                        float sv = kvMgr.blockVScale(bt, blk)[s];
                        int off8 = s * kvDim + kvOff;
                        for (int ii = 0; ii < rows; ii++) {
                            float score = Matmul.dotInt8(k8, off8, q, (i0 + ii) * qDim + qOff, headDim, sc)
                                    * invSqrt;
                            accMulExp(score, ii, m, l, acc, headDim);
                            float w = (float) Math.exp(score - m[ii]);
                            l[ii] += w;
                            Matmul.axpyInt8(w, v8, off8, acc, ii * headDim, headDim, sv);
                        }
                    } else {
                        ByteBuffer kBuf = kvMgr.blockK(bt, blk);
                        ByteBuffer vBuf = kvMgr.blockV(bt, blk);
                        int byteOff = (s * kvDim + kvOff) * 4;
                        for (int ii = 0; ii < rows; ii++) {
                            float score = Matmul.dot(q, (i0 + ii) * qDim + qOff, kBuf, byteOff, headDim)
                                    * invSqrt;
                            accMulExp(score, ii, m, l, acc, headDim);
                            float w = (float) Math.exp(score - m[ii]);
                            l[ii] += w;
                            Matmul.axpy(w, vBuf, byteOff, acc, ii * headDim, headDim);
                        }
                    }
                }

                // 2. chunk 内 causal 段：query（全局 startIdx+i0+ii）仅见 j ≤ i0+ii 的本地 KV
                for (int j = 0; j < i0 + rows; j++) {
                    int kj = j * kvDim + kvOff;
                    int iiFrom = Math.max(j - i0, 0); // 因果：本块内行 i0+ii ≥ j 才参与
                    for (int ii = iiFrom; ii < rows; ii++) {
                        float score = Matmul.dot(q, (i0 + ii) * qDim + qOff, k, kj, headDim) * invSqrt;
                        accMulExp(score, ii, m, l, acc, headDim);
                        float w = (float) Math.exp(score - m[ii]);
                        l[ii] += w;
                        Matmul.axpy(w, v, kj, acc, ii * headDim, headDim);
                    }
                }

                // 3. 归一化写出本块
                for (int ii = 0; ii < rows; ii++) {
                    float invSum = 1f / l[ii];
                    int oOff = (i0 + ii) * qDim + qOff;
                    for (int d = 0; d < headDim; d++) {
                        out[oOff + d] = acc[ii * headDim + d] * invSum;
                    }
                }
            }
        });
    }

    /**
     * Online softmax 的 max 更新：score 刷新行 ii 的 running max 时，
     * 用 exp(oldMax-newMax) 修正该行的分母 l 与未归一化累加器 acc。
     */
    private static void accMulExp(float score, int ii, float[] m, float[] l, float[] acc, int headDim) {
        if (score > m[ii]) {
            float correction = (float) Math.exp(m[ii] - score);
            l[ii] *= correction;
            int base = ii * headDim;
            for (int d = 0; d < headDim; d++) {
                acc[base + d] *= correction;
            }
            m[ii] = score;
        }
    }
}
