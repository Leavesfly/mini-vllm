package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * FlashAttentionTest —— 分块 prefill 内核的正确性测试。
 *
 * 参照系是测试内手写的朴素两遍 causal attention（物化全部 scores + 标准 softmax）。
 * 核心断言：flash 与朴素参考在 fp 容差内一致，覆盖：
 * 1. 纯 causal prefill（startIdx=0），seqLen 跨越 Q 块边界（>64）验证分块与因果边界
 * 2. 带 paged 前缀的 chunked prefill：两段 chunk 的结果与一次性 prefill 一致
 * 3. GQA 头映射（nHead ≠ nKVHead）
 * 4. 因果性：未来位置的极端大 K/V 不得影响当前 query 输出
 * 5. INT8 量化前缀路径（容差放宽到量化误差量级）
 */
class FlashAttentionTest {

    private static final int N_HEAD = 4;
    private static final int N_KV_HEAD = 2; // GQA：group=2
    private static final int HEAD_DIM = 32;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_KV_HEAD * HEAD_DIM;
    private static final int BLOCK_SIZE = 16;
    private static final float TOL = 2e-3f; // online softmax 仅改变求和顺序，容差覆盖 fp 重排

    private static float[] randRows(Random rnd, int rows, int cols) {
        float[] a = new float[rows * cols];
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) (rnd.nextGaussian() * 0.5); // 温和量级，避免 score 极端
        }
        return a;
    }

    /** 朴素参考：逐 query 物化 scores + 标准 softmax（kAll/vAll 为全序列 KV） */
    private static float[] naive(float[] q, float[] kAll, float[] vAll, int seqLen, int startIdx) {
        float[] out = new float[seqLen * Q_DIM];
        float invSqrt = 1f / (float) Math.sqrt(HEAD_DIM);
        for (int h = 0; h < N_HEAD; h++) {
            int qOff = h * HEAD_DIM;
            int kvOff = (h / (N_HEAD / N_KV_HEAD)) * HEAD_DIM;
            for (int i = 0; i < seqLen; i++) {
                int gPos = startIdx + i; // 全局位置：可见 j ∈ [0, gPos]
                float[] scores = new float[gPos + 1];
                for (int j = 0; j <= gPos; j++) {
                    float s = 0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        s += q[i * Q_DIM + qOff + d] * kAll[j * KV_DIM + kvOff + d];
                    }
                    scores[j] = s * invSqrt;
                }
                float max = Float.NEGATIVE_INFINITY;
                for (float s : scores) {
                    max = Math.max(max, s);
                }
                float sum = 0f;
                for (int j = 0; j < scores.length; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int d = 0; d < HEAD_DIM; d++) {
                    float acc = 0f;
                    for (int j = 0; j <= gPos; j++) {
                        acc += scores[j] * vAll[j * KV_DIM + kvOff + d];
                    }
                    out[i * Q_DIM + qOff + d] = acc / sum;
                }
            }
        }
        return out;
    }

    private static void assertClose(float[] expected, float[] actual, float tol, String what) {
        double maxDiff = 0;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(expected[i] - actual[i]));
        }
        assertTrue(maxDiff <= tol, what + " 最大偏差 " + maxDiff + " 超过容差 " + tol);
    }

    @Test
    void pureCausalMatchesNaiveAcrossTileBoundary() {
        Random rnd = new Random(7);
        int seqLen = 130; // > BQ(64)，跨两个 Q 块，覆盖块间因果边界
        float[] q = randRows(rnd, seqLen, Q_DIM);
        float[] k = randRows(rnd, seqLen, KV_DIM);
        float[] v = randRows(rnd, seqLen, KV_DIM);

        float[] out = new float[seqLen * Q_DIM];
        FlashAttention.causalPrefill(q, k, v, seqLen, 0, N_HEAD, N_KV_HEAD, HEAD_DIM,
                null, null, out);
        assertClose(naive(q, k, v, seqLen, 0), out, TOL, "纯 causal prefill");
    }

    @Test
    void chunkedPrefillWithPagedPrefixMatchesOneShot() {
        Random rnd = new Random(11);
        int total = 100;
        int chunk1 = 37; // 故意不对齐 block 边界（37 % 16 ≠ 0）
        int seqLen = total - chunk1;
        float[] qAll = randRows(rnd, total, Q_DIM);
        float[] kAll = randRows(rnd, total, KV_DIM);
        float[] vAll = randRows(rnd, total, KV_DIM);

        // 一次性：全序列纯 causal（参考用朴素算法，另跑 flash 对照）
        float[] q2 = new float[seqLen * Q_DIM];
        System.arraycopy(qAll, chunk1 * Q_DIM, q2, 0, q2.length);

        // 两段式：chunk1 的 K/V 写入 paged cache，chunk2 带前缀走 flash
        KVCacheManager kvMgr = new KVCacheManager(64, BLOCK_SIZE, KV_DIM);
        BlockTable bt = new BlockTable();
        kvMgr.ensureCapacity(bt, total);
        float[] kr = new float[KV_DIM];
        float[] vr = new float[KV_DIM];
        for (int t = 0; t < chunk1; t++) {
            System.arraycopy(kAll, t * KV_DIM, kr, 0, KV_DIM);
            System.arraycopy(vAll, t * KV_DIM, vr, 0, KV_DIM);
            kvMgr.writeKV(bt, t, kr, vr);
        }
        float[] k2 = new float[seqLen * KV_DIM];
        float[] v2 = new float[seqLen * KV_DIM];
        System.arraycopy(kAll, chunk1 * KV_DIM, k2, 0, k2.length);
        System.arraycopy(vAll, chunk1 * KV_DIM, v2, 0, v2.length);

        float[] out = new float[seqLen * Q_DIM];
        FlashAttention.causalPrefill(q2, k2, v2, seqLen, chunk1, N_HEAD, N_KV_HEAD, HEAD_DIM,
                kvMgr, bt, out);
        assertClose(naive(q2, kAll, vAll, seqLen, chunk1), out, TOL, "带前缀 chunked prefill");
    }

    @Test
    void futureTokensDoNotLeak() {
        Random rnd = new Random(13);
        int seqLen = 70;
        float[] q = randRows(rnd, seqLen, Q_DIM);
        float[] k = randRows(rnd, seqLen, KV_DIM);
        float[] v = randRows(rnd, seqLen, KV_DIM);
        // 在"未来"位置埋入极端值：若因果泄漏，前面 query 的输出会剧烈变化
        for (int d = 0; d < KV_DIM; d++) {
            k[(seqLen - 1) * KV_DIM + d] = 1e4f;
            v[(seqLen - 2) * KV_DIM + d] = 1e4f;
        }
        float[] out = new float[seqLen * Q_DIM];
        FlashAttention.causalPrefill(q, k, v, seqLen, 0, N_HEAD, N_KV_HEAD, HEAD_DIM,
                null, null, out);
        assertClose(naive(q, k, v, seqLen, 0), out, TOL, "因果屏蔽");
    }

    @Test
    void int8PrefixPathConsistentWithF32() {
        Random rnd = new Random(17);
        int total = 50;
        int chunk1 = 32;
        int seqLen = total - chunk1;
        float[] qAll = randRows(rnd, total, Q_DIM);
        float[] kAll = randRows(rnd, total, KV_DIM);
        float[] vAll = randRows(rnd, total, KV_DIM);

        float[] q2 = new float[seqLen * Q_DIM];
        float[] k2 = new float[seqLen * KV_DIM];
        float[] v2 = new float[seqLen * KV_DIM];
        System.arraycopy(qAll, chunk1 * Q_DIM, q2, 0, q2.length);
        System.arraycopy(kAll, chunk1 * KV_DIM, k2, 0, k2.length);
        System.arraycopy(vAll, chunk1 * KV_DIM, v2, 0, v2.length);

        float[] kr = new float[KV_DIM];
        float[] vr = new float[KV_DIM];
        // f32 前缀
        KVCacheManager f32Mgr = new KVCacheManager(64, BLOCK_SIZE, KV_DIM);
        BlockTable btF = new BlockTable();
        f32Mgr.ensureCapacity(btF, total);
        // INT8 前缀
        KVCacheManager i8Mgr = new KVCacheManager(64, BLOCK_SIZE, KV_DIM, true);
        BlockTable btI = new BlockTable();
        i8Mgr.ensureCapacity(btI, total);
        for (int t = 0; t < chunk1; t++) {
            System.arraycopy(kAll, t * KV_DIM, kr, 0, KV_DIM);
            System.arraycopy(vAll, t * KV_DIM, vr, 0, KV_DIM);
            f32Mgr.writeKV(btF, t, kr, vr);
            i8Mgr.writeKV(btI, t, kr, vr);
        }

        float[] outF32 = new float[seqLen * Q_DIM];
        FlashAttention.causalPrefill(q2, k2, v2, seqLen, chunk1, N_HEAD, N_KV_HEAD, HEAD_DIM,
                f32Mgr, btF, outF32);
        float[] outI8 = new float[seqLen * Q_DIM];
        FlashAttention.causalPrefill(q2, k2, v2, seqLen, chunk1, N_HEAD, N_KV_HEAD, HEAD_DIM,
                i8Mgr, btI, outI8);
        // INT8 量化误差（per-row scale=absmax/127）经 softmax 加权后仍应远小于 5e-2
        assertClose(outF32, outI8, 5e-2f, "INT8 前缀 vs f32 前缀");
    }
}
