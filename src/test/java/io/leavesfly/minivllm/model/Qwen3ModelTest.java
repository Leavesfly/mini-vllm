package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.math.Tensor;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Qwen3Model 随机权重自洽性测试。
 *
 * 核心验证：同一随机模型，PyTorch 风格 forward、引擎 prefill、引擎 decode
 * 三条路径的 logits 必须一致（GQA 头映射 / RoPE 位置 / QK-Norm 若有一处错，
 * 三条路径就会分叉）。
 */
class Qwen3ModelTest {

    /** 小尺寸 Qwen3 配置：qDim(64) ≠ dModel(32)，GQA 4头/2KV头 */
    private static ModelConfig tinyQwen3() {
        return new ModelConfig()
                .arch("qwen3")
                .name("qwen3-tiny")
                .vocabSize(50)
                .dModel(32)
                .nHead(4)
                .nKVHead(2)
                .headDimExplicit(16) // qDim = 4*16 = 64 ≠ dModel=32
                .nLayer(2)
                .dFfn(64)
                .blockSize(4)
                .maxSeqLen(64)
                .ropeTheta(10000f)
                .rmsNormEps(1e-6f);
    }

    @Test
    void gqaDimensionSemantics() {
        ModelConfig c = tinyQwen3();
        assertEquals(16, c.headDim());
        assertEquals(2, c.kvHeads());
        assertEquals(64, c.qDim());  // nHead * headDim
        assertEquals(32, c.kvDim()); // nKVHead * headDim
    }

    @Test
    void prefillDecodeForwardConsistency() {
        ModelConfig cfg = tinyQwen3();
        Qwen3Model model = randomQwen3(cfg, 42L);
        int[] ids = {3, 1, 4, 1, 5, 9, 2, 6};
        int seqLen = ids.length;

        // 路径 A：PyTorch 风格整段 forward
        Tensor all = model.forward(ids);
        assertEquals(seqLen, all.shape()[0]);
        assertEquals(cfg.vocabSize(), all.shape()[1]);

        // 路径 B：引擎 prefill（一次性整段）
        KVCacheManager kvMgr = newKvMgr(cfg);
        BlockTable[] bts = newBlockTables(cfg);
        ensureCapacity(kvMgr, bts, seqLen);
        float[] prefillLast = model.prefillLogits(ids, kvMgr, bts, 0);
        float[] forwardLast = row(all, seqLen - 1, cfg.vocabSize());
        assertArrayEquals(forwardLast, prefillLast, 1e-4f, "prefill 与 forward 最后位置不一致");

        // 路径 C：prefill 前 3 个 + 逐 token decode，逐位置与 forward 对比
        KVCacheManager kvMgr2 = newKvMgr(cfg);
        BlockTable[] bts2 = newBlockTables(cfg);
        ensureCapacity(kvMgr2, bts2, seqLen);
        int[] head = {ids[0], ids[1], ids[2]};
        model.prefillLogits(head, kvMgr2, bts2, 0);
        for (int t = 3; t < seqLen; t++) {
            float[] dec = model.decodeLogits(ids[t], t, kvMgr2, bts2);
            assertArrayEquals(row(all, t, cfg.vocabSize()), dec, 1e-4f,
                    "decode 位置 " + t + " 与 forward 不一致");
        }
    }

    @Test
    void gqaSharesKvHeadsWithinGroup() {
        // nHead=2, nKVHead=1, headDim=4, dModel=8：两个 Q 头共享同一个 KV 头。
        // 把 qProj 的两个头行设为相同 -> 两个头的 attention 输出必须完全相同。
        ModelConfig c = new ModelConfig()
                .arch("qwen3")
                .vocabSize(10)
                .dModel(8)
                .nHead(2)
                .nKVHead(1)
                .headDimExplicit(4)
                .nLayer(1)
                .dFfn(16)
                .blockSize(4)
                .maxSeqLen(16);

        Random rnd = new Random(7L);
        int dModel = 8, qDim = 8, kvDim = 4;
        float[] qw = new float[qDim * dModel];
        for (int i = 0; i < qDim * dModel; i++) {
            qw[i] = (float) (rnd.nextGaussian() * 0.3);
        }
        // 头 1 的行 = 头 0 的行（行优先 [out, in]，头 h 占行 [h*headDim, (h+1)*headDim)）
        System.arraycopy(qw, 0, qw, 4 * dModel, 4 * dModel);
        Linear qProj = Linear.of(qw, dModel, qDim);
        Linear kProj = Linear.of(randN(rnd, kvDim * dModel, 0.3f), dModel, kvDim);
        Linear vProj = Linear.of(randN(rnd, kvDim * dModel, 0.3f), dModel, kvDim);
        // oProj 恒等：输出 = 两个头输出直接拼接
        float[] id = new float[qDim * dModel];
        for (int i = 0; i < qDim; i++) {
            id[i * dModel + i] = 1f;
        }
        Linear oProj = Linear.of(id, qDim, dModel);
        RmsNorm ones = new RmsNorm(new float[]{1, 1, 1, 1}, 1e-6f);
        RotaryEmbedding rope = new RotaryEmbedding(4, 16, 10000f);
        Qwen3Attention attn = new Qwen3Attention(c, qProj, kProj, vProj, oProj, ones, ones, rope);

        int seqLen = 5;
        float[] input = randN(rnd, seqLen * dModel, 0.5f);
        float[] out = attn.forwardDense(input, seqLen);
        for (int t = 0; t < seqLen; t++) {
            for (int d = 0; d < 4; d++) {
                assertEquals(out[t * dModel + d], out[t * dModel + 4 + d], 1e-5f,
                        "GQA 组内头输出不一致 t=" + t + " d=" + d);
            }
        }
    }

    @Test
    void orderMatters() {
        // 因果注意力 + RoPE 下，[a,b] 与 [b,a] 的最后位置 logits 必须不同。
        // 注：相同 token 重复序列无法探测 RoPE（各位置 V 相同，注意力权重不影响输出），
        //     故用两个不同 token 的乱序对验证位置敏感性。
        ModelConfig cfg = tinyQwen3();
        Qwen3Model model = randomQwen3(cfg, 1L);
        float[] ab = model.forwardLastLogits(new int[]{7, 42});
        float[] ba = model.forwardLastLogits(new int[]{42, 7});
        float maxDiff = 0f;
        for (int i = 0; i < ab.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(ab[i] - ba[i]));
        }
        assertTrue(maxDiff > 1e-3f, "[a,b] 与 [b,a] 的最后位置 logits 应不同");
    }

    // ===================== 测试辅助 =====================

    private static KVCacheManager newKvMgr(ModelConfig cfg) {
        return new KVCacheManager(64, cfg.blockSize(), cfg.kvDim());
    }

    private static void ensureCapacity(KVCacheManager kvMgr, BlockTable[] bts, int tokens) {
        for (BlockTable bt : bts) {
            assertTrue(kvMgr.ensureCapacity(bt, tokens), "KV 池容量不足");
        }
    }

    private static BlockTable[] newBlockTables(ModelConfig cfg) {
        BlockTable[] bts = new BlockTable[cfg.nLayer()];
        for (int i = 0; i < bts.length; i++) {
            bts[i] = new BlockTable();
        }
        return bts;
    }

    private static float[] row(Tensor t, int r, int width) {
        float[] out = new float[width];
        System.arraycopy(t.data(), r * width, out, 0, width);
        return out;
    }

    /** 随机初始化一个小尺寸 Qwen3Model（std=0.02，与 ModelLoader 风格一致） */
    static Qwen3Model randomQwen3(ModelConfig cfg, long seed) {
        Random rnd = new Random(seed);
        Embedding wte = new Embedding(randN(rnd, cfg.vocabSize() * cfg.dModel(), 0.02f),
                cfg.vocabSize(), cfg.dModel());
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen(), cfg.ropeTheta());
        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer()];
        for (int i = 0; i < cfg.nLayer(); i++) {
            RmsNorm ln1 = rmsOnes(cfg.dModel(), cfg.rmsNormEps());
            RmsNorm ln2 = rmsOnes(cfg.dModel(), cfg.rmsNormEps());
            Linear q = Linear.of(randN(rnd, cfg.qDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.qDim());
            Linear k = Linear.of(randN(rnd, cfg.kvDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.kvDim());
            Linear v = Linear.of(randN(rnd, cfg.kvDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.kvDim());
            Linear o = Linear.of(randN(rnd, cfg.dModel() * cfg.qDim(), 0.02f), cfg.qDim(), cfg.dModel());
            RmsNorm qNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps());
            RmsNorm kNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps());
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = Linear.of(randN(rnd, cfg.dFfn() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.dFfn());
            Linear up = Linear.of(randN(rnd, cfg.dFfn() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.dFfn());
            Linear down = Linear.of(randN(rnd, cfg.dModel() * cfg.dFfn(), 0.02f), cfg.dFfn(), cfg.dModel());
            blocks[i] = new Qwen3Block(ln1, ln2, attn, new SwiGluFfn(gate, up, down), cfg.dModel());
        }
        return new Qwen3Model(cfg, wte, blocks, rmsOnes(cfg.dModel(), cfg.rmsNormEps()));
    }

    private static RmsNorm rmsOnes(int dim, float eps) {
        float[] w = new float[dim];
        java.util.Arrays.fill(w, 1f);
        return new RmsNorm(w, eps);
    }

    private static float[] randN(Random rnd, int n, float std) {
        float[] r = new float[n];
        for (int i = 0; i < n; i += 2) {
            double u1 = Math.max(rnd.nextDouble(), 1e-12);
            double u2 = rnd.nextDouble();
            double radius = Math.sqrt(-2.0 * Math.log(u1));
            r[i] = (float) (radius * Math.cos(2 * Math.PI * u2) * std);
            if (i + 1 < n) {
                r[i + 1] = (float) (radius * Math.sin(2 * Math.PI * u2) * std);
            }
        }
        return r;
    }
}
