package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.math.RmsNorm;
import io.leavesfly.minivllm.model.Embedding;
import io.leavesfly.minivllm.model.Linear;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.Qwen3Attention;
import io.leavesfly.minivllm.model.Qwen3Block;
import io.leavesfly.minivllm.model.Qwen3Model;
import io.leavesfly.minivllm.model.RotaryEmbedding;
import io.leavesfly.minivllm.model.SwiGluFfn;

import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Qwen3Loader —— 把 HuggingFace Qwen3 权重字典组装成 Qwen3Model。
 *
 * 权重命名约定（HF Qwen3ForCausalLM，共 338 个张量，tied 无 lm_head）：
 *   model.embed_tokens.weight                          [vocab, dModel]
 *   model.layers.{i}.input_layernorm.weight            [dModel]
 *   model.layers.{i}.self_attn.q_proj.weight           [qDim, dModel]   qDim = nHead*headDim
 *   model.layers.{i}.self_attn.k_proj.weight           [kvDim, dModel]  kvDim = nKVHead*headDim
 *   model.layers.{i}.self_attn.v_proj.weight           [kvDim, dModel]
 *   model.layers.{i}.self_attn.o_proj.weight           [dModel, qDim]
 *   model.layers.{i}.self_attn.q_norm.weight           [headDim]
 *   model.layers.{i}.self_attn.k_norm.weight           [headDim]
 *   model.layers.{i}.post_attention_layernorm.weight   [dModel]
 *   model.layers.{i}.mlp.gate_proj.weight              [dFfn, dModel]
 *   model.layers.{i}.mlp.up_proj.weight                [dFfn, dModel]
 *   model.layers.{i}.mlp.down_proj.weight              [dModel, dFfn]
 *   model.norm.weight                                  [dModel]
 *
 * 学习要点：
 * 1. Qwen3 全部 Linear 无 bias（attention_bias=false），用 Linear.of 构造。
 * 2. tie_word_embeddings=true 时没有 lm_head.weight，logits 投影复用 embed_tokens。
 * 3. RoPE 的 cos/sin 表与层无关（同 headDim、同 theta），全模型共享一个实例。
 * 4. 加载后校验：每个权重 shape 匹配、所有张量被消费、参数量与官方一致。
 */
public final class Qwen3Loader {

    private Qwen3Loader() {
    }

    /** 从 safetensors 权重字典构造 Qwen3Model（校验 shape 与张量完备性） */
    public static Qwen3Model load(ModelConfig cfg, Map<String, float[]> weights) {
        Set<String> consumed = new HashSet<>();
        Embedding wte = new Embedding(
                get(weights, consumed, "model.embed_tokens.weight", cfg.vocabSize * cfg.dModel),
                cfg.vocabSize, cfg.dModel);
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen, cfg.ropeTheta);

        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            String p = "model.layers." + i + ".";
            RmsNorm ln1 = rms(weights, consumed, p + "input_layernorm.weight", cfg.dModel, cfg.rmsNormEps);
            RmsNorm ln2 = rms(weights, consumed, p + "post_attention_layernorm.weight", cfg.dModel, cfg.rmsNormEps);
            Linear q = linear(weights, consumed, p + "self_attn.q_proj.weight", cfg.dModel, cfg.qDim());
            Linear k = linear(weights, consumed, p + "self_attn.k_proj.weight", cfg.dModel, cfg.kvDim());
            Linear v = linear(weights, consumed, p + "self_attn.v_proj.weight", cfg.dModel, cfg.kvDim());
            Linear o = linear(weights, consumed, p + "self_attn.o_proj.weight", cfg.qDim(), cfg.dModel);
            RmsNorm qNorm = rms(weights, consumed, p + "self_attn.q_norm.weight", cfg.headDim(), cfg.rmsNormEps);
            RmsNorm kNorm = rms(weights, consumed, p + "self_attn.k_norm.weight", cfg.headDim(), cfg.rmsNormEps);
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = linear(weights, consumed, p + "mlp.gate_proj.weight", cfg.dModel, cfg.dFfn);
            Linear up = linear(weights, consumed, p + "mlp.up_proj.weight", cfg.dModel, cfg.dFfn);
            Linear down = linear(weights, consumed, p + "mlp.down_proj.weight", cfg.dFfn, cfg.dModel);
            blocks[i] = new Qwen3Block(ln1, ln2, attn, new SwiGluFfn(gate, up, down), cfg.dModel);
        }
        RmsNorm lnF = rms(weights, consumed, "model.norm.weight", cfg.dModel, cfg.rmsNormEps);

        // 完备性校验：tied 模式下无 lm_head.weight；所有张量必须被消费
        Set<String> extra = new HashSet<>(weights.keySet());
        extra.removeAll(consumed);
        if (cfg.tieWordEmbeddings) {
            extra.remove("lm_head.weight"); // 个别导出会带一份拷贝，容忍
        }
        if (!extra.isEmpty()) {
            throw new IllegalArgumentException("存在未消费的权重张量: " + extra);
        }
        Qwen3Model model = new Qwen3Model(cfg, wte, blocks, lnF);
        long expect = expectedParams(cfg);
        if (model.numParameters() != expect) {
            throw new IllegalArgumentException(
                    "参数量不符: " + model.numParameters() + " 期望 " + expect);
        }
        return model;
    }

    /**
     * 从 bf16 位权重字典构造 Qwen3Model（大矩阵以 bf16 常驻，内存/带宽减半）。
     * 与 {@link #load} 等价，仅 embed_tokens 与各 Linear 权重以 short[] 存储；
     * RmsNorm 的 gamma（小数组）仍转回 f32。点积时逐元素加宽，数值与 F32 路径一致。
     */
    public static Qwen3Model loadBf16(ModelConfig cfg, Map<String, short[]> weights) {
        Set<String> consumed = new HashSet<>();
        Embedding wte = Embedding.ofBf16(
                getBits(weights, consumed, "model.embed_tokens.weight", cfg.vocabSize * cfg.dModel),
                cfg.vocabSize, cfg.dModel);
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen, cfg.ropeTheta);

        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            String p = "model.layers." + i + ".";
            RmsNorm ln1 = rmsBits(weights, consumed, p + "input_layernorm.weight", cfg.dModel, cfg.rmsNormEps);
            RmsNorm ln2 = rmsBits(weights, consumed, p + "post_attention_layernorm.weight", cfg.dModel, cfg.rmsNormEps);
            Linear q = linearBits(weights, consumed, p + "self_attn.q_proj.weight", cfg.dModel, cfg.qDim());
            Linear k = linearBits(weights, consumed, p + "self_attn.k_proj.weight", cfg.dModel, cfg.kvDim());
            Linear v = linearBits(weights, consumed, p + "self_attn.v_proj.weight", cfg.dModel, cfg.kvDim());
            Linear o = linearBits(weights, consumed, p + "self_attn.o_proj.weight", cfg.qDim(), cfg.dModel);
            RmsNorm qNorm = rmsBits(weights, consumed, p + "self_attn.q_norm.weight", cfg.headDim(), cfg.rmsNormEps);
            RmsNorm kNorm = rmsBits(weights, consumed, p + "self_attn.k_norm.weight", cfg.headDim(), cfg.rmsNormEps);
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = linearBits(weights, consumed, p + "mlp.gate_proj.weight", cfg.dModel, cfg.dFfn);
            Linear up = linearBits(weights, consumed, p + "mlp.up_proj.weight", cfg.dModel, cfg.dFfn);
            Linear down = linearBits(weights, consumed, p + "mlp.down_proj.weight", cfg.dFfn, cfg.dModel);
            blocks[i] = new Qwen3Block(ln1, ln2, attn, new SwiGluFfn(gate, up, down), cfg.dModel);
        }
        RmsNorm lnF = rmsBits(weights, consumed, "model.norm.weight", cfg.dModel, cfg.rmsNormEps);

        Set<String> extra = new HashSet<>(weights.keySet());
        extra.removeAll(consumed);
        if (cfg.tieWordEmbeddings) {
            extra.remove("lm_head.weight");
        }
        if (!extra.isEmpty()) {
            throw new IllegalArgumentException("存在未消费的权重张量: " + extra);
        }
        Qwen3Model model = new Qwen3Model(cfg, wte, blocks, lnF);
        long expect = expectedParams(cfg);
        if (model.numParameters() != expect) {
            throw new IllegalArgumentException(
                    "参数量不符: " + model.numParameters() + " 期望 " + expect);
        }
        return model;
    }

    /** 随机初始化（无权重文件时跑通流程，输出无意义） */
    public static Qwen3Model randomInit(ModelConfig cfg) {
        Random rnd = new Random(42L);
        Embedding wte = new Embedding(randN(rnd, cfg.vocabSize * cfg.dModel, 0.02f),
                cfg.vocabSize, cfg.dModel);
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen, cfg.ropeTheta);
        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            RmsNorm ln1 = rmsOnes(cfg.dModel, cfg.rmsNormEps);
            RmsNorm ln2 = rmsOnes(cfg.dModel, cfg.rmsNormEps);
            Linear q = Linear.of(randN(rnd, cfg.qDim() * cfg.dModel, 0.02f), cfg.dModel, cfg.qDim());
            Linear k = Linear.of(randN(rnd, cfg.kvDim() * cfg.dModel, 0.02f), cfg.dModel, cfg.kvDim());
            Linear v = Linear.of(randN(rnd, cfg.kvDim() * cfg.dModel, 0.02f), cfg.dModel, cfg.kvDim());
            Linear o = Linear.of(randN(rnd, cfg.dModel * cfg.qDim(), 0.02f), cfg.qDim(), cfg.dModel);
            RmsNorm qNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps);
            RmsNorm kNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps);
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = Linear.of(randN(rnd, cfg.dFfn * cfg.dModel, 0.02f), cfg.dModel, cfg.dFfn);
            Linear up = Linear.of(randN(rnd, cfg.dFfn * cfg.dModel, 0.02f), cfg.dModel, cfg.dFfn);
            Linear down = Linear.of(randN(rnd, cfg.dModel * cfg.dFfn, 0.02f), cfg.dFfn, cfg.dModel);
            blocks[i] = new Qwen3Block(ln1, ln2, attn, new SwiGluFfn(gate, up, down), cfg.dModel);
        }
        return new Qwen3Model(cfg, wte, blocks, rmsOnes(cfg.dModel, cfg.rmsNormEps));
    }

    /** 按配置计算期望参数量（tied：lm_head 不重复计） */
    public static long expectedParams(ModelConfig cfg) {
        long perLayer = 2L * cfg.dModel                       // ln1 + ln2
                + (long) cfg.qDim() * cfg.dModel              // q_proj
                + 2L * cfg.kvDim() * cfg.dModel               // k/v_proj
                + (long) cfg.dModel * cfg.qDim()              // o_proj
                + 2L * cfg.headDim()                          // q/k norm
                + 3L * cfg.dFfn * cfg.dModel;                 // gate/up/down
        return (long) cfg.vocabSize * cfg.dModel              // embed_tokens
                + perLayer * cfg.nLayer
                + cfg.dModel;                                 // final norm
    }

    // ===================== 辅助 =====================

    private static float[] get(Map<String, float[]> w, Set<String> consumed, String name, int expect) {
        float[] d = w.get(name);
        if (d == null) {
            throw new IllegalArgumentException("缺少权重: " + name);
        }
        if (d.length != expect) {
            throw new IllegalArgumentException("权重 " + name + " 长度 " + d.length + " 期望 " + expect);
        }
        consumed.add(name);
        return d;
    }

    // ---------- bf16 位版辅助 ----------

    private static short[] getBits(Map<String, short[]> w, Set<String> consumed, String name, int expect) {
        short[] d = w.get(name);
        if (d == null) {
            throw new IllegalArgumentException("缺少权重: " + name);
        }
        if (d.length != expect) {
            throw new IllegalArgumentException("权重 " + name + " 长度 " + d.length + " 期望 " + expect);
        }
        consumed.add(name);
        return d;
    }

    private static Linear linearBits(Map<String, short[]> w, Set<String> consumed, String name,
                                     int in, int out) {
        return Linear.ofBf16(getBits(w, consumed, name, out * in), in, out);
    }

    /** RmsNorm gamma 为小数组，从 bf16 位转回 f32 */
    private static RmsNorm rmsBits(Map<String, short[]> w, Set<String> consumed, String name,
                                   int dim, float eps) {
        short[] bits = getBits(w, consumed, name, dim);
        float[] g = new float[dim];
        for (int i = 0; i < dim; i++) {
            g[i] = Bf16.bf16ToFloat(bits[i] & 0xFFFF);
        }
        return new RmsNorm(g, eps);
    }

    private static RmsNorm rms(Map<String, float[]> w, Set<String> consumed, String name,
                               int dim, float eps) {
        return new RmsNorm(get(w, consumed, name, dim), eps);
    }

    private static Linear linear(Map<String, float[]> w, Set<String> consumed, String name,
                                 int in, int out) {
        return Linear.of(get(w, consumed, name, out * in), in, out);
    }

    private static RmsNorm rmsOnes(int dim, float eps) {
        float[] w = new float[dim];
        java.util.Arrays.fill(w, 1f);
        return new RmsNorm(w, eps);
    }

    /** Box-Muller 生成正态分布随机数 */
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
