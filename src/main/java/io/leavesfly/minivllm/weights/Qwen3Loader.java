package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.math.ArrayUtil;
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

import io.leavesfly.minivllm.weights.Quantize.Int8Weight;

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


    // ─── 权重源抽象：统一 f32/bf16/int8 三种加载路径 ───

    /** 权重源接口：封装不同精度下的权重获取与转换逻辑 */
    private interface WeightSource {
        Embedding embedding(String name, int vocabSize, int dModel);
        Linear linear(String name, int in, int out);
        RmsNorm rmsNorm(String name, int dim, float eps);
        void markConsumed(String name);
        Set<String> allKeys();
    }

    /** 权重源抽象基类：统一管理 consumed 跟踪与校验逻辑 */
    private abstract static class AbstractWeightSource implements WeightSource {
        private final Set<String> consumed = new HashSet<>();

        @Override
        public void markConsumed(String name) { consumed.add(name); }

        Set<String> consumed() { return consumed; }

        /** 校验权重存在且长度匹配，标记已消费，返回原始数据 */
        protected <T> T requireWeight(String name, T data, int actualLen, int expectLen) {
            if (data == null) {
                throw new IllegalArgumentException("缺少权重: " + name);
            }
            if (actualLen != expectLen) {
                throw new IllegalArgumentException("权重 " + name + " 长度 " + actualLen + " 期望 " + expectLen);
            }
            consumed.add(name);
            return data;
        }
    }

    /** F32 权重源 */
    private static final class F32Source extends AbstractWeightSource {
        private final Map<String, float[]> weights;

        F32Source(Map<String, float[]> weights) {
            this.weights = weights;
        }

        @Override
        public Embedding embedding(String name, int vocabSize, int dModel) {
            return new Embedding(get(name, vocabSize * dModel), vocabSize, dModel);
        }

        @Override
        public Linear linear(String name, int in, int out) {
            return Linear.of(get(name, out * in), in, out);
        }

        @Override
        public RmsNorm rmsNorm(String name, int dim, float eps) {
            return new RmsNorm(get(name, dim), eps);
        }

        @Override
        public Set<String> allKeys() { return weights.keySet(); }

        private float[] get(String name, int expect) {
            float[] d = weights.get(name);
            return requireWeight(name, d, d == null ? 0 : d.length, expect);
        }
    }

    /** BF16 权重源 */
    private static final class Bf16Source extends AbstractWeightSource {
        private final Map<String, short[]> weights;

        Bf16Source(Map<String, short[]> weights) {
            this.weights = weights;
        }

        @Override
        public Embedding embedding(String name, int vocabSize, int dModel) {
            return Embedding.ofBf16(get(name, vocabSize * dModel), vocabSize, dModel);
        }

        @Override
        public Linear linear(String name, int in, int out) {
            return Linear.ofBf16(get(name, out * in), in, out);
        }

        @Override
        public RmsNorm rmsNorm(String name, int dim, float eps) {
            return new RmsNorm(bf16ToFloatArray(get(name, dim)), eps);
        }

        @Override
        public Set<String> allKeys() { return weights.keySet(); }

        private short[] get(String name, int expect) {
            short[] d = weights.get(name);
            return requireWeight(name, d, d == null ? 0 : d.length, expect);
        }
    }

    /** INT8 量化权重源（从 bf16 量化） */
    private static final class Int8Source extends AbstractWeightSource {
        private final Map<String, short[]> weights;

        Int8Source(Map<String, short[]> weights) {
            this.weights = weights;
        }

        @Override
        public Embedding embedding(String name, int vocabSize, int dModel) {
            short[] bits = get(name, vocabSize * dModel);
            Int8Weight q = Quantize.quantizeBf16(bits, vocabSize, dModel);
            return Embedding.ofInt8(q.data, q.scale, vocabSize, dModel);
        }

        @Override
        public Linear linear(String name, int in, int out) {
            short[] bits = get(name, out * in);
            Int8Weight q = Quantize.quantizeBf16(bits, out, in);
            return Linear.ofInt8(q.data, q.scale, in, out);
        }

        @Override
        public RmsNorm rmsNorm(String name, int dim, float eps) {
            return new RmsNorm(bf16ToFloatArray(get(name, dim)), eps);
        }

        @Override
        public Set<String> allKeys() { return weights.keySet(); }

        private short[] get(String name, int expect) {
            short[] d = weights.get(name);
            return requireWeight(name, d, d == null ? 0 : d.length, expect);
        }
    }

    /** bf16 位数组转 f32（RmsNorm gamma 等小数组共用） */
    private static float[] bf16ToFloatArray(short[] bits) {
        float[] out = new float[bits.length];
        for (int i = 0; i < bits.length; i++) {
            out[i] = Bf16.bf16ToFloat(bits[i] & 0xFFFF);
        }
        return out;
    }


    // ─── 公共加载入口 ───

    /** 从 safetensors 权重字典构造 Qwen3Model（校验 shape 与张量完备性） */
    public static Qwen3Model load(ModelConfig cfg, Map<String, float[]> weights) {
        F32Source source = new F32Source(weights);
        Qwen3Model model = buildModel(cfg, source);
        validateCompleteness(cfg, source.allKeys(), source.consumed());
        validateParams(cfg, model);
        return model;
    }

    /**
     * 从 bf16 位权重字典构造 Qwen3Model（大矩阵以 bf16 常驻，内存/带宽减半）。
     * 与 {@link #load} 等价，仅 embed_tokens 与各 Linear 权重以 short[] 存储；
     * RmsNorm 的 gamma（小数组）仍转回 f32。点积时逐元素加宽，数值与 F32 路径一致。
     */
    public static Qwen3Model loadBf16(ModelConfig cfg, Map<String, short[]> weights) {
        Bf16Source source = new Bf16Source(weights);
        Qwen3Model model = buildModel(cfg, source);
        validateCompleteness(cfg, source.allKeys(), source.consumed());
        validateParams(cfg, model);
        return model;
    }

    /**
     * 从 bf16 位权重字典构造 INT8 量化版 Qwen3Model。
     * 加载时先读 bf16，再逐张量 per-row 对称量化为 int8 + scale，
     * 最终模型权重仅占 bf16 的一半内存，decode 带宽减半。
     * RmsNorm gamma（小数组）仍转回 f32，不量化。
     */
    public static Qwen3Model loadInt8(ModelConfig cfg, Map<String, short[]> weights) {
        Int8Source source = new Int8Source(weights);
        Qwen3Model model = buildModel(cfg, source);
        validateCompleteness(cfg, source.allKeys(), source.consumed());
        return model;
    }

    /** 随机初始化（无权重文件时跑通流程，输出无意义） */
    public static Qwen3Model randomInit(ModelConfig cfg) {
        Random rnd = new Random(42L);
        Embedding wte = new Embedding(ArrayUtil.randNormal(rnd, cfg.vocabSize() * cfg.dModel(), 0.02f),
                cfg.vocabSize(), cfg.dModel());
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen(), cfg.ropeTheta());
        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer()];
        for (int i = 0; i < cfg.nLayer(); i++) {
            RmsNorm inputNorm = rmsOnes(cfg.dModel(), cfg.rmsNormEps());
            RmsNorm postAttnNorm = rmsOnes(cfg.dModel(), cfg.rmsNormEps());
            Linear q = Linear.of(ArrayUtil.randNormal(rnd, cfg.qDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.qDim());
            Linear k = Linear.of(ArrayUtil.randNormal(rnd, cfg.kvDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.kvDim());
            Linear v = Linear.of(ArrayUtil.randNormal(rnd, cfg.kvDim() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.kvDim());
            Linear o = Linear.of(ArrayUtil.randNormal(rnd, cfg.dModel() * cfg.qDim(), 0.02f), cfg.qDim(), cfg.dModel());
            RmsNorm qNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps());
            RmsNorm kNorm = rmsOnes(cfg.headDim(), cfg.rmsNormEps());
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = Linear.of(ArrayUtil.randNormal(rnd, cfg.dFfn() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.dFfn());
            Linear up = Linear.of(ArrayUtil.randNormal(rnd, cfg.dFfn() * cfg.dModel(), 0.02f), cfg.dModel(), cfg.dFfn());
            Linear down = Linear.of(ArrayUtil.randNormal(rnd, cfg.dModel() * cfg.dFfn(), 0.02f), cfg.dFfn(), cfg.dModel());
            blocks[i] = new Qwen3Block(inputNorm, postAttnNorm, attn, new SwiGluFfn(gate, up, down), cfg.dModel());
        }
        return new Qwen3Model(cfg, wte, blocks, rmsOnes(cfg.dModel(), cfg.rmsNormEps()));
    }

    /** 按配置计算期望参数量（tied：lm_head 不重复计） */
    public static long expectedParams(ModelConfig cfg) {
        long perLayer = 2L * cfg.dModel()                       // inputNorm + postAttnNorm
                + (long) cfg.qDim() * cfg.dModel()              // q_proj
                + 2L * cfg.kvDim() * cfg.dModel()               // k/v_proj
                + (long) cfg.dModel() * cfg.qDim()              // o_proj
                + 2L * cfg.headDim()                          // q/k norm
                + 3L * cfg.dFfn() * cfg.dModel();                 // gate/up/down
        return (long) cfg.vocabSize() * cfg.dModel()              // embed_tokens
                + perLayer * cfg.nLayer()
                + cfg.dModel();                                 // final norm
    }


    // ─── 统一骨架构建 ───

    /** 使用 WeightSource 构建模型骨架（消除 load/loadBf16/loadInt8 的重复代码） */
    private static Qwen3Model buildModel(ModelConfig cfg, WeightSource src) {
        Embedding wte = src.embedding("model.embed_tokens.weight", cfg.vocabSize(), cfg.dModel());
        RotaryEmbedding rope = new RotaryEmbedding(cfg.headDim(), cfg.maxSeqLen(), cfg.ropeTheta());

        Qwen3Block[] blocks = new Qwen3Block[cfg.nLayer()];
        for (int i = 0; i < cfg.nLayer(); i++) {
            String p = "model.layers." + i + ".";
            RmsNorm inputNorm = src.rmsNorm(p + "input_layernorm.weight", cfg.dModel(), cfg.rmsNormEps());
            RmsNorm postAttnNorm = src.rmsNorm(p + "post_attention_layernorm.weight", cfg.dModel(), cfg.rmsNormEps());
            Linear q = src.linear(p + "self_attn.q_proj.weight", cfg.dModel(), cfg.qDim());
            Linear k = src.linear(p + "self_attn.k_proj.weight", cfg.dModel(), cfg.kvDim());
            Linear v = src.linear(p + "self_attn.v_proj.weight", cfg.dModel(), cfg.kvDim());
            Linear o = src.linear(p + "self_attn.o_proj.weight", cfg.qDim(), cfg.dModel());
            RmsNorm qNorm = src.rmsNorm(p + "self_attn.q_norm.weight", cfg.headDim(), cfg.rmsNormEps());
            RmsNorm kNorm = src.rmsNorm(p + "self_attn.k_norm.weight", cfg.headDim(), cfg.rmsNormEps());
            Qwen3Attention attn = new Qwen3Attention(cfg, q, k, v, o, qNorm, kNorm, rope);
            Linear gate = src.linear(p + "mlp.gate_proj.weight", cfg.dModel(), cfg.dFfn());
            Linear up = src.linear(p + "mlp.up_proj.weight", cfg.dModel(), cfg.dFfn());
            Linear down = src.linear(p + "mlp.down_proj.weight", cfg.dFfn(), cfg.dModel());
            blocks[i] = new Qwen3Block(inputNorm, postAttnNorm, attn, new SwiGluFfn(gate, up, down), cfg.dModel());
        }
        RmsNorm finalNorm = src.rmsNorm("model.norm.weight", cfg.dModel(), cfg.rmsNormEps());
        return new Qwen3Model(cfg, wte, blocks, finalNorm);
    }


    // ─── 校验辅助 ───

    /** 完备性校验：tied 模式下无 lm_head.weight；所有张量必须被消费 */
    private static void validateCompleteness(ModelConfig cfg, Set<String> allKeys, Set<String> consumed) {
        Set<String> extra = new HashSet<>(allKeys);
        extra.removeAll(consumed);
        if (cfg.tieWordEmbeddings()) {
            extra.remove("lm_head.weight"); // 个别导出会带一份拷贝，容忍
        }
        if (!extra.isEmpty()) {
            throw new IllegalArgumentException("存在未消费的权重张量: " + extra);
        }
    }

    private static void validateParams(ModelConfig cfg, Qwen3Model model) {
        long expect = expectedParams(cfg);
        if (model.numParameters() != expect) {
            throw new IllegalArgumentException(
                    "参数量不符: " + model.numParameters() + " 期望 " + expect);
        }
    }


    // ─── 工具方法 ───

    private static RmsNorm rmsOnes(int dim, float eps) {
        float[] w = new float[dim];
        java.util.Arrays.fill(w, 1f);
        return new RmsNorm(w, eps);
    }
}
