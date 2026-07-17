package io.leavesfly.minivllm.model;

/**
 * 模型超参数配置 —— 同时支持 GPT-2 与 GPT-3 架构。
 *
 * GPT-3 与 GPT-2 的关系（见论文 "Language Models are Few-Shot Learners"）：
 *   "same model and architecture as GPT-2 ... with the exception that we use
 *    alternating dense and locally banded sparse attention patterns"。
 * 也就是说 GPT-3 = GPT-2 结构 + 交替的「稠密 / 局部带状稀疏」注意力，外加更大的规模。
 *
 * 学习要点：
 * 1. dModel 是隐藏层维度，nHead 是注意力头数，headDim = dModel / nHead。
 * 2. dFfn 是 FFN 中间层维度，GPT-2/GPT-3 通常取 4 * dModel。
 * 3. blockSize 是 PagedAttention 的物理块大小（每个 block 容纳的 token 数）。
 * 4. useSparseAttention 打开后，奇数层退化为只看最近 sparseWindow 个 token 的局部注意力（GPT-3）。
 * 5. 学习项目用小尺寸即可跑通，例如 dModel=64, nHead=4, nLayer=2, vocabSize=256。
 */
public final class ModelConfig {

    public int vocabSize = 256;
    public int dModel = 64;
    public int nHead = 4;
    public int nLayer = 2;
    public int dFfn = 256;     // 4 * dModel
    public int blockSize = 16; // PagedAttention 块大小
    public int maxSeqLen = 512;
    public float layerNormEps = 1e-5f;
    public boolean tieWordEmbeddings = true; // GPT-2/GPT-3 词嵌入与 lm_head 共享权重

    // ---------- Qwen3 / 现代 LLM 架构字段 ----------
    /** 架构标识："gpt2"（含 GPT-3 变体）或 "qwen3" */
    public String arch = "gpt2";
    /** GQA 的 KV 头数；0 表示与 nHead 相同（即 MHA） */
    public int nKVHead = 0;
    /** 显式注意力头维度；0 表示 dModel/nHead。Qwen3 为 128（与 dModel 无关） */
    public int headDimExplicit = 0;
    /** RoPE 的 theta 基数（Qwen3 为 1000000） */
    public float ropeTheta = 10000f;
    /** RMSNorm 的 eps（Qwen3 为 1e-6） */
    public float rmsNormEps = 1e-6f;
    /** EOS token id 集合（Qwen3 为 [151645, 151643]） */
    public int[] eosTokenIds = new int[0];

    // ---------- GPT-3 相关（相对 GPT-2 的唯一架构差异 + 规模预设）----------
    /**
     * 是否启用交替「稠密 / 局部带状稀疏」注意力（GPT-3 的唯一架构区别）。
     * 关闭时即纯 GPT-2 稠密注意力。
     */
    public boolean useSparseAttention = false;
    /** 局部带状窗口：sparse 层每个 query 仅关注最近 sparseWindow 个 token */
    public int sparseWindow = 128;
    /** 模型名（便于打印/区分预设，如 "gpt3-small"） */
    public String name = "mini";

    /** 每个注意力头的维度（Qwen3 用显式 headDim，否则 dModel/nHead） */
    public int headDim() {
        return headDimExplicit > 0 ? headDimExplicit : dModel / nHead;
    }

    /** KV 头数（GQA）；未设置时等于 nHead（MHA） */
    public int kvHeads() {
        return nKVHead > 0 ? nKVHead : nHead;
    }

    /** 每 token 的 K/V 向量长度 = kvHeads * headDim（KV cache 的存储维度） */
    public int kvDim() {
        return kvHeads() * headDim();
    }

    /** Q 投影输出维度 = nHead * headDim（Qwen3 中 ≠ dModel） */
    public int qDim() {
        return nHead * headDim();
    }

    /**
     * 第 layer 层是否使用稀疏注意力。
     * GPT-3 采用交替模式：偶数层 dense，奇数层 locally banded sparse。
     */
    public boolean isSparseLayer(int layer) {
        return useSparseAttention && (layer & 1) == 1;
    }

    /**
     * GPT-2/GPT-3 "modified initialization"：残差路径上的投影权重
     * (attn.o_proj / mlp.fc2) 用更小的标准差初始化，std = baseStd / sqrt(2 * nLayer)，
     * 以抵消残差累加带来的方差增长（论文中的 modified initialization）。
     */
    public float residualInitStd(float baseStd) {
        return baseStd / (float) Math.sqrt(2.0 * nLayer);
    }

    /** 便捷构造（学习用小模型，GPT-2 风格 dense 注意力） */
    public static ModelConfig small() {
        return new ModelConfig();
    }

    /**
     * 学习用 GPT-3 风格小模型：开启交替稀疏注意力，规模仍可在 CPU 秒级跑通。
     * 配合 ByteTokenizer（字节级词表）即可直接运行。
     */
    public static ModelConfig gpt3Nano() {
        ModelConfig c = new ModelConfig();
        c.name = "gpt3-nano";
        c.vocabSize = 256;   // 字节级词表，配合 ByteTokenizer
        c.dModel = 128;
        c.nHead = 4;
        c.nLayer = 4;
        c.dFfn = 512;
        c.blockSize = 16;
        c.maxSeqLen = 512;
        c.useSparseAttention = true;
        c.sparseWindow = 64;
        return c;
    }

    // ---------- 标准 GPT-3 规模预设（vocab=50257, nCtx=2048, 需 BPE 分词与真实权重）----------
    // dFfn 均为 4*dModel；交替稠密/稀疏注意力默认开启。

    /** GPT-3 Small (125M)：L=12, d=768, h=12 */
    public static ModelConfig gpt3Small() {
        return gpt3("gpt3-small", 12, 768, 12);
    }

    /** GPT-3 Medium (350M)：L=24, d=1024, h=16 */
    public static ModelConfig gpt3Medium() {
        return gpt3("gpt3-medium", 24, 1024, 16);
    }

    /** GPT-3 Large (760M)：L=24, d=1536, h=16 */
    public static ModelConfig gpt3Large() {
        return gpt3("gpt3-large", 24, 1536, 16);
    }

    /** GPT-3 XL (1.3B)：L=24, d=2048, h=24 */
    public static ModelConfig gpt3XL() {
        return gpt3("gpt3-xl", 24, 2048, 24);
    }

    /** GPT-3 2.7B：L=32, d=2560, h=32 */
    public static ModelConfig gpt3_2_7B() {
        return gpt3("gpt3-2.7B", 32, 2560, 32);
    }

    /** GPT-3 6.7B：L=32, d=4096, h=32 */
    public static ModelConfig gpt3_6_7B() {
        return gpt3("gpt3-6.7B", 32, 4096, 32);
    }

    /** GPT-3 13B：L=40, d=5120, h=40 */
    public static ModelConfig gpt3_13B() {
        return gpt3("gpt3-13B", 40, 5120, 40);
    }

    /** GPT-3 175B（davinci）：L=96, d=12288, h=96 */
    public static ModelConfig gpt3_175B() {
        return gpt3("gpt3-175B", 96, 12288, 96);
    }

    /**
     * 从 HuggingFace config.json 解析 Qwen3 配置。
     * 对应字段：hidden_size / num_attention_heads / num_key_value_heads / head_dim /
     * num_hidden_layers / intermediate_size / vocab_size / rope_theta / rms_norm_eps /
     * max_position_embeddings / tie_word_embeddings / eos_token_id。
     */
    public static ModelConfig fromConfigJson(java.util.Map<String, Object> json) {
        ModelConfig c = new ModelConfig();
        c.arch = "qwen3";
        c.name = String.valueOf(json.getOrDefault("model_type", "qwen3"));
        c.vocabSize = intOf(json, "vocab_size", 151936);
        c.dModel = intOf(json, "hidden_size", 1024);
        c.nHead = intOf(json, "num_attention_heads", 16);
        c.nKVHead = intOf(json, "num_key_value_heads", c.nHead);
        c.headDimExplicit = intOf(json, "head_dim", c.dModel / c.nHead);
        c.nLayer = intOf(json, "num_hidden_layers", 28);
        c.dFfn = intOf(json, "intermediate_size", 3072);
        c.maxSeqLen = intOf(json, "max_position_embeddings", 40960);
        c.ropeTheta = floatOf(json, "rope_theta", 1000000f);
        c.rmsNormEps = floatOf(json, "rms_norm_eps", 1e-6f);
        c.tieWordEmbeddings = boolOf(json, "tie_word_embeddings", true);
        Object eos = json.get("eos_token_id");
        if (eos instanceof java.util.List) {
            java.util.List<?> list = (java.util.List<?>) eos;
            c.eosTokenIds = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                c.eosTokenIds[i] = ((Number) list.get(i)).intValue();
            }
        } else if (eos instanceof Number) {
            c.eosTokenIds = new int[]{((Number) eos).intValue()};
        }
        return c;
    }

    private static int intOf(java.util.Map<String, Object> j, String k, int def) {
        Object v = j.get(k);
        return v instanceof Number ? ((Number) v).intValue() : def;
    }

    private static float floatOf(java.util.Map<String, Object> j, String k, float def) {
        Object v = j.get(k);
        return v instanceof Number ? ((Number) v).floatValue() : def;
    }

    private static boolean boolOf(java.util.Map<String, Object> j, String k, boolean def) {
        Object v = j.get(k);
        return v instanceof Boolean ? (Boolean) v : def;
    }

    /** 按 GPT-3 论文规格构造标准预设 */
    private static ModelConfig gpt3(String name, int nLayer, int dModel, int nHead) {
        ModelConfig c = new ModelConfig();
        c.name = name;
        c.vocabSize = 50257;   // GPT-2/GPT-3 BPE 词表
        c.dModel = dModel;
        c.nHead = nHead;
        c.nLayer = nLayer;
        c.dFfn = 4 * dModel;
        c.maxSeqLen = 2048;    // GPT-3 上下文窗口 nCtx
        c.blockSize = 16;
        c.useSparseAttention = true; // GPT-3 交替稠密/稀疏
        c.sparseWindow = 128;
        return c;
    }

    @Override
    public String toString() {
        return "ModelConfig{name=" + name + ", vocab=" + vocabSize + ", d=" + dModel
                + ", h=" + nHead + ", L=" + nLayer + ", ffn=" + dFfn + ", blk=" + blockSize
                + ", sparse=" + useSparseAttention
                + (useSparseAttention ? "(w=" + sparseWindow + ")" : "") + "}";
    }
}
