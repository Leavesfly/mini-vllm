package io.leavesfly.minivllm.core;

import java.util.Map;

/**
 * SamplingParams —— 一次生成请求的采样参数（对照 vLLM 同名类）。
 *
 * 学习要点：
 * 1. 把采样参数聚合为一个不可变值对象，
 *    避免标量在 API → 引擎 → Sequence → Sampler 的签名中层层穿透。
 * 2. 默认值只在 {@link #DEFAULT} 一处定义，HTTP 层与引擎层共用，
 *    不会再出现两层各写一份、日后不一致的问题。
 * 3. 阶段五补齐主流引擎标配参数：repetition/frequency penalty、min-p、logit bias，
 *    语义与 HF transformers / vLLM 对齐（默认值均为“不生效”）。
 *
 * @param maxTokens         最多生成的 token 数
 * @param temperature       温度；≈0 时退化为 greedy（取 argmax）
 * @param topK              top-k 截断；0 表示不启用
 * @param topP              nucleus 采样阈值；1 表示不启用
 * @param repetitionPenalty 重复惩罚；1 表示不启用。>1 时对已出现 token 的 logit 正除以 r、负乘以 r
 * @param frequencyPenalty  频次惩罚；0 表示不启用。logit 减去 出现次数×该值（OpenAI 语义）
 * @param minP              min-p 截断：概率 < maxProb×minP 的 token 被滤掉；0 表示不启用
 * @param logitBias         token id → logit 加性偏置；null 表示不启用
 */
public record SamplingParams(int maxTokens, float temperature, int topK, float topP,
                             float repetitionPenalty, float frequencyPenalty, float minP,
                             Map<Integer, Float> logitBias) {

    /** 全局默认采样参数（OpenAI API 未显式传参时使用） */
    public static final SamplingParams DEFAULT = new SamplingParams(2048, 0.8f, 0, 0.9f);

    /** 兼容构造：仅基础四参数，新增参数取不生效默认值 */
    public SamplingParams(int maxTokens, float temperature, int topK, float topP) {
        this(maxTokens, temperature, topK, topP, 1.0f, 0.0f, 0.0f, null);
    }

    public SamplingParams {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens 必须 > 0，实际=" + maxTokens);
        }
    }

    /** 派生一个仅替换 maxTokens 的副本 */
    public SamplingParams withMaxTokens(int maxTokens) {
        return new SamplingParams(maxTokens, temperature, topK, topP,
                repetitionPenalty, frequencyPenalty, minP, logitBias);
    }
}
