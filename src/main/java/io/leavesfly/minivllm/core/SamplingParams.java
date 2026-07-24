package io.leavesfly.minivllm.core;

/**
 * SamplingParams —— 一次生成请求的采样参数（对照 vLLM 同名类）。
 *
 * 学习要点：
 * 1. 把 maxTokens/temperature/topK/topP 聚合为一个不可变值对象，
 *    避免四个标量在 API → 引擎 → Sequence → Sampler 的签名中层层穿透。
 * 2. 默认值只在 {@link #DEFAULT} 一处定义，HTTP 层与引擎层共用，
 *    不会再出现两层各写一份、日后不一致的问题。
 *
 * @param maxTokens   最多生成的 token 数
 * @param temperature 温度；≈0 时退化为 greedy（取 argmax）
 * @param topK        top-k 截断；0 表示不启用
 * @param topP        nucleus 采样阈值；1 表示不启用
 */
public record SamplingParams(int maxTokens, float temperature, int topK, float topP) {

    /** 全局默认采样参数（OpenAI API 未显式传参时使用） */
    public static final SamplingParams DEFAULT = new SamplingParams(512, 0.8f, 0, 0.9f);

    public SamplingParams {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("maxTokens 必须 > 0，实际=" + maxTokens);
        }
    }

    /** 派生一个仅替换 maxTokens 的副本 */
    public SamplingParams withMaxTokens(int maxTokens) {
        return new SamplingParams(maxTokens, temperature, topK, topP);
    }
}
