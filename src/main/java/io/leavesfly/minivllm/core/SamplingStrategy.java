package io.leavesfly.minivllm.core;

/**
 * SamplingStrategy —— 从 logits 选出下一个 token 的策略接口。
 *
 * 学习要点：
 * 1. 引擎（LLMEngine）只依赖本接口，不关心具体算法：
 *    默认实现 {@link DefaultSamplingStrategy} 走 temperature/top-k/top-p 流程，
 *    未来可扩展 repetition penalty、beam search、结构化输出约束等。
 * 2. 参数随请求传入（每个 Sequence 有自己的 SamplingParams），
 *    策略实现自身不保存请求级状态，避免跨请求串扰。
 */
public interface SamplingStrategy {

    /**
     * 从词表 logits 中选出下一个 token。
     *
     * @param logits [vocabSize] 当前位置的词表 logits
     * @param params 当前请求的采样参数
     * @return 选中的 token id
     */
    int sample(float[] logits, SamplingParams params);
}
