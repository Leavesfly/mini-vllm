package io.leavesfly.minivllm.core;

import java.util.List;

/**
 * SamplingStrategy —— 从 logits 选出下一个 token 的策略接口。
 *
 * 学习要点：
 * 1. 引擎（LLMEngine）只依赖本接口，不关心具体算法：
 *    默认实现 {@link DefaultSamplingStrategy} 走 penalty + temperature/top-k/top-p/min-p 流程，
 *    未来可扩展 beam search、结构化输出约束等。
 * 2. 参数随请求传入（每个 Sequence 有自己的 SamplingParams），
 *    策略实现自身不保存请求级状态，避免跨请求串扰。
 */
public interface SamplingStrategy {

    /**
     * 从词表 logits 中选出下一个 token（无历史上下文，不应用 penalty）。
     *
     * @param logits [vocabSize] 当前位置的词表 logits
     * @param params 当前请求的采样参数
     * @return 选中的 token id
     */
    int sample(float[] logits, SamplingParams params);

    /**
     * 带历史上下文的采样：repetition/frequency penalty 依赖已出现的 token。
     * logits 为模型刚产出的新数组，实现可就地修改。
     * 默认实现忽略历史（兼容旧策略）；引擎统一走本方法。
     *
     * @param promptTokens 请求的 prompt tokens（repetition penalty 覆盖 prompt+输出，对照 vLLM）
     * @param outputTokens 已生成的 tokens（frequency penalty 仅计生成部分，对照 OpenAI）
     */
    default int sample(float[] logits, SamplingParams params,
                       int[] promptTokens, List<Integer> outputTokens) {
        return sample(logits, params);
    }
}
