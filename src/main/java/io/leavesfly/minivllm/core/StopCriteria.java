package io.leavesfly.minivllm.core;

/**
 * StopCriteria —— 判断生成何时停止、哪些 token 属于停止标记。
 *
 * 学习要点：
 * 1. 停止判断有两个独立关注点：
 *    - isStopToken：某个 token 是否为停止标记（EOS 等）。命中时引擎不把它
 *      解码成文本输出，否则 <|im_end|> 等会泄露到响应并污染多轮 ChatML 上下文。
 *    - shouldStop：整个序列是否应当结束（命中 EOS / 达到 maxTokens / 被中止）。
 * 2. 默认实现 {@link EosMaxTokensCriteria} 覆盖 OpenAI API 语义；
 *    扩展 stop strings（对照 OpenAI 的 stop 参数）时只需新增实现，引擎不变。
 */
public interface StopCriteria {

    /** token 是否为停止标记（命中时不输出对应文本） */
    boolean isStopToken(Sequence seq, int token);

    /** 序列是否应当结束（引擎 sweep 阶段调用） */
    boolean shouldStop(Sequence seq);
}
