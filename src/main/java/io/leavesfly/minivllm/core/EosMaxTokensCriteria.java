package io.leavesfly.minivllm.core;

/**
 * EosMaxTokensCriteria —— 默认停止判据：EOS 命中 或 生成数达到 maxTokens。
 *
 * 学习要点：
 * 序列级状态（EOS 集合、maxTokens、已生成数、阶段）都在 {@link Sequence} 上，
 * 因此 shouldStop 直接委托 {@link Sequence#isFinished()}，保持单一事实来源；
 * 本类的价值在于给引擎一个可替换的判断入口。
 */
public final class EosMaxTokensCriteria implements StopCriteria {

    @Override
    public boolean isStopToken(Sequence seq, int token) {
        for (int eos : seq.eosTokens()) {
            if (token == eos) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean shouldStop(Sequence seq) {
        return seq.isFinished();
    }
}
