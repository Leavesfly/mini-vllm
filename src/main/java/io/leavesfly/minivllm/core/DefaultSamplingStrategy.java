package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.math.Sampler;

/**
 * DefaultSamplingStrategy —— 默认采样策略：temperature → top-k → top-p → 多项采样。
 *
 * 学习要点：
 * 算法本体在 {@link Sampler}（math 包，可独立单测），本类只做适配：
 * 把「configure + sample 两步调用」合并为一次原子调用，
 * 消除引擎循环里共享可变状态的隐患（引擎线程单线程驱动，因此无需加锁）。
 */
public final class DefaultSamplingStrategy implements SamplingStrategy {

    private final Sampler sampler;

    public DefaultSamplingStrategy(long seed) {
        this.sampler = new Sampler(seed);
    }

    @Override
    public int sample(float[] logits, SamplingParams params) {
        sampler.configure(params.temperature(), params.topK(), params.topP());
        return sampler.sample(logits);
    }
}
