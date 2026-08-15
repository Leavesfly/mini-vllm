package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.math.Sampler;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * DefaultSamplingStrategy —— 默认采样策略：
 * penalty（repetition/frequency/logit bias）→ temperature → min-p → top-k → top-p → 多项采样。
 *
 * 学习要点：
 * 1. 算法本体在 {@link Sampler}（math 包，可独立单测），penalty 属“上下文相关”的
 *    logit 变换，在本类应用后再交给 Sampler——对照 vLLM Sampler 的 apply_penalties 阶段。
 * 2. repetition penalty 覆盖 prompt+已生成 token（vLLM 语义）；正 logit 除以 r、
 *    负 logit 乘以 r（r>1 时双向压低）；每个 token 只罚一次。
 * 3. frequency penalty 仅计已生成 token 的出现次数（OpenAI 语义）：logit -= count × 参数。
 * 4. 「configure + sample 两步调用」合并为一次原子调用，
 *    消除引擎循环里共享可变状态的隐患（引擎线程单线程驱动，因此无需加锁）。
 */
public final class DefaultSamplingStrategy implements SamplingStrategy {

    private final Sampler sampler;

    public DefaultSamplingStrategy(long seed) {
        this.sampler = new Sampler(seed);
    }

    @Override
    public int sample(float[] logits, SamplingParams params) {
        sampler.configure(params.temperature(), params.topK(), params.topP(), params.minP());
        return sampler.sample(logits);
    }

    @Override
    public int sample(float[] logits, SamplingParams params,
                      int[] promptTokens, List<Integer> outputTokens) {
        applyPenalties(logits, params, promptTokens, outputTokens);
        return sample(logits, params);
    }

    /** 对 logits 就地应用 repetition / frequency penalty 与 logit bias（均为不生效默认值时零开销返回） */
    private static void applyPenalties(float[] logits, SamplingParams params,
                                       int[] promptTokens, List<Integer> outputTokens) {
        float rep = params.repetitionPenalty();
        float freq = params.frequencyPenalty();
        Map<Integer, Float> bias = params.logitBias();
        boolean useRep = rep != 1.0f;
        boolean useFreq = freq != 0.0f;
        boolean useBias = bias != null && !bias.isEmpty();
        if (!useRep && !useFreq && !useBias) {
            return;
        }
        if (useRep) {
            // 每个 token 只罚一次（重复施加会指数压低，非预期语义）
            Set<Integer> seen = new HashSet<>();
            for (int t : promptTokens) {
                if (seen.add(t)) {
                    penalizeRepetition(logits, t, rep);
                }
            }
            for (int t : outputTokens) {
                if (seen.add(t)) {
                    penalizeRepetition(logits, t, rep);
                }
            }
        }
        if (useFreq) {
            Map<Integer, Integer> counts = new HashMap<>();
            for (int t : outputTokens) {
                counts.merge(t, 1, Integer::sum);
            }
            for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
                int t = e.getKey();
                if (t >= 0 && t < logits.length) {
                    logits[t] -= freq * e.getValue();
                }
            }
        }
        if (useBias) {
            for (Map.Entry<Integer, Float> e : bias.entrySet()) {
                int t = e.getKey();
                if (t >= 0 && t < logits.length) {
                    logits[t] += e.getValue();
                }
            }
        }
    }

    /** repetition penalty：正 logit 除以 r，负 logit 乘以 r（r>1 双向压低，与 HF/vLLM 一致） */
    private static void penalizeRepetition(float[] logits, int token, float r) {
        if (token < 0 || token >= logits.length) {
            return;
        }
        logits[token] = logits[token] > 0 ? logits[token] / r : logits[token] * r;
    }
}
