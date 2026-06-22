package io.leavesfly.minivllm.math;

import java.util.Random;

/**
 * 采样器 —— 从 logits 生成下一个 token。
 *
 * 学习要点：
 * 1. greedy：直接取 argmax，输出确定，适合对比基准。
 * 2. temperature：缩放 logits 后采样，T<1 更确定，T>1 更随机。
 * 3. top-k：只在概率最高的 k 个里采样，过滤长尾噪声。
 * 4. top-p(nucleus)：只在累积概率达到 p 的最小集合里采样，动态裁剪。
 * 实际服务中这些参数由 OpenAI API 的 temperature/top_p/top_k 传入。
 */
public final class Sampler {

    private final Random random;
    public float temperature = 1.0f;
    public int topK = 0;   // 0 表示不限制
    public float topP = 1.0f; // 1.0 表示不限制

    public Sampler(long seed) {
        this.random = new Random(seed);
    }

    /** greedy：返回最大 logits 的下标 */
    public static int argmax(float[] logits) {
        int best = 0;
        float max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
                best = i;
            }
        }
        return best;
    }

    /**
     * 完整采样流程：temperature → top-k → top-p → 多项采样。
     * @return 选中的 token id
     */
    public int sample(float[] logits) {
        // temperature 极小时退化为 greedy，避免数值问题
        if (temperature <= 1e-3f || (topK == 1)) {
            return argmax(logits);
        }

        float[] probs = Softmax.softmaxWithTemp(logits, temperature);
        int[] indices = indexSort(probs); // 按概率降序排列的下标

        // top-k 裁剪
        int limit = indices.length;
        if (topK > 0 && topK < limit) {
            limit = topK;
        }
        // top-p 裁剪：累积概率达到 topP 即止
        float cum = 0f;
        int cut = limit;
        for (int i = 0; i < limit; i++) {
            cum += probs[indices[i]];
            if (cum >= topP) {
                cut = i + 1;
                break;
            }
        }
        limit = cut;

        // 在候选集合内重新归一化并多项采样
        float sum = 0f;
        for (int i = 0; i < limit; i++) {
            sum += probs[indices[i]];
        }
        float r = random.nextFloat() * sum;
        cum = 0f;
        for (int i = 0; i < limit; i++) {
            cum += probs[indices[i]];
            if (r <= cum) {
                return indices[i];
            }
        }
        return indices[limit - 1]; // 浮点兜底
    }

    /** 返回按 probs 降序排列的原下标数组（朴素选择，词表不大够用） */
    private static int[] indexSort(float[] probs) {
        int n = probs.length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // 插入排序：词表通常不大，且只需前若干个；学习项目用简单实现
        for (int i = 1; i < n; i++) {
            int cur = idx[i];
            float curP = probs[cur];
            int j = i - 1;
            while (j >= 0 && probs[idx[j]] < curP) {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = cur;
        }
        return idx;
    }
}
