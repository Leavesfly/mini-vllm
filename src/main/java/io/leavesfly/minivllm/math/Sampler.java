package io.leavesfly.minivllm.math;

import java.util.Comparator;
import java.util.PriorityQueue;
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
    private float temperature = 1.0f;
    private int topK = 0;      // 0 表示不限制
    private float topP = 1.0f; // 1.0 表示不限制

    public Sampler(long seed) {
        this.random = new Random(seed);
    }

    /** 配置采样参数（由引擎在每次采样前调用） */
    public void configure(float temperature, int topK, float topP) {
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
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
     *
     * 性能要点（词表 151936 时必须考虑）：
     * - 无截断（topK=0 且 topP>=1）时直接全词表多项采样，O(n) 无需排序；
     * - 有截断时用大小为 k 的最小堆做 top-k 选择，O(n log k)，
     *   避免对 15 万元素做全排序（原插入排序 O(n²) 在大词表下不可用）。
     *
     * @return 选中的 token id
     */
    public int sample(float[] logits) {
        // temperature 极小时退化为 greedy，避免数值问题
        if (temperature <= 1e-3f || (topK == 1)) {
            return argmax(logits);
        }

        float[] probs = Softmax.softmaxWithTemp(logits, temperature);
        int n = probs.length;
        int k = topK > 0 ? Math.min(topK, n) : n;

        if (k == n && topP >= 1f) {
            // 无截断：全词表多项采样
            float sum = 0f;
            for (float p : probs) {
                sum += p;
            }
            float r = random.nextFloat() * sum;
            float cum = 0f;
            for (int i = 0; i < n; i++) {
                cum += probs[i];
                if (r <= cum) {
                    return i;
                }
            }
            return n - 1; // 浮点兜底
        }

        // top-k 堆选择（候选按概率降序），再按 top-p 截断
        int[] cand = topKIndices(probs, k);
        float sum = 0f;
        int cut = cand.length;
        for (int i = 0; i < cand.length; i++) {
            sum += probs[cand[i]];
            if (sum >= topP) {
                cut = i + 1;
                break;
            }
        }

        // 在候选集合内重新归一化并多项采样
        float r = random.nextFloat() * sum;
        float cum = 0f;
        for (int i = 0; i < cut; i++) {
            cum += probs[cand[i]];
            if (r <= cum) {
                return cand[i];
            }
        }
        return cand[cut - 1]; // 浮点兜底
    }

    /** 用大小为 k 的最小堆选出概率最高的 k 个下标，按概率降序返回 */
    private static int[] topKIndices(float[] probs, int k) {
        PriorityQueue<Integer> heap = new PriorityQueue<>(k,
                Comparator.comparingDouble(i -> probs[i]));
        for (int i = 0; i < probs.length; i++) {
            if (heap.size() < k) {
                heap.offer(i);
            } else if (probs[i] > probs[heap.peek()]) {
                heap.poll();
                heap.offer(i);
            }
        }
        int[] out = new int[heap.size()];
        for (int i = out.length - 1; i >= 0; i--) {
            out[i] = heap.poll(); // 堆顶最小，逆序得降序
        }
        return out;
    }
}
