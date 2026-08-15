package io.leavesfly.minivllm.core;

import java.util.List;

/**
 * PromptLookup —— n-gram 自投机草稿生成（对照 vLLM 的 prompt_lookup 投机模式）。
 *
 * 学习要点：
 * 1. 投机采样需要 draft 模型与 target 模型同词表；当没有更小的同词表模型时，
 *    可以从序列自身的上下文里"查"草稿：LLM 生成代码/重复模式文本时，输出经常
 *    是上下文的局部复制——末尾 n 个 token 若在上下文中出现过，其后续 k 个 token
 *    就是高命中率的草稿。
 * 2. 查找规则（与 vLLM 一致）：needle 取序列末尾 n 个 token，n 从 3 递减到 1，
 *    在序列（排除末尾自身）中找最后一次出现，取其后的至多 maxDraft 个 token。
 * 3. 草稿被 target 模型验证后按最长一致前缀接受——greedy 下结果与普通 decode
 *    严格一致（投机是无损加速）；无匹配时返回 null，本步走普通 decode。
 */
public final class PromptLookup {

    /** needle 最大长度（vLLM prompt_lookup 默认 3） */
    private static final int MAX_NEEDLE = 3;

    private PromptLookup() {
    }

    /**
     * 查找草稿 token。
     *
     * @param promptTokens prompt 的 token 序列
     * @param outputTokens 已生成的 token 序列
     * @param maxDraft     草稿最大长度（vLLM 的 num_speculative_tokens）
     * @return 草稿 token（长度 ≤ maxDraft）；无匹配返回 null
     */
    public static int[] findDraft(int[] promptTokens, List<Integer> outputTokens, int maxDraft) {
        int totalLen = promptTokens.length + outputTokens.size();
        if (totalLen < 2 || maxDraft <= 0) {
            return null;
        }
        // 合并为单个序列便于下标操作（draft 可能跨越 prompt/输出边界）
        int[] seq = new int[totalLen];
        System.arraycopy(promptTokens, 0, seq, 0, promptTokens.length);
        for (int i = 0; i < outputTokens.size(); i++) {
            seq[promptTokens.length + i] = outputTokens.get(i);
        }
        for (int n = Math.min(MAX_NEEDLE, totalLen - 1); n >= 1; n--) {
            int needleStart = totalLen - n;
            // 在 seq[0, needleStart) 中从后往前找 needle 的最后一次出现
            for (int i = needleStart - 1; i >= 0; i--) {
                if (matches(seq, i, needleStart, n)) {
                    int from = i + n;
                    int len = Math.min(maxDraft, totalLen - from);
                    if (len <= 0) {
                        break; // 该 needle 长度下匹配点后续无内容，尝试更短 needle
                    }
                    int[] draft = new int[len];
                    System.arraycopy(seq, from, draft, 0, len);
                    return draft;
                }
            }
        }
        return null;
    }

    /** seq[aStart..aStart+len) 与 seq[bStart..bStart+len) 是否逐元素相等 */
    private static boolean matches(int[] seq, int aStart, int bStart, int len) {
        for (int j = 0; j < len; j++) {
            if (seq[aStart + j] != seq[bStart + j]) {
                return false;
            }
        }
        return true;
    }
}
