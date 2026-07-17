package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Sequence —— 一个推理请求的完整运行时状态。
 *
 * 学习要点（对照 vLLM SequenceGroup）：
 * 1. 每个请求持有一组 BlockTable（每层一个），因为每层注意力有独立的 KV cache。
 * 2. 生命周期：WAITING（排队）→ PREFILL（处理 prompt）→ DECODE（逐 token 生成）→ FINISHED。
 * 3. 生成参数(maxTokens/temperature/topK/topP)对应 OpenAI API 的同名参数。
 * 4. onToken 回调用于流式输出：每生成一个 token 就解码并推送文本片段给客户端。
 */
public final class Sequence {

    public enum Stage { WAITING, PREFILL, DECODE, FINISHED, ABORTED }

    public final int id;
    public final int[] promptTokens;
    public final List<Integer> outputTokens = new ArrayList<>();
    /** 每层一个 BlockTable（KV cache 按层独立） */
    public final BlockTable[] blockTables;
    public volatile Stage stage = Stage.WAITING;

    // 生成参数（对应 OpenAI API）
    public final int maxTokens;
    public final float temperature;
    public final int topK;
    public final float topP;
    public final int[] eosTokens; // 空数组表示无 EOS

    /** 流式回调：每生成一个 token 解码后触发 */
    public final Consumer<String> onToken;

    /** BPE 分词时的流式增量解码器（由引擎注入，处理跨 token UTF-8 边界） */
    public volatile BpeTokenizer.IncrementalDecoder incDecoder;

    public Sequence(int id, int[] promptTokens, int maxTokens,
                    float temperature, int topK, float topP, int eosToken,
                    int nLayer, Consumer<String> onToken) {
        this(id, promptTokens, maxTokens, temperature, topK, topP,
                eosToken < 0 ? new int[0] : new int[]{eosToken}, nLayer, onToken);
    }

    public Sequence(int id, int[] promptTokens, int maxTokens,
                    float temperature, int topK, float topP, int[] eosTokens,
                    int nLayer, Consumer<String> onToken) {
        this.id = id;
        this.promptTokens = promptTokens;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.topK = topK;
        this.topP = topP;
        this.eosTokens = eosTokens;
        this.onToken = onToken;
        this.blockTables = new BlockTable[nLayer];
        for (int i = 0; i < nLayer; i++) {
            blockTables[i] = new BlockTable();
        }
    }

    /** 已生成的 token 是否达到上限或命中 EOS */
    public boolean isFinished() {
        if (stage == Stage.FINISHED || stage == Stage.ABORTED) {
            return true;
        }
        if (outputTokens.size() >= maxTokens) {
            return true;
        }
        if (!outputTokens.isEmpty()) {
            int last = outputTokens.get(outputTokens.size() - 1);
            for (int eos : eosTokens) {
                if (last == eos) {
                    return true;
                }
            }
        }
        return false;
    }

    /** 当前序列总 token 数 = prompt + 已生成 */
    public int totalLen() {
        return promptTokens.length + outputTokens.size();
    }
}
