package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.tokenizer.IncrementalDecoder;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

/**
 * Sequence —— 一个推理请求的完整运行时状态。
 *
 * 学习要点（对照 vLLM SequenceGroup）：
 * 1. 每个请求持有一组 BlockTable（每层一个），因为每层注意力有独立的 KV cache。
 * 2. 生命周期：WAITING（排队）→ PREFILL（处理 prompt）→ DECODE（逐 token 生成）→ FINISHED。
 * 3. 采样参数聚合在 {@link SamplingParams}，对应 OpenAI API 的同名参数。
 * 4. onToken 回调用于流式输出：每生成一个 token 就解码并推送文本片段给客户端。
 */
public final class Sequence {

    public enum Stage { WAITING, PREFILL, DECODE, FINISHED, ABORTED }

    private final int id;
    private final int[] promptTokens;
    private final List<Integer> outputTokens = new ArrayList<>();
    /** 每层一个 BlockTable（KV cache 按层独立） */
    private final BlockTable[] blockTables;
    private volatile Stage stage = Stage.WAITING;

    // 采样参数（对应 OpenAI API）
    private final SamplingParams params;
    private final int[] eosTokens; // 空数组表示无 EOS

    /** 流式回调：每生成一个 token 解码后触发 */
    private final Consumer<String> onToken;

    /** 流式增量解码器（由引擎注入，处理跨 token UTF-8 边界） */
    private volatile IncrementalDecoder incDecoder;

    /** 完成信号：请求结束时 countDown，等待方可用 awaitDone() 阻塞而非轮询 */
    private final CountDownLatch done = new CountDownLatch(1);

    public Sequence(int id, int[] promptTokens, SamplingParams params, int[] eosTokens,
                    int nLayer, Consumer<String> onToken) {
        this.id = id;
        this.promptTokens = promptTokens;
        this.params = params;
        this.eosTokens = eosTokens;
        this.onToken = onToken;
        this.blockTables = new BlockTable[nLayer];
        for (int i = 0; i < nLayer; i++) {
            blockTables[i] = new BlockTable();
        }
    }


    // ─── 访问器 ───

    public int id() {
        return id;
    }

    public int[] promptTokens() {
        return promptTokens;
    }

    public List<Integer> outputTokens() {
        return outputTokens;
    }

    public BlockTable[] blockTables() {
        return blockTables;
    }

    public Stage stage() {
        return stage;
    }

    public void setStage(Stage stage) {
        this.stage = stage;
    }

    public SamplingParams params() {
        return params;
    }

    public int[] eosTokens() {
        return eosTokens;
    }

    public Consumer<String> onToken() {
        return onToken;
    }

    public IncrementalDecoder incDecoder() {
        return incDecoder;
    }

    public void setIncDecoder(IncrementalDecoder incDecoder) {
        this.incDecoder = incDecoder;
    }

    /** 已生成的 token 是否达到上限或命中 EOS */
    public boolean isFinished() {
        if (stage == Stage.FINISHED || stage == Stage.ABORTED) {
            return true;
        }
        if (outputTokens.size() >= params.maxTokens()) {
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

    /** 标记完成（由引擎 sweep 时调用），触发等待方的 awaitDone() 返回 */
    public void markDone() {
        done.countDown();
    }

    /** 阻塞等待请求完成（替代 Thread.sleep 轮询） */
    public void awaitDone() throws InterruptedException {
        done.await();
    }

    /** 当前序列总 token 数 = prompt + 已生成 */
    public int totalLen() {
        return promptTokens.length + outputTokens.size();
    }
}
