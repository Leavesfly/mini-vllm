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
 * 2. 生命周期：WAITING（排队）→ PREFILL（处理 prompt）→ DECODE（逐 token 生成）→ FINISHED；
 *    显存不足时可能被抢占：DECODE/PREFILL → PREEMPTED（释放全部 KV block 退回 waiting
 *    队首，重新 admit 时以 prompt+已生成 tokens 为上下文重算——recompute 式抢占）。
 * 3. 采样参数聚合在 {@link SamplingParams}，对应 OpenAI API 的同名参数。
 * 4. onToken 回调用于流式输出：每生成一个 token 就解码并推送文本片段给客户端。
 */
public final class Sequence {

    public enum Stage { WAITING, PREFILL, DECODE, PREEMPTED, FINISHED, ABORTED }

    private final int id;
    private final int[] promptTokens;
    private final List<Integer> outputTokens = new ArrayList<>();
    /** 每层一个 BlockTable（KV cache 按层独立） */
    private final BlockTable[] blockTables;
    private volatile Stage stage = Stage.WAITING;

    /** 已完成 prefill 的上下文 token 数（前缀共享命中部分计入；支撑 Chunked Prefill 进度跟踪） */
    private int prefilledTokens;

    /** 被抢占次数（防饿死标记 + 观测；被抢占者重新 admit 时插入 waiting 队首优先重算） */
    private int preemptCount;

    /** 自抢占尝试标记：自抢占后仍分配不出说明池容不下本序列，再失败即 ABORT（防死循环） */
    private boolean selfPreempted;

    /** 请求加入时刻（TTFT 起点；由引擎 addRequest 时写入） */
    private volatile long arrivalNanos;

    /** 上一个生成 token 的时刻（ITL 打点基准） */
    private volatile long lastTokenNanos;

    /** 取消标记：客户端断连/显式 cancel 后置位，引擎下一步清扫并释放 KV */
    private volatile boolean cancelled;

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

    public int prefilledTokens() {
        return prefilledTokens;
    }

    public void setPrefilledTokens(int prefilledTokens) {
        this.prefilledTokens = prefilledTokens;
    }

    public int preemptCount() {
        return preemptCount;
    }

    public boolean selfPreempted() {
        return selfPreempted;
    }

    public void setSelfPreempted(boolean selfPreempted) {
        this.selfPreempted = selfPreempted;
    }

    public long arrivalNanos() {
        return arrivalNanos;
    }

    public void setArrivalNanos(long arrivalNanos) {
        this.arrivalNanos = arrivalNanos;
    }

    public long lastTokenNanos() {
        return lastTokenNanos;
    }

    public void setLastTokenNanos(long lastTokenNanos) {
        this.lastTokenNanos = lastTokenNanos;
    }

    /**
     * 请求取消（HTTP 断连/客户端放弃时调用，任意线程安全）：
     * 只置标记，实际 KV 释放与队列移除由引擎线程在下一步清扫时统一执行，
     * 避免跨线程碰 BlockTable/BlockPool 状态。
     */
    public void cancel() {
        cancelled = true;
    }

    public boolean cancelled() {
        return cancelled;
    }

    /** 被抢占时由引擎调用：计数 +1，进度归零（KV 已释放，需整体重算） */
    public void markPreempted() {
        preemptCount++;
        prefilledTokens = 0;
    }

    /** 当前完整上下文 token 数 = prompt + 已生成（新请求即 prompt 长度） */
    public int contextLen() {
        return promptTokens.length + outputTokens.size();
    }

    /**
     * 当前完整上下文 tokens = prompt + 已生成（重算式抢占重新 admit 后的 prefill 对象）。
     * 无已生成 token 时直接返回 promptTokens（零拷贝）。
     */
    public int[] contextTokens() {
        if (outputTokens.isEmpty()) {
            return promptTokens;
        }
        int[] all = new int[promptTokens.length + outputTokens.size()];
        System.arraycopy(promptTokens, 0, all, 0, promptTokens.length);
        for (int i = 0; i < outputTokens.size(); i++) {
            all[promptTokens.length + i] = outputTokens.get(i);
        }
        return all;
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
        if (stage == Stage.FINISHED || stage == Stage.ABORTED || cancelled) {
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
