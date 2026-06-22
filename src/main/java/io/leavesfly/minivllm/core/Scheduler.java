package io.leavesfly.minivllm.core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * Scheduler —— Continuous Batching 的队列管理器。
 *
 * 学习要点（对照 vLLM Scheduler）：
 * 1. 两个队列：waiting（待 prefill 的新请求）与 running（正在 decode 的请求）。
 * 2. 每个调度步：从 waiting 按"显存与并发上限"接纳(admit)新请求做 prefill，
 *    对 running 中请求各走一步 decode，再把完成的请求清扫(sweep)出去——
 *    这就是 continuous batching：请求随时进出，batch 始终尽量满。
 * 3. waiting 用并发安全队列，因为 HTTP 线程随时 add，engine 线程消费。
 */
public final class Scheduler {

    private final ConcurrentLinkedDeque<Sequence> waiting = new ConcurrentLinkedDeque<>();
    private final List<Sequence> running = new ArrayList<>();
    private final int maxNumSeqs;

    public Scheduler(int maxNumSeqs) {
        this.maxNumSeqs = maxNumSeqs;
    }

    /** HTTP 线程调用：加入新请求 */
    public void add(Sequence seq) {
        waiting.add(seq);
    }

    /** 待 prefill 队列（engine 线程读取） */
    public ConcurrentLinkedDeque<Sequence> waiting() {
        return waiting;
    }

    /** 正在运行的请求列表（engine 线程读写） */
    public List<Sequence> running() {
        return running;
    }

    public int maxNumSeqs() {
        return maxNumSeqs;
    }

    /** 是否还有工作要做 */
    public boolean hasWork() {
        return !waiting.isEmpty() || !running.isEmpty();
    }
}
