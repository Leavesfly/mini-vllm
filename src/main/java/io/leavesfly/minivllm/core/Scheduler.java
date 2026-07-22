package io.leavesfly.minivllm.core;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.function.Predicate;

/**
 * Scheduler —— Continuous Batching 的队列管理器。
 *
 * 学习要点（对照 vLLM Scheduler）：
 * 1. 两个队列：waiting（待 prefill 的新请求）与 running（正在 decode 的请求）。
 * 2. 每个调度步：从 waiting 按“显存与并发上限”接纳(admit)新请求做 prefill，
 *    对 running 中请求各走一步 decode，再把完成的请求清扫(sweep)出去——
 *    这就是 continuous batching：请求随时进出，batch 始终尽量满。
 * 3. waiting 用并发安全队列，因为 HTTP 线程随时 add，engine 线程消费。
 * 4. running 仅由 engine 线程读写，不暴露可变引用给外部。
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

    /** 查看待 prefill 队列头部（不取出） */
    public Sequence peekWaiting() {
        return waiting.peek();
    }

    /** 从 waiting 取出队首请求 */
    public Sequence pollWaiting() {
        return waiting.poll();
    }

    /** 将请求加入 running 列表 */
    public void addRunning(Sequence seq) {
        running.add(seq);
    }

    /** 当前 running 列表的不可变视图（仅用于遍历，不可修改） */
    public List<Sequence> runningView() {
        return List.copyOf(running);
    }

    /** 按条件移除 running 中的请求（engine 线程调用） */
    public void removeRunningIf(Predicate<Sequence> condition) {
        running.removeIf(condition);
    }

    /** 当前正在运行的请求数 */
    public int runningCount() {
        return running.size();
    }

    /** 待 prefill 队列是否为空 */
    public boolean waitingIsEmpty() {
        return waiting.isEmpty();
    }

    /** 待 prefill 队列中的请求数 */
    public int waitingCount() {
        return waiting.size();
    }

    public int maxNumSeqs() {
        return maxNumSeqs;
    }

    /** 是否还有工作要做 */
    public boolean hasWork() {
        return !waiting.isEmpty() || !running.isEmpty();
    }
}
