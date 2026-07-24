package io.leavesfly.minivllm.core;

/**
 * FifoPolicy —— 先来先服务的默认调度策略（对照 vLLM 默认 FCFS）。
 *
 * 学习要点：
 * 直接取 waiting 队首即 FIFO。策略只做「选择」，不做「移除」——
 * 移除由引擎在 KV cache 分配成功后执行，保证显存不足时请求仍留在队列。
 */
public final class FifoPolicy implements SchedulingPolicy {

    @Override
    public Sequence nextToAdmit(Scheduler scheduler) {
        return scheduler.peekWaiting();
    }
}
