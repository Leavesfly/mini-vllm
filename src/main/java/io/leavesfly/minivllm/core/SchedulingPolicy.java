package io.leavesfly.minivllm.core;

/**
 * SchedulingPolicy —— 决定 admit 顺序的调度策略。
 *
 * 学习要点：
 * 1. 引擎的 admit 阶段每次向策略要一个候选序列，分配 KV cache 成功后才正式
 *    从 waiting 队列移除；显存不足时候选留在队列，等 running 释放后再试。
 * 2. 默认 {@link FifoPolicy} 先来先服务（与 vLLM 默认一致）。
 * 3. 后续扩展点（本版未实现）：
 *    - 优先级调度：按请求优先级/预估长度重排 waiting；
 *    - preemption：显存吃紧时换出低优先级 running 序列（vLLM 的 swap/recompute），
 *      而不是像现在这样直接 ABORTED。
 */
public interface SchedulingPolicy {

    /**
     * 选出下一个待 admit 的候选序列（不移除；由引擎在分配成功后移除）。
     *
     * @return 候选序列；waiting 为空时返回 null
     */
    Sequence nextToAdmit(Scheduler scheduler);
}
