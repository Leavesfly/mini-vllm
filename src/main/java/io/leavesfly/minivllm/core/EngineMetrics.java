package io.leavesfly.minivllm.core;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * EngineMetrics —— 引擎级运行指标（对照 vLLM 的 Prometheus 指标集，学习版精简）。
 *
 * 学习要点：
 * 1. TTFT（Time To First Token）：请求加入到首 token 产出的延迟，反映 prefill/排队体验。
 * 2. ITL（Inter-Token Latency）：相邻两个生成 token 的间隔，反映 decode 流畅度——
 *    Chunked Prefill 优化是否生效就体现在这里（长 prompt 不再造成 ITL 大尖刺）。
 * 3. 吞吐（tok/s）：累计生成 token / 引擎运行时长，宏观容量指标。
 * 4. 队列长度与 KV 利用率在查询时从 Scheduler/KVCacheManager 现读（不在此打点），
 *    因为它们随每步调度变化，快照式读取比累计更真实。
 * 5. 全部用 LongAdder/AtomicLong 无锁累加：引擎线程写、HTTP 线程读，
 *    读取允许轻微滞后（监控语义可接受）。
 */
public final class EngineMetrics {

    private final long startNanos = System.nanoTime();

    private final LongAdder generatedTokens = new LongAdder();
    private final LongAdder finishedRequests = new LongAdder();
    private final LongAdder preemptions = new LongAdder();

    // 投机采样：草稿 token 总数 / 被接受总数（接受率反映 prompt-lookup 命中质量）
    private final LongAdder specDraftTokens = new LongAdder();
    private final LongAdder specAcceptedTokens = new LongAdder();

    // TTFT：累计纳秒 / 最大值 / 样本数
    private final LongAdder ttftSumNanos = new LongAdder();
    private final AtomicLong ttftMaxNanos = new AtomicLong();
    private final LongAdder ttftCount = new LongAdder();

    // ITL：累计纳秒 / 最大值 / 样本数
    private final LongAdder itlSumNanos = new LongAdder();
    private final AtomicLong itlMaxNanos = new AtomicLong();
    private final LongAdder itlCount = new LongAdder();

    /** 首 token 产出（每请求一次）：latencyNanos = 产出时刻 - 请求加入时刻 */
    public void recordFirstToken(long latencyNanos) {
        ttftSumNanos.add(latencyNanos);
        ttftCount.increment();
        ttftMaxNanos.accumulateAndGet(latencyNanos, Math::max);
    }

    /** 后续 token 产出：gapNanos = 本 token 时刻 - 上一 token 时刻 */
    public void recordInterToken(long gapNanos) {
        itlSumNanos.add(gapNanos);
        itlCount.increment();
        itlMaxNanos.accumulateAndGet(gapNanos, Math::max);
    }

    /** 每个成功生成的 token 计数（吞吐分子） */
    public void recordGeneratedToken() {
        generatedTokens.increment();
    }

    /** 请求结束（FINISHED，不含 ABORT） */
    public void recordFinishedRequest() {
        finishedRequests.increment();
    }

    /** 一次抢占发生 */
    public void recordPreemption() {
        preemptions.increment();
    }

    /** 一次投机验证完成：drafted 个草稿 token 中接受了 accepted 个 */
    public void recordSpeculative(int drafted, int accepted) {
        specDraftTokens.add(drafted);
        specAcceptedTokens.add(accepted);
    }

    public long generatedTokens() {
        return generatedTokens.sum();
    }

    /** 引擎运行时长（秒），吞吐的分母 */
    public double uptimeSeconds() {
        return (System.nanoTime() - startNanos) / 1e9;
    }

    /**
     * 指标快照（有序 Map，供 SimpleJson 序列化到 /metrics 端点）。
     *
     * @param waiting   当前 waiting 队列长度（Scheduler 现读）
     * @param running   当前 running 列表长度（Scheduler 现读）
     * @param kvUsage   KV block 占用率 0~1（KVCacheManager 现读）
     */
    public Map<String, Object> snapshot(int waiting, int running, double kvUsage) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("waiting", waiting);
        m.put("running", running);
        m.put("kv_cache_usage", round(kvUsage, 4));
        m.put("generated_tokens", generatedTokens.sum());
        m.put("finished_requests", finishedRequests.sum());
        m.put("preemptions", preemptions.sum());
        long sd = specDraftTokens.sum();
        m.put("spec_draft_tokens", sd);
        m.put("spec_accepted_tokens", specAcceptedTokens.sum());
        m.put("spec_acceptance", sd > 0 ? round((double) specAcceptedTokens.sum() / sd, 3) : 0.0);
        double up = uptimeSeconds();
        m.put("throughput_tok_per_sec", up > 0 ? round(generatedTokens.sum() / up, 2) : 0.0);
        long tc = ttftCount.sum();
        m.put("avg_ttft_ms", tc > 0 ? round(ttftSumNanos.sum() / 1e6 / tc, 2) : 0.0);
        m.put("max_ttft_ms", round(ttftMaxNanos.get() / 1e6, 2));
        long ic = itlCount.sum();
        m.put("avg_itl_ms", ic > 0 ? round(itlSumNanos.sum() / 1e6 / ic, 3) : 0.0);
        m.put("max_itl_ms", round(itlMaxNanos.get() / 1e6, 3));
        return m;
    }

    private static double round(double v, int digits) {
        double f = Math.pow(10, digits);
        return Math.round(v * f) / f;
    }
}
