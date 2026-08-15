package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * LLMEngine —— 引擎核心，驱动 Continuous Batching 的 admit → 混合批前向 → sweep 循环。
 *
 * 学习要点（对照 vLLM LLMEngine + Scheduler）：
 * 1. admitNew：从 waiting 取新请求，做前缀共享 + KV 容量分配后转 PREFILL 阶段；
 *    受 maxNumSeqs 与 KV 显存约束；显存不足时回滚已分配 block 并停止接纳。
 * 2. Chunked Prefill：每步按 tokenBudget 组混合批——decode 请求各占 1 token（优先），
 *    剩余预算切给 PREFILL 序列的 prompt chunk；长 prompt 分多步完成，不阻塞 decode。
 * 3. sweepFinished：完成的请求释放 KV cache（按引用计数），移出 running。
 * 4. Preemption（recompute 版，对照 vLLM POLICY_RECOMPUTE）：decode 时 KV 不足不再直接
 *    ABORT，而是抢占 running 中最晚加入的序列——释放其全部 block 退回 waiting 队首，
 *    重新 admit 时以 prompt+已生成 tokens 为上下文重算；被抢占序列已注册的 block 留在
 *    前缀缓存中，重算大概率命中共享；preemptCount + 队首插入防饿死。
 * 循环前三步 + 抢占正是 continuous batching 的本质：请求随时进出，算力始终尽量满载。
 *
 * 可替换策略（扩展点，均经构造器注入）：
 *   - {@link SchedulingPolicy}：admit 顺序（默认 FIFO）
 *   - {@link SamplingStrategy}：logits → token（默认 temperature/top-k/top-p）
 *   - {@link StopCriteria}：停止判断（默认 EOS + maxTokens）
 *
 * 两种驱动模式：
 *   - start()：独立线程持续 step（服务模式，配合 HTTP API）。
 *   - generate()：同步驱动 step 直到单请求完成（测试 / 单请求）。
 */
public final class LLMEngine {

    private final LlmModel model;
    private final KVCacheManager kvMgr;
    private final SimpleTokenizer tokenizer;
    private final SamplingStrategy sampling;
    private final StopCriteria stopCriteria;
    private final SchedulingPolicy schedulingPolicy;
    private final Scheduler scheduler;
    private final ModelConfig cfg;
    private final int nLayer;
    private final int[] eosTokens;

    private final AtomicInteger nextId = new AtomicInteger(0);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile boolean verbose = false;
    /**
     * 投机采样的草稿长度（0=关闭，默认）。>0 时单序列 decode 走 prompt-lookup 投机：
     * 从上下文 n-gram 查草稿，一次前向验证并按最长一致前缀接受——greedy 下输出
     * 与普通 decode 严格一致，是零成本的无损加速（对照 vLLM prompt_lookup 模式）。
     */
    private volatile int speculativeK = 0;
    /**
     * 草稿模型投机（可选，对照 vLLM draft-model 模式）：设置后单序列 greedy decode
     * 由小模型起草 k 个 token、目标模型一次前向验证；未设置时回退 prompt-lookup 自投机。
     */
    private volatile DraftProposer draftProposer;

    /** 引擎运行指标（TTFT/ITL/吞吐等，/metrics 端点读取） */
    private final EngineMetrics metrics = new EngineMetrics();
    /** 空闲唤醒信号：addRequest 时 release，引擎线程无工作时 acquire 阻塞（替代 sleep 轮询） */
    private final Semaphore wakeup = new Semaphore(0);

    /** 默认策略组合：FIFO 调度 + temperature/top-k/top-p 采样 + EOS/maxTokens 停止 */
    public LLMEngine(LlmModel model, KVCacheManager kvMgr, SimpleTokenizer tokenizer,
                     int maxNumSeqs, int[] eosTokens, long seed) {
        this(model, kvMgr, tokenizer, maxNumSeqs, eosTokens,
                new DefaultSamplingStrategy(seed), new EosMaxTokensCriteria(), new FifoPolicy());
    }

    /** 完整构造：自定义采样 / 停止 / 调度策略 */
    public LLMEngine(LlmModel model, KVCacheManager kvMgr, SimpleTokenizer tokenizer,
                     int maxNumSeqs, int[] eosTokens, SamplingStrategy sampling,
                     StopCriteria stopCriteria, SchedulingPolicy schedulingPolicy) {
        this(model, kvMgr, tokenizer, maxNumSeqs, Scheduler.DEFAULT_TOKEN_BUDGET, eosTokens,
                sampling, stopCriteria, schedulingPolicy);
    }

    /** 完整构造 + 自定义每步 token 预算（Chunked Prefill 粒度） */
    public LLMEngine(LlmModel model, KVCacheManager kvMgr, SimpleTokenizer tokenizer,
                     int maxNumSeqs, int tokenBudget, int[] eosTokens, SamplingStrategy sampling,
                     StopCriteria stopCriteria, SchedulingPolicy schedulingPolicy) {
        this.model = model;
        this.kvMgr = kvMgr;
        this.tokenizer = tokenizer;
        this.cfg = model.config();
        this.nLayer = cfg.nLayer();
        this.eosTokens = eosTokens;
        this.sampling = sampling;
        this.stopCriteria = stopCriteria;
        this.schedulingPolicy = schedulingPolicy;
        this.scheduler = new Scheduler(maxNumSeqs, tokenBudget);
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    /** 设置投机采样草稿长度（0 关闭；仅 greedy 单序列 decode 生效） */
    public void setSpeculativeK(int k) {
        this.speculativeK = Math.max(0, k);
    }

    /**
     * 接入草稿模型（需与目标模型同词表；仍需 speculativeK > 0 才会启用）。
     * 草稿模型独占 draftKv 池，起草/追平/截断由 {@link DraftProposer} 管理。
     */
    public void setDraftModel(LlmModel draftModel, KVCacheManager draftKv) {
        if (draftModel.config().vocabSize() != cfg.vocabSize()) {
            throw new IllegalArgumentException("草稿模型词表大小 " + draftModel.config().vocabSize()
                    + " 与目标模型 " + cfg.vocabSize() + " 不一致，无法投机");
        }
        this.draftProposer = new DraftProposer(draftModel, draftKv);
    }

    public ModelConfig config() {
        return cfg;
    }

    public Scheduler scheduler() {
        return scheduler;
    }

    public EngineMetrics metrics() {
        return metrics;
    }

    /**
     * 指标快照（/metrics 端点用）：队列长度与 KV 利用率现读，累计指标取 metrics。
     * KV 占用率 = 1 - 空闲/总量（含缓存态 block：它们随时可驱逐但当前不可发放）。
     */
    public Map<String, Object> metricsSnapshot() {
        int total = kvMgr.pool().numBlocks();
        double usage = total > 0 ? 1.0 - (double) kvMgr.freeBlocks() / total : 0.0;
        return metrics.snapshot(scheduler.waitingCount(), scheduler.runningCount(), usage);
    }

    /** 加入一个异步请求，返回 Sequence 供调用方跟踪状态/流式回调 */
    public Sequence addRequest(String prompt, SamplingParams params, Consumer<String> onToken) {
        int[] promptTokens = tokenizer.encode(prompt);
        Sequence seq = new Sequence(nextId.getAndIncrement(), promptTokens, params,
                eosTokens, nLayer, onToken);
        // 注入增量解码器：BPE 实现缓冲跨 token 的 UTF-8 字节避免乱码，
        // 其余实现逐 token 直接解码（由分词器接口的默认方法提供）
        seq.setIncDecoder(tokenizer.incrementalDecoder());
        seq.setArrivalNanos(System.nanoTime()); // TTFT 起点
        scheduler.add(seq);
        wakeup.release(); // 唤醒可能阻塞在空闲等待的引擎线程
        return seq;
    }

    // ─── 调度循环 ───

    /** 执行一个调度步：清扫取消 → admit（只分配）→ 混合批前向（prefill chunk + decode）→ sweep */
    public void step() {
        sweepCancelledWaiting();
        admitNew();
        // decode 优先：先收集本步要 decode 的序列（各占 1 token 预算）
        List<Sequence> decodeBatch = collectDecodeBatch();
        // 剩余预算切给 prefill chunk（保底 1 token，防止 decode 占满预算时 prefill 饿死）
        int prefillBudget = Math.max(1, scheduler.tokenBudget() - decodeBatch.size());
        runPrefillChunks(prefillBudget);
        runDecodeBatch(decodeBatch);
        sweepFinished();
        if (verbose) {
            System.out.printf("[engine] running=%d waiting=%s freeBlocks=%d%n",
                    scheduler.runningCount(), scheduler.waitingIsEmpty() ? "0" : "pending", kvMgr.freeBlocks());
        }
    }

    /** 清扫已取消的排队请求：尚未分配 KV，直接移除并通知等待方 */
    private void sweepCancelledWaiting() {
        scheduler.removeWaitingIf(seq -> {
            if (!seq.cancelled()) {
                return false;
            }
            seq.setStage(Sequence.Stage.ABORTED);
            seq.markDone();
            return true;
        });
    }

    /**
     * 接纳新请求：前缀共享 + KV 容量分配，转 PREFILL 阶段加入 running。
     * 不在本方法内做前向：prefill 由 {@link #runPrefillChunks} 按 token 预算分块执行。
     * 被抢占序列重新 admit 时上下文 = prompt+已生成 tokens（recompute），
     * 且大概率命中自己留在前缀缓存中的 block，降低重算代价。
     */
    private void admitNew() {
        while (scheduler.runningCount() < scheduler.maxNumSeqs()) {
            Sequence seq = schedulingPolicy.nextToAdmit(scheduler);
            if (seq == null) break;
            int ctxLen = seq.contextLen();
            // 前缀共享：相同 token 前缀的 KV block 直接复用，命中部分无需重算 prefill
            int shared = kvMgr.trySharePrefix(seq.contextTokens(), seq.blockTables());
            // 为每层分配 KV cache block（含共享块在内的总容量）
            boolean ok = true;
            for (BlockTable bt : seq.blockTables()) {
                if (!kvMgr.ensureCapacity(bt, ctxLen)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                // 显存不足：回滚已分配 block（含共享引用）
                for (BlockTable bt : seq.blockTables()) {
                    if (bt.numBlocks() > 0) kvMgr.free(bt);
                }
                if (seq.selfPreempted() || scheduler.runningCount() == 0) {
                    // 自抢占后仍分配不出，或没有任何在跑序列可释放：
                    // 池容不下单个序列（极端配置），等待无意义，ABORT 避免引擎无限空转
                    scheduler.removeWaiting(seq);
                    seq.setStage(Sequence.Stage.ABORTED);
                    seq.markDone();
                    continue;
                }
                break; // 留在 waiting 等下次（等 running 释放）
            }
            scheduler.removeWaiting(seq); // 分配成功，正式从 waiting 移除
            seq.setSelfPreempted(false); // 重置自抢占标记（新失败周期重新计）
            seq.setPrefilledTokens(shared); // 共享命中的 token 视为已完成 prefill
            seq.setStage(Sequence.Stage.PREFILL);
            scheduler.addRunning(seq);
        }
    }

    /**
     * Chunked Prefill：按预算逐块推进 PREFILL 序列的前向。
     * 每块写 KV cache、更新进度；上下文算完的序列转 DECODE：
     * 新请求采样首 token；被抢占重算者已有生成 token，直接续接不重复采样。
     * @return 实际消耗的 prefill token 数
     */
    private int runPrefillChunks(int budget) {
        int used = 0;
        for (Sequence seq : scheduler.runningView()) {
            if (budget <= 0) break;
            if (seq.stage() != Sequence.Stage.PREFILL || seq.cancelled()) continue;
            int ctxLen = seq.contextLen();
            int done = seq.prefilledTokens();
            int chunk = Math.min(ctxLen - done, budget);
            if (chunk <= 0) continue;
            // 增量 prefill：从 done 处算 chunk 个 token（带前缀注意力读已缓存 KV）
            int[] tokens = Arrays.copyOfRange(seq.contextTokens(), done, done + chunk);
            float[] logits = model.prefillLogits(tokens, kvMgr, seq.blockTables(), done);
            seq.setPrefilledTokens(done + chunk);
            budget -= chunk;
            used += chunk;
            if (seq.prefilledTokens() >= ctxLen) {
                // 上下文全部算完：注册前缀供后续共享（含被抢占者已生成的部分）
                kvMgr.registerPrefix(seq.contextTokens(), seq.blockTables());
                if (seq.outputTokens().isEmpty()) {
                    // 新请求：采样首 token；重算者已有输出 token，直接进入 decode 续接
                    int nextToken = sampling.sample(logits, seq.params(),
                            seq.promptTokens(), seq.outputTokens());
                    seq.outputTokens().add(nextToken);
                    recordTokenEmitted(seq);
                    emitToken(seq, nextToken);
                }
                seq.setStage(Sequence.Stage.DECODE);
            }
        }
        return used;
    }

    /**
     * 收集本步可 decode 的序列：显存不足时先抢占其它序列腾出 block（recompute 版，
     * 对照 vLLM Scheduler._preempt）；抢占完所有候选仍不足才 ABORT。
     */
    private List<Sequence> collectDecodeBatch() {
        List<Sequence> batch = new ArrayList<>();
        for (Sequence seq : scheduler.runningView()) {
            if (seq.stage() != Sequence.Stage.DECODE || seq.cancelled()) continue;
            int prevBlocks = totalBlocks(seq);
            while (!ensureDecodeCapacity(seq)) {
                if (totalBlocks(seq) > prevBlocks) {
                    prevBlocks = totalBlocks(seq); // 有进展（部分层已分配），重试剩余层
                    continue;
                }
                Sequence victim = choosePreemptVictim(seq);
                if (victim == null) {
                    seq.setStage(Sequence.Stage.ABORTED); // 池容不下单个序列的极端场景
                    break;
                }
                if (victim == seq) {
                    seq.setSelfPreempted(true); // 自抢占：重试仍失败时 admit 会 ABORT 兜底
                }
                preempt(victim);
                batch.remove(victim); // victim 可能已加入本步 batch（队序更靠前），block 已释放不可再前向
                if (victim == seq) break; // 抢占了自己：本步回队等待，下步重新 admit
            }
            if (seq.stage() == Sequence.Stage.DECODE) {
                batch.add(seq);
            }
        }
        return batch;
    }

    /** decode 新 token 的 KV 需要落位：各层容量至少 totalLen */
    private boolean ensureDecodeCapacity(Sequence seq) {
        int need = seq.totalLen();
        for (BlockTable bt : seq.blockTables()) {
            if (!kvMgr.ensureCapacity(bt, need)) {
                return false;
            }
        }
        return true;
    }

    private int totalBlocks(Sequence seq) {
        int n = 0;
        for (BlockTable bt : seq.blockTables()) {
            n += bt.numBlocks();
        }
        return n;
    }

    /**
     * 选抢占牺牲者：running 中最晚加入的活跃序列（对照 vLLM 抢占队尾）；
     * 优先抢别人，实在没有其它候选才抢请求者自己。
     */
    private Sequence choosePreemptVictim(Sequence requester) {
        List<Sequence> snapshot = scheduler.runningView();
        Sequence self = null;
        for (int i = snapshot.size() - 1; i >= 0; i--) {
            Sequence s = snapshot.get(i);
            if (s.cancelled()
                    || (s.stage() != Sequence.Stage.DECODE && s.stage() != Sequence.Stage.PREFILL)) {
                continue;
            }
            if (s == requester) {
                self = s;
            } else {
                return s;
            }
        }
        return self;
    }

    /**
     * 抢占（recompute 版）：释放牺牲者的全部 KV block，进度归零，退回 waiting 队首。
     * 已注册前缀缓存的 block 转入缓存态（不丢内容），重新 admit 时大概率命中共享；
     * 队首插入 + preemptCount 提升优先级，防止反复被抢占而饿死。
     */
    private void preempt(Sequence victim) {
        for (BlockTable bt : victim.blockTables()) {
            kvMgr.free(bt);
        }
        victim.markPreempted();
        victim.setStage(Sequence.Stage.PREEMPTED);
        metrics.recordPreemption();
        scheduler.removeRunningIf(s -> s == victim);
        scheduler.addFirstWaiting(victim);
        if (verbose) {
            System.out.printf("[engine] preempt seq#%d（第 %d 次，已生成 %d tokens，退回队首重算）%n",
                    victim.id(), victim.preemptCount(), victim.outputTokens().size());
        }
    }

    /** 批量 decode 前向 + 逐序列采样输出（跨序列批处理成一次前向） */
    private void runDecodeBatch(List<Sequence> batch) {
        if (batch.isEmpty()) return;

        // 投机路径：单序列 + greedy + 上下文有 n-gram 草稿时，一次前向验证多个 token
        if (batch.size() == 1 && speculativeK > 0 && runSpeculativeStep(batch.get(0))) {
            return;
        }

        // 1. 组装批量输入：把 B 个序列堆成一次前向（权重只读一次、跨 B 复用）
        int b = batch.size();
        int[] lastTokens = new int[b];
        int[] curIdxs = new int[b];
        BlockTable[][] bts = new BlockTable[b][];
        for (int i = 0; i < b; i++) {
            Sequence seq = batch.get(i);
            lastTokens[i] = seq.outputTokens().get(seq.outputTokens().size() - 1);
            curIdxs[i] = seq.totalLen() - 1;
            bts[i] = seq.blockTables();
        }

        // 2. 批量前向 -> 每个序列的 logits
        float[][] logits = model.decodeLogitsBatch(lastTokens, curIdxs, kvMgr, bts);

        // 3. 逐序列采样并输出
        for (int i = 0; i < b; i++) {
            Sequence seq = batch.get(i);
            int nextToken = sampling.sample(logits[i], seq.params(),
                    seq.promptTokens(), seq.outputTokens());
            seq.outputTokens().add(nextToken);
            recordTokenEmitted(seq);
            emitToken(seq, nextToken);
        }
    }

    /**
     * 投机采样步（对照 vLLM speculative decoding）。
     *
     * 草稿二选一：已接入草稿模型（draft-model 模式）则由 {@link DraftProposer} 自回归
     * 起草；否则从上下文 n-gram 查草稿（prompt_lookup 模式）。后续验证路径一致：
     * 一次前向验证 [lastToken]+草稿 共 k+1 个位置 → 按最长一致前缀接受
     * → 从未一致位置采样新 token → 截断被拒绝草稿的 KV（目标侧与草稿侧同步回滚）。
     * 仅 greedy（temperature≈0）启用：argmax 比对保证输出与普通 decode 逐 token 一致；
     * 非 greedy 的拒绝采样（rsample from norm(max(0, p−q))）留作后续扩展。
     *
     * @return true 表示本步已按投机路径完成（false 时调用方走普通 decode）
     */
    private boolean runSpeculativeStep(Sequence seq) {
        if (seq.params().temperature() >= 1e-5f) {
            return false; // 非 greedy：本步回退普通 decode
        }
        // 草稿来源：草稿模型优先，未配置时回退上下文 n-gram（prompt-lookup）
        DraftProposer proposer = draftProposer;
        int[] draft = proposer != null
                ? proposer.propose(seq, speculativeK)
                : PromptLookup.findDraft(seq.promptTokens(), seq.outputTokens(), speculativeK);
        if (draft == null || draft.length == 0) {
            return false;
        }
        int len = seq.totalLen(); // prompt + 已生成 的当前总长
        // 验证前向覆盖位置 [len-1, len+k-1]（重写末尾 token 的 KV 是幂等覆写）
        int need = len + draft.length;
        for (BlockTable bt : seq.blockTables()) {
            if (!kvMgr.ensureCapacity(bt, need)) {
                return false; // 池容量不足：本步回退普通 decode（collectDecodeBatch 已保证 len 容量）
            }
        }
        int[] tokens = new int[draft.length + 1];
        tokens[0] = seq.outputTokens().get(seq.outputTokens().size() - 1);
        System.arraycopy(draft, 0, tokens, 1, draft.length);
        float[][] logits = model.prefillLogitsAll(tokens, kvMgr, seq.blockTables(), len - 1);

        // 最长一致前缀接受；被接受的 EOS 草稿使生成即刻收尾（不再多看后续位置）
        int accepted = 0;
        boolean hitEos = false;
        for (int j = 0; j < draft.length; j++) {
            if (argmax(logits[j]) != draft[j]) {
                break;
            }
            accepted++;
            if (stopCriteria.isStopToken(seq, draft[j])) {
                hitEos = true;
                break;
            }
        }
        // 新 token 来自首个未一致位置的分布（全接受时为草稿末尾之后的位置）
        int newToken = hitEos ? -1
                : sampling.sample(logits[accepted], seq.params(), seq.promptTokens(), seq.outputTokens());
        int produced = accepted + (hitEos ? 0 : 1);
        // maxTokens 配额裁剪：超出部分不进 outputTokens，KV 同步截断
        int budget = seq.params().maxTokens() - seq.outputTokens().size();
        produced = Math.min(produced, Math.max(0, budget));
        for (BlockTable bt : seq.blockTables()) {
            kvMgr.truncateTo(bt, len + produced);
        }
        if (proposer != null) {
            proposer.sync(seq, len + produced); // 草稿侧 KV 同步截断（被拒绝草稿的块释放）
        }
        metrics.recordSpeculative(draft.length, accepted);
        // 追加被接受的草稿（budget 截断时最多 produced 个）；旧写法 produced-1 在截断时
        // 一个不追加，outputTokens 永远到不了 maxTokens —— 测试曾因此活锁
        int draftsKept = Math.min(accepted, produced);
        for (int t = 0; t < draftsKept; t++) {
            seq.outputTokens().add(draft[t]);
            recordTokenEmitted(seq);
            emitToken(seq, draft[t]);
        }
        if (!hitEos && produced > accepted) {
            seq.outputTokens().add(newToken);
            recordTokenEmitted(seq);
            emitToken(seq, newToken);
        }
        return true;
    }

    /** logits 的 argmax（greedy 接受判定） */
    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * 生成一个 token 后的指标打点：首 token 记 TTFT，后续记 ITL；
     * 被抢占后重算的序列不重复记 TTFT（outputTokens 仍保留历史，首 token 判定天然正确）。
     */
    private void recordTokenEmitted(Sequence seq) {
        long now = System.nanoTime();
        if (seq.outputTokens().size() == 1) {
            metrics.recordFirstToken(now - seq.arrivalNanos());
        } else {
            metrics.recordInterToken(now - seq.lastTokenNanos());
        }
        seq.setLastTokenNanos(now);
        metrics.recordGeneratedToken();
    }

    /** 清扫完成/中止的请求，释放 KV cache（含草稿模型侧的 KV 状态） */
    private void sweepFinished() {
        scheduler.removeRunningIf(seq -> {
            if (seq.cancelled()) {
                // 客户端断连/显式取消：释放 KV 后 ABORT，等待方（HTTP 线程）立即返回
                freeSequenceKv(seq);
                seq.setStage(Sequence.Stage.ABORTED);
                seq.markDone();
                return true;
            }
            if (stopCriteria.shouldStop(seq)) {
                // 冲刷增量解码器中剩余的字节（不完整的 UTF-8 尾部）
                if (seq.incDecoder() != null && seq.onToken() != null) {
                    String rest = seq.incDecoder().flush();
                    if (!rest.isEmpty()) {
                        seq.onToken().accept(rest);
                    }
                }
                freeSequenceKv(seq);
                seq.setStage(Sequence.Stage.FINISHED);
                metrics.recordFinishedRequest();
                seq.markDone(); // 触发等待方的 awaitDone() 返回
                return true;
            }
            return false;
        });
    }

    /** 释放序列的全部 KV：目标侧各层 block + 草稿模型侧状态（若启用投机） */
    private void freeSequenceKv(Sequence seq) {
        for (BlockTable bt : seq.blockTables()) {
            kvMgr.free(bt);
        }
        DraftProposer proposer = draftProposer;
        if (proposer != null) {
            proposer.free(seq);
        }
    }

    private void emitToken(Sequence seq, int token) {
        if (seq.onToken() == null) {
            return;
        }
        // 停止 token 不输出文本（否则 <|im_end|> 等会泄露到响应里，
        // 并在多轮对话中污染 ChatML 上下文）
        if (stopCriteria.isStopToken(seq, token)) {
            return;
        }
        if (seq.incDecoder() != null) {
            String piece = seq.incDecoder().accept(token);
            if (!piece.isEmpty()) {
                seq.onToken().accept(piece);
            }
        } else {
            seq.onToken().accept(tokenizer.decode(new int[]{token}));
        }
    }

    // ─── 驱动模式 ───

    /**
     * 服务模式：独立线程持续 step。
     * 空闲时用信号量阻塞等待（而非 sleep 轮询）：addRequest 时 release 即刻唤醒，
     * 零请求时线程完全挂起不占 CPU。
     */
    public void start() {
        if (running.getAndSet(true)) return;
        Thread t = new Thread(() -> {
            while (running.get()) {
                if (scheduler.hasWork()) {
                    step();
                } else {
                    try {
                        wakeup.acquire();
                    } catch (InterruptedException e) {
                        break;
                    }
                }
            }
        }, "mini-vllm-engine");
        t.setDaemon(true);
        t.start();
    }

    public void stop() {
        running.set(false);
    }

    /**
     * 同步生成：加入请求后自己驱动 step 直到完成，返回完整文本。
     * 适合测试与单请求调用；多请求并发请用 start() + addRequest()。
     */
    public String generate(String prompt, SamplingParams params) {
        List<String> collected = Collections.synchronizedList(new ArrayList<>());
        Sequence seq = addRequest(prompt, params, collected::add);
        while (!seq.isFinished()) {
            if (scheduler.hasWork()) {
                step();
            }
        }
        return String.join("", collected);
    }

    /** 默认参数同步生成 */
    public String generate(String prompt, int maxTokens) {
        return generate(prompt, SamplingParams.DEFAULT.withMaxTokens(maxTokens));
    }
}
