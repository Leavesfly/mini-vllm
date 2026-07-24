package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * LLMEngine —— 引擎核心，驱动 Continuous Batching 的 admit → decode → sweep 循环。
 *
 * 学习要点（对照 vLLM LLMEngine + Scheduler）：
 * 1. admitNew：从 waiting 取新请求做 prefill。受 maxNumSeqs 与 KV 显存约束；
 *    显存不足时回滚已分配 block 并停止接纳，等 running 中请求释放后再试。
 * 2. decodeStep：对每个 running 请求各走一步 decode（生成 1 个 token），通过 PagedAttention 复用 KV cache。
 * 3. sweepFinished：完成的请求释放 KV cache（按引用计数），移出 running——腾出的显存立刻供 waiting 请求使用。
 * 这三步循环正是 continuous batching 的本质：请求随时进出，GPU（这里用 CPU 模拟）始终尽量满载。
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
        this.model = model;
        this.kvMgr = kvMgr;
        this.tokenizer = tokenizer;
        this.cfg = model.config();
        this.nLayer = cfg.nLayer();
        this.eosTokens = eosTokens;
        this.sampling = sampling;
        this.stopCriteria = stopCriteria;
        this.schedulingPolicy = schedulingPolicy;
        this.scheduler = new Scheduler(maxNumSeqs);
    }

    public void setVerbose(boolean v) {
        this.verbose = v;
    }

    public ModelConfig config() {
        return cfg;
    }

    public Scheduler scheduler() {
        return scheduler;
    }

    /** 加入一个异步请求，返回 Sequence 供调用方跟踪状态/流式回调 */
    public Sequence addRequest(String prompt, SamplingParams params, Consumer<String> onToken) {
        int[] promptTokens = tokenizer.encode(prompt);
        Sequence seq = new Sequence(nextId.getAndIncrement(), promptTokens, params,
                eosTokens, nLayer, onToken);
        // 注入增量解码器：BPE 实现缓冲跨 token 的 UTF-8 字节避免乱码，
        // 其余实现逐 token 直接解码（由分词器接口的默认方法提供）
        seq.setIncDecoder(tokenizer.incrementalDecoder());
        scheduler.add(seq);
        return seq;
    }

    // ─── 调度循环 ───

    /** 执行一个调度步骤：admit → decode → sweep */
    public void step() {
        admitNew();
        decodeStep();
        sweepFinished();
        if (verbose) {
            System.out.printf("[engine] running=%d waiting=%s freeBlocks=%d%n",
                    scheduler.runningCount(), scheduler.waitingIsEmpty() ? "0" : "pending", kvMgr.freeBlocks());
        }
    }

    /** 接纳新请求：prefill 并加入 running */
    private void admitNew() {
        while (scheduler.runningCount() < scheduler.maxNumSeqs()) {
            Sequence seq = schedulingPolicy.nextToAdmit(scheduler);
            if (seq == null) break;
            int promptLen = seq.promptTokens().length;
            // 为每层分配 KV cache block
            boolean ok = true;
            for (BlockTable bt : seq.blockTables()) {
                if (!kvMgr.ensureCapacity(bt, promptLen)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                // 显存不足：回滚已分配 block，留在 waiting 等下次（等 running 释放）
                for (BlockTable bt : seq.blockTables()) {
                    if (bt.numBlocks() > 0) kvMgr.free(bt);
                }
                break;
            }
            scheduler.removeWaiting(seq); // 分配成功，正式从 waiting 移除
            seq.setStage(Sequence.Stage.PREFILL);
            // prefill：一次性处理整段 prompt，写入 KV cache，生成第一个 token
            float[] logits = model.prefillLogits(seq.promptTokens(), kvMgr, seq.blockTables(), 0);
            int nextToken = sampling.sample(logits, seq.params());
            seq.outputTokens().add(nextToken);
            seq.setStage(Sequence.Stage.DECODE);
            scheduler.addRunning(seq);
            emitToken(seq, nextToken);
        }
    }

    /** 对所有 running 请求做一步 decode（跨序列批处理成一次前向） */
    private void decodeStep() {
        // 1. 收集处于 DECODE 且能扩容成功的序列（显存不足者置 ABORTED 并排除）
        List<Sequence> batch = new ArrayList<>();
        for (Sequence seq : scheduler.runningView()) {
            if (seq.stage() != Sequence.Stage.DECODE) continue;
            int need = seq.totalLen(); // 新 token 的 K/V 需要落位，容量至少 curIdx+1
            boolean ok = true;
            for (BlockTable bt : seq.blockTables()) {
                if (!kvMgr.ensureCapacity(bt, need)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                seq.setStage(Sequence.Stage.ABORTED); // 显存不足，中止（学习版不做 preemption）
                continue;
            }
            batch.add(seq);
        }
        if (batch.isEmpty()) return;

        // 2. 组装批量输入：把 B 个序列堆成一次前向（权重只读一次、跨 B 复用）
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

        // 3. 批量前向 -> 每个序列的 logits
        float[][] logits = model.decodeLogitsBatch(lastTokens, curIdxs, kvMgr, bts);

        // 4. 逐序列采样并输出
        for (int i = 0; i < b; i++) {
            Sequence seq = batch.get(i);
            int nextToken = sampling.sample(logits[i], seq.params());
            seq.outputTokens().add(nextToken);
            emitToken(seq, nextToken);
        }
    }

    /** 清扫完成/中止的请求，释放 KV cache */
    private void sweepFinished() {
        scheduler.removeRunningIf(seq -> {
            if (stopCriteria.shouldStop(seq)) {
                // 冲刷增量解码器中剩余的字节（不完整的 UTF-8 尾部）
                if (seq.incDecoder() != null && seq.onToken() != null) {
                    String rest = seq.incDecoder().flush();
                    if (!rest.isEmpty()) {
                        seq.onToken().accept(rest);
                    }
                }
                for (BlockTable bt : seq.blockTables()) {
                    kvMgr.free(bt);
                }
                seq.setStage(Sequence.Stage.FINISHED);
                seq.markDone(); // 触发等待方的 awaitDone() 返回
                return true;
            }
            return false;
        });
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

    /** 服务模式：独立线程持续 step */
    public void start() {
        if (running.getAndSet(true)) return;
        Thread t = new Thread(() -> {
            while (running.get()) {
                if (scheduler.hasWork()) {
                    step();
                } else {
                    try {
                        Thread.sleep(1);
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
