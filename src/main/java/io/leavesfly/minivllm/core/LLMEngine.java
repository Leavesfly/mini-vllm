package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.Transformer;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.math.Sampler;

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
 * 两种驱动模式：
 *   - start()：独立线程持续 step（服务模式，配合 HTTP API）。
 *   - generate()：同步驱动 step 直到单请求完成（测试 / 单请求）。
 */
public final class LLMEngine {

    private final Transformer model;
    private final KVCacheManager kvMgr;
    private final SimpleTokenizer tokenizer;
    private final Sampler sampler;
    private final Scheduler scheduler;
    private final ModelConfig cfg;
    private final int nLayer;
    private final int eosToken;

    private final AtomicInteger nextId = new AtomicInteger(0);
    private final AtomicBoolean running = new AtomicBoolean(false);
    private volatile boolean verbose = false;

    public LLMEngine(Transformer model, KVCacheManager kvMgr, SimpleTokenizer tokenizer,
                     int maxNumSeqs, int eosToken, long seed) {
        this.model = model;
        this.kvMgr = kvMgr;
        this.tokenizer = tokenizer;
        this.cfg = model.config();
        this.nLayer = cfg.nLayer;
        this.eosToken = eosToken;
        this.sampler = new Sampler(seed);
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
    public Sequence addRequest(String prompt, int maxTokens,
                               float temperature, int topK, float topP,
                               Consumer<String> onToken) {
        int[] promptTokens = tokenizer.encode(prompt);
        Sequence seq = new Sequence(nextId.getAndIncrement(), promptTokens, maxTokens,
                temperature, topK, topP, eosToken, nLayer, onToken);
        scheduler.add(seq);
        return seq;
    }

    // ===================== 调度循环 =====================

    /** 执行一个调度步骤：admit → decode → sweep */
    public void step() {
        admitNew();
        decodeStep();
        sweepFinished();
        if (verbose) {
            System.out.printf("[engine] running=%d waiting=%d freeBlocks=%d%n",
                    scheduler.running().size(), scheduler.waiting().size(), kvMgr.pool.freeBlocks());
        }
    }

    /** 接纳新请求：prefill 并加入 running */
    private void admitNew() {
        while (scheduler.running().size() < scheduler.maxNumSeqs()) {
            Sequence seq = scheduler.waiting().peek();
            if (seq == null) break;
            int promptLen = seq.promptTokens.length;
            // 为每层分配 KV cache block
            boolean ok = true;
            for (BlockTable bt : seq.blockTables) {
                if (!kvMgr.ensureCapacity(bt, promptLen)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                // 显存不足：回滚已分配 block，留在 waiting 等下次（等 running 释放）
                for (BlockTable bt : seq.blockTables) {
                    if (bt.numBlocks() > 0) kvMgr.free(bt);
                }
                break;
            }
            scheduler.waiting().poll(); // 正式取出
            seq.stage = Sequence.Stage.PREFILL;
            // prefill：一次性处理整段 prompt，写入 KV cache，生成第一个 token
            float[] hidden = model.prefill(seq.promptTokens, kvMgr, seq.blockTables, 0);
            float[] last = model.lastRow(hidden, promptLen);
            float[] logits = model.logits(last);
            configureSampler(seq);
            int nextToken = sampler.sample(logits);
            seq.outputTokens.add(nextToken);
            seq.stage = Sequence.Stage.DECODE;
            scheduler.running().add(seq);
            emitToken(seq, nextToken);
        }
    }

    /** 对每个 running 请求各走一步 decode */
    private void decodeStep() {
        for (Sequence seq : scheduler.running()) {
            if (seq.stage != Sequence.Stage.DECODE) continue;
            int lastToken = seq.outputTokens.get(seq.outputTokens.size() - 1);
            int curIdx = seq.totalLen() - 1; // 当前 token 的全局位置
            int need = curIdx + 1;
            // 按需扩容 KV cache（新 token 的 K/V 需要落位）
            boolean ok = true;
            for (BlockTable bt : seq.blockTables) {
                if (!kvMgr.ensureCapacity(bt, need)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                seq.stage = Sequence.Stage.ABORTED; // 显存不足，中止（学习版不做 preemption）
                continue;
            }
            float[] hidden = model.decode(lastToken, curIdx, kvMgr, seq.blockTables);
            float[] logits = model.logits(hidden);
            configureSampler(seq);
            int nextToken = sampler.sample(logits);
            seq.outputTokens.add(nextToken);
            emitToken(seq, nextToken);
        }
    }

    /** 清扫完成/中止的请求，释放 KV cache */
    private void sweepFinished() {
        scheduler.running().removeIf(seq -> {
            if (seq.isFinished()) {
                for (BlockTable bt : seq.blockTables) {
                    kvMgr.free(bt);
                }
                seq.stage = Sequence.Stage.FINISHED;
                return true;
            }
            return false;
        });
    }

    private void configureSampler(Sequence seq) {
        sampler.temperature = seq.temperature;
        sampler.topK = seq.topK;
        sampler.topP = seq.topP;
    }

    private void emitToken(Sequence seq, int token) {
        if (seq.onToken != null) {
            seq.onToken.accept(tokenizer.decode(new int[]{token}));
        }
    }

    // ===================== 驱动模式 =====================

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
    public String generate(String prompt, int maxTokens,
                           float temperature, int topK, float topP) {
        List<String> collected = Collections.synchronizedList(new ArrayList<>());
        Sequence seq = addRequest(prompt, maxTokens, temperature, topK, topP, collected::add);
        while (!seq.isFinished()) {
            if (scheduler.hasWork()) {
                step();
            }
        }
        return String.join("", collected);
    }

    /** 默认参数同步生成 */
    public String generate(String prompt, int maxTokens) {
        return generate(prompt, maxTokens, 0.8f, 0, 0.9f);
    }
}
