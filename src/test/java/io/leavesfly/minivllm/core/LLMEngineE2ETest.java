package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.TransformerModel;
import io.leavesfly.minivllm.tokenizer.ByteTokenizer;
import io.leavesfly.minivllm.weights.ModelLoader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * LLMEngine 端到端测试 —— 从 prompt 经 tokenizer → prefill/decode（PagedAttention + Continuous
 * Batching）→ 采样 → 生成结果，打通整条推理链路。
 *
 * 说明：随机初始化模型（seed 固定）输出无语义，但过程完全确定，可用于验证：
 * 1. 生成长度受 maxTokens 约束；
 * 2. greedy（temperature≈0）解码在相同权重下可复现；
 * 3. EOS 命中即提前停止；
 * 4. Continuous Batching：多请求随时进出、全部正常完成；
 * 5. 流式回调与生成 token 一一对应；
 * 6. 采样输出恒在合法词表范围内；
 * 7. GPT-3 交替稀疏注意力模型的 decode 路径端到端可跑通。
 */
class LLMEngineE2ETest {

    private static final long SEED = 12345L;

    /** 构造一个可跑通的引擎（随机初始化模型 + 字节级分词器 + PagedAttention 内存池） */
    private static LLMEngine newEngine(ModelConfig cfg, int maxNumSeqs, int eosToken, long seed) {
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        ByteTokenizer tokenizer = new ByteTokenizer();
        int[] eosTokens = eosToken < 0 ? new int[0] : new int[]{eosToken};
        return new LLMEngine(model, kvMgr, tokenizer, maxNumSeqs, eosTokens, seed);
    }

    /** greedy 解码参数：temperature=0（argmax），不启用 top-k/top-p */
    private static SamplingParams greedy(int maxTokens) {
        return new SamplingParams(maxTokens, 0f, 0, 1f);
    }

    /** 同步驱动 admit→decode→sweep，直到没有待办工作（带死循环保护） */
    private static void driveToCompletion(LLMEngine engine) {
        int guard = 0;
        while (engine.scheduler().hasWork()) {
            engine.step();
            if (++guard > 1_000_000) {
                fail("引擎未在合理步数内收敛，疑似死循环");
            }
        }
    }

    private static List<Integer> copy(List<Integer> src) {
        return new ArrayList<>(src);
    }

    // ========================================================================

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void singleRequestGeneratesExactlyMaxTokens() {
        LLMEngine engine = newEngine(ModelConfig.small(), 4, -1, SEED);
        int maxTokens = 8;

        Sequence seq = engine.addRequest("Hello", greedy(maxTokens), null);
        driveToCompletion(engine);

        assertTrue(seq.isFinished());
        assertEquals(Sequence.Stage.FINISHED, seq.stage());
        // 无 EOS 时应恰好生成 maxTokens 个 token
        assertEquals(maxTokens, seq.outputTokens().size());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void convenienceGenerateReturnsNonNullText() {
        LLMEngine engine = newEngine(ModelConfig.small(), 4, -1, SEED);
        String out = engine.generate("The quick brown fox", greedy(6));
        assertNotNull(out);
        // greedy 下每步都会产出 token，文本不应异常抛错；长度非负即可
        assertTrue(out.length() >= 0);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void greedyDecodingIsReproducibleAcrossEngines() {
        // 两个独立引擎，相同配置（randomInit 固定 seed=42）与相同 greedy 参数，输出应完全一致
        LLMEngine e1 = newEngine(ModelConfig.small(), 2, -1, SEED);
        LLMEngine e2 = newEngine(ModelConfig.small(), 2, -1, SEED + 999);

        Sequence s1 = e1.addRequest("reproduce me", greedy(10), null);
        driveToCompletion(e1);
        Sequence s2 = e2.addRequest("reproduce me", greedy(10), null);
        driveToCompletion(e2);

        assertEquals(copy(s1.outputTokens()), copy(s2.outputTokens()),
                "greedy 解码在相同权重下应可复现（与采样随机种子无关）");
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void int8KvCacheGreedyOutputMatchesF32() {
        // KV cache INT8 量化（阶段 3）端到端：同权重下 greedy 生成序列应与 f32 一致
        // （量化误差远小于相邻 logits 间距，argmax 不翻转）。
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model1 = ModelLoader.randomInit(cfg);
        TransformerModel model2 = ModelLoader.randomInit(cfg); // randomInit 固定 seed，权重相同
        KVCacheManager kvF32 = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        KVCacheManager kvInt8 = new KVCacheManager(512, cfg.blockSize(), cfg.dModel(), true);
        LLMEngine eF32 = new LLMEngine(model1, kvF32, new ByteTokenizer(), 2, new int[0], SEED);
        LLMEngine eInt8 = new LLMEngine(model2, kvInt8, new ByteTokenizer(), 2, new int[0], SEED);

        Sequence sF32 = eF32.addRequest("quantize my kv cache", greedy(12), null);
        driveToCompletion(eF32);
        Sequence sInt8 = eInt8.addRequest("quantize my kv cache", greedy(12), null);
        driveToCompletion(eInt8);

        assertTrue(sInt8.isFinished());
        assertEquals(copy(sF32.outputTokens()), copy(sInt8.outputTokens()),
                "INT8 KV cache 下 greedy 输出应与 f32 一致");
    }

    /** 小 KV 池引擎：逼出 preemption（两序列并发时池容不下全部 decode 增量） */
    private static LLMEngine newTinyPoolEngine(ModelConfig cfg, TransformerModel model, int numBlocks) {
        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize(), cfg.dModel());
        return new LLMEngine(model, kvMgr, new ByteTokenizer(), 4, new int[0], SEED);
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void preemptionRecomputeOutputMatchesUnpreemptedBaseline() {
        // 阶段 4：小池 + 两并发长序列 → 触发抢占；recompute 后所有请求完成，
        // 且 greedy 输出与宽松池基线完全一致（重算语义正确的最强验证）。
        ModelConfig cfg = ModelConfig.small();
        String prompt = "preempt me please ";
        int maxTokens = 40;

        // 基线：宽松池，无抢占
        LLMEngine base = newEngine(cfg, 4, -1, SEED);
        Sequence b1 = base.addRequest(prompt, greedy(maxTokens), null);
        Sequence b2 = base.addRequest(prompt, greedy(maxTokens), null);
        driveToCompletion(base);

        // 小池：两序列并发 decode 增量超出池容量，必触发抢占
        LLMEngine tiny = newTinyPoolEngine(cfg, ModelLoader.randomInit(cfg), 10);
        Sequence t1 = tiny.addRequest(prompt, greedy(maxTokens), null);
        Sequence t2 = tiny.addRequest(prompt, greedy(maxTokens), null);
        driveToCompletion(tiny);

        assertTrue(t1.isFinished() && t2.isFinished(), "小池下所有请求应最终完成");
        assertEquals(Sequence.Stage.FINISHED, t1.stage());
        assertEquals(Sequence.Stage.FINISHED, t2.stage());
        assertEquals(maxTokens, t1.outputTokens().size());
        assertEquals(maxTokens, t2.outputTokens().size());
        assertTrue(t1.preemptCount() + t2.preemptCount() > 0, "本场景应至少触发一次抢占");
        // recompute 正确性：与无抢占基线逐 token 一致
        assertEquals(copy(b1.outputTokens()), copy(t1.outputTokens()), "抢占重算后输出与基线不一致（请求 1）");
        assertEquals(copy(b2.outputTokens()), copy(t2.outputTokens()), "抢占重算后输出与基线不一致（请求 2）");
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void manyRequestsAllCompleteUnderTinyKvPool() {
        // 阶段 4：多长序列 + 小池，验证反复抢占无死循环、全部完成
        ModelConfig cfg = ModelConfig.small();
        LLMEngine tiny = newTinyPoolEngine(cfg, ModelLoader.randomInit(cfg), 10);
        List<Sequence> seqs = new ArrayList<>();
        String[] prompts = {"alpha request ", "beta request ", "gamma request ", "delta request "};
        for (String p : prompts) {
            seqs.add(tiny.addRequest(p, greedy(24), null));
        }
        driveToCompletion(tiny); // 内置 100 万步死循环保护
        for (Sequence s : seqs) {
            assertEquals(Sequence.Stage.FINISHED, s.stage(), "请求 " + s.id() + " 未完成");
            assertEquals(24, s.outputTokens().size());
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void eosTokenStopsGenerationEarly() {
        int maxTokens = 6;
        // 探针：无 EOS 先跑一遍，记录 greedy 生成序列（确定性）
        LLMEngine probe = newEngine(ModelConfig.small(), 2, -1, SEED);
        Sequence probeSeq = probe.addRequest("stop here", greedy(maxTokens), null);
        driveToCompletion(probe);
        assertEquals(maxTokens, probeSeq.outputTokens().size());
        // 取 decode 阶段会产生的一个 token 作为 EOS（第 2 个生成 token）
        int eos = probeSeq.outputTokens().get(1);

        // 设置该 token 为 EOS，并给足额度；greedy 确定性 → 生成到该 token 即停止
        LLMEngine engine = newEngine(ModelConfig.small(), 2, eos, SEED);
        Sequence seq = engine.addRequest("stop here", greedy(100), null);
        driveToCompletion(engine);

        assertTrue(seq.isFinished());
        assertTrue(seq.outputTokens().size() < 100, "命中 EOS 应远早于 maxTokens 停止");
        assertEquals(eos, seq.outputTokens().get(seq.outputTokens().size() - 1),
                "停止时的最后一个 token 应为 EOS");
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void continuousBatchingCompletesAllRequests() {
        // maxNumSeqs=2 但提交 5 个请求，强制排队 → 验证 admit/sweep 让请求随时进出且全部完成
        LLMEngine engine = newEngine(ModelConfig.small(), 2, -1, SEED);
        int nReq = 5;
        int maxTokens = 6;
        List<Sequence> seqs = new ArrayList<>();
        for (int i = 0; i < nReq; i++) {
            seqs.add(engine.addRequest("request-" + i, greedy(maxTokens), null));
        }

        driveToCompletion(engine);

        for (Sequence seq : seqs) {
            assertTrue(seq.isFinished(), "请求 " + seq.id() + " 应已完成");
            assertEquals(maxTokens, seq.outputTokens().size());
        }
        // 完成后 running 队列应清空
        assertEquals(0, engine.scheduler().runningCount());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void streamingCallbackFiresOncePerGeneratedToken() {
        LLMEngine engine = newEngine(ModelConfig.small(), 2, -1, SEED);
        List<String> streamed = new CopyOnWriteArrayList<>();
        int maxTokens = 7;

        Sequence seq = engine.addRequest("stream", greedy(maxTokens), streamed::add);
        driveToCompletion(engine);

        // 每生成一个 token 触发一次回调：回调次数 == 生成 token 数
        assertEquals(seq.outputTokens().size(), streamed.size());
        assertEquals(maxTokens, streamed.size());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void sampledTokensAlwaysWithinVocab() {
        ModelConfig cfg = ModelConfig.small();
        LLMEngine engine = newEngine(cfg, 2, -1, SEED);
        // 打开 temperature/top-k/top-p 采样路径
        Sequence seq = engine.addRequest("sample path", new SamplingParams(12, 0.8f, 40, 0.9f), null);
        driveToCompletion(engine);

        assertFalse(seq.outputTokens().isEmpty());
        for (int tok : seq.outputTokens()) {
            assertTrue(tok >= 0 && tok < cfg.vocabSize(),
                    "采样 token 必须落在 [0, vocabSize) 内，实际=" + tok);
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void gpt3SparseModelGeneratesEndToEnd() {
        // GPT-3 风格模型（交替 dense/局部带状稀疏注意力）端到端解码
        ModelConfig cfg = ModelConfig.gpt3Nano();
        LLMEngine engine = newEngine(cfg, 2, -1, SEED);
        int maxTokens = 6;

        Sequence seq = engine.addRequest("gpt3 nano end to end", greedy(maxTokens), null);
        driveToCompletion(engine);

        assertTrue(seq.isFinished());
        assertEquals(maxTokens, seq.outputTokens().size());
        for (int tok : seq.outputTokens()) {
            assertTrue(tok >= 0 && tok < cfg.vocabSize());
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void prefixSharingProducesIdenticalGreedyOutput() {
        // 同一 prompt 请求两次：第二次命中前缀共享（跳过部分 prefill），
        // greedy 确定性下两次输出必须完全一致——验证共享路径数值正确。
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, new int[0], SEED);
        // > blockSize(16) 的 prompt 才会产生可共享的完整 block
        String prompt = "The quick brown fox jumps over the lazy dog again and again";

        Sequence s1 = engine.addRequest(prompt, greedy(10), null);
        driveToCompletion(engine);
        Sequence s2 = engine.addRequest(prompt, greedy(10), null);
        driveToCompletion(engine);

        assertEquals(copy(s1.outputTokens()), copy(s2.outputTokens()),
                "命中前缀共享的请求应与不共享时输出完全一致");
        // 两个请求都已释放，注册过的前缀转入缓存态（证明 registerPrefix 生效）
        assertTrue(kvMgr.cachedBlocks() > 0, "完成请求的前缀 block 应留在缓存中供后续共享");
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void longPrefillIsChunkedAcrossSteps() {
        // tokenBudget=17：无 decode 时每步最多 prefill 17 token，
        // 80 token 的 prompt 必须分多步完成（Chunked Prefill 不一次性阻塞）。
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, 17,
                new int[0], new DefaultSamplingStrategy(SEED), new EosMaxTokensCriteria(), new FifoPolicy());
        String longPrompt = "This is a fairly long prompt that should be chunked into pieces!";

        Sequence seq = engine.addRequest(longPrompt, greedy(4), null);
        int promptLen = seq.promptTokens().length;
        assertTrue(promptLen > 17, "测试前提：prompt 应超过单步预算");

        engine.step(); // admit + 首个 chunk
        assertEquals(Sequence.Stage.PREFILL, seq.stage(), "首步后应仍在分块 prefill 中");
        assertTrue(seq.prefilledTokens() > 0 && seq.prefilledTokens() < promptLen,
                "首步只应完成部分 prefill，实际=" + seq.prefilledTokens());

        driveToCompletion(engine);
        assertTrue(seq.isFinished());
        assertEquals(4, seq.outputTokens().size());
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void chunkedPrefillOutputMatchesOneShot() {
        // 同一 prompt：大预算一步 prefill vs 小预算多步分块，greedy 输出必须完全一致
        String prompt = "The quick brown fox jumps over the lazy dog again and again";
        int maxTokens = 8;

        LLMEngine oneShot = newEngine(ModelConfig.small(), 2, -1, SEED);
        Sequence s1 = oneShot.addRequest(prompt, greedy(maxTokens), null);
        driveToCompletion(oneShot);

        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine chunked = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, 24,
                new int[0], new DefaultSamplingStrategy(SEED), new EosMaxTokensCriteria(), new FifoPolicy());
        Sequence s2 = chunked.addRequest(prompt, greedy(maxTokens), null);
        driveToCompletion(chunked);

        assertEquals(copy(s1.outputTokens()), copy(s2.outputTokens()),
                "分块 prefill 与整段 prefill 的 greedy 输出应完全一致");
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void decodeNotStarvedWhileLongPrefillChunks() {
        // 已在 decode 的请求与长 prefill 共存：每步 decode 优先，两者都正常推进
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, 24,
                new int[0], new DefaultSamplingStrategy(SEED), new EosMaxTokensCriteria(), new FifoPolicy());

        // 短请求先进入 decode，再提交长 prompt
        Sequence shortSeq = engine.addRequest("hi", greedy(12), null);
        for (int i = 0; i < 3; i++) engine.step(); // 短请求完成 prefill 并 decode 几步
        assertEquals(Sequence.Stage.DECODE, shortSeq.stage());

        Sequence longSeq = engine.addRequest(
                "a much longer prompt that needs multiple prefill chunks to finish processing", greedy(4), null);
        int shortBefore = shortSeq.outputTokens().size();
        for (int i = 0; i < 3; i++) engine.step();
        assertTrue(shortSeq.outputTokens().size() > shortBefore,
                "长 prefill 分块期间，decode 请求应持续推进不被阻塞");

        driveToCompletion(engine);
        assertTrue(shortSeq.isFinished() && longSeq.isFinished());
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void asyncServiceModeProcessesRequest() throws InterruptedException {
        // 服务模式：后台线程持续 step，验证 start()/addRequest()/stop() 全链路
        // （引擎空闲时阻塞在信号量上，addRequest 唤醒——本测试即验证事件唤醒路径）
        LLMEngine engine = newEngine(ModelConfig.small(), 4, -1, SEED);
        engine.start();
        try {
            List<String> streamed = new CopyOnWriteArrayList<>();
            Sequence seq = engine.addRequest("async", greedy(5), streamed::add);

            long deadline = System.currentTimeMillis() + 20_000;
            while (!seq.isFinished() && System.currentTimeMillis() < deadline) {
                Thread.sleep(5);
            }

            assertTrue(seq.isFinished(), "后台服务模式应在超时前完成请求");
            assertEquals(5, seq.outputTokens().size());
            assertEquals(seq.outputTokens().size(), streamed.size());
        } finally {
            engine.stop();
        }
    }

    // ========== 阶段五：指标与请求取消 ==========

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void metricsRecordGenerationStats() {
        LLMEngine engine = newEngine(ModelConfig.small(), 2, -1, SEED);
        engine.addRequest("metrics please", greedy(6), null);
        driveToCompletion(engine);

        Map<String, Object> snap = engine.metricsSnapshot();
        assertEquals(6L, ((Number) snap.get("generated_tokens")).longValue());
        assertEquals(1L, ((Number) snap.get("finished_requests")).longValue());
        assertTrue((Double) snap.get("avg_ttft_ms") > 0, "TTFT 应被打点（>0）");
        assertTrue((Double) snap.get("avg_itl_ms") > 0, "ITL 应被打点（>0）");
        assertEquals(0, ((Number) snap.get("running")).intValue());
        assertEquals(0, ((Number) snap.get("waiting")).intValue());
        assertTrue((Double) snap.get("throughput_tok_per_sec") > 0);
    }

    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void preemptionsAreCountedInMetrics() {
        // 小池触发抢占（复用阶段 4 场景）：指标中 preemptions 应 > 0
        ModelConfig cfg = ModelConfig.small();
        LLMEngine tiny = newTinyPoolEngine(cfg, ModelLoader.randomInit(cfg), 10);
        tiny.addRequest("alpha request ", greedy(24), null);
        tiny.addRequest("beta request ", greedy(24), null);
        driveToCompletion(tiny);
        assertTrue(((Number) tiny.metricsSnapshot().get("preemptions")).longValue() > 0,
                "小池场景应触发过抢占且被计入指标");
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void cancelledWaitingRequestIsAborted() {
        LLMEngine engine = newEngine(ModelConfig.small(), 2, -1, SEED);
        Sequence seq = engine.addRequest("cancel me", greedy(8), null);
        seq.cancel(); // 尚未 admit，仍在 waiting
        engine.step();
        assertEquals(Sequence.Stage.ABORTED, seq.stage());
        assertTrue(seq.isFinished());
        assertEquals(0, engine.scheduler().runningCount());
        assertEquals(0, engine.scheduler().waitingCount());
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void cancelledRunningRequestReleasesKv() {
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, new int[0], SEED);
        int freeInitial = kvMgr.freeBlocks();

        Sequence seq = engine.addRequest("cancel me mid decode", greedy(100), null);
        engine.step(); // prefill 完成并进入 decode
        assertEquals(Sequence.Stage.DECODE, seq.stage());
        assertTrue(kvMgr.freeBlocks() < freeInitial, "decode 中应已占用 KV block");

        seq.cancel();
        engine.step(); // 清扫：ABORT + 释放
        assertEquals(Sequence.Stage.ABORTED, seq.stage());
        assertTrue(seq.isFinished());
        // 释放后 block 回到空闲或前缀缓存态（prefill 完成时已注册），总量守恒
        assertEquals(freeInitial, kvMgr.freeBlocks() + kvMgr.cachedBlocks(),
                "取消后 KV block 应全部释放（空闲 + 缓存态）");
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void requestTooBigForPoolAbortsInsteadOfLivelock() {
        // 回归：池容不下单个请求且无在跑序列可释放时，必须 ABORT 而非永远留在
        // waiting 空转（旧行为下 step 无限循环、generate 死锁）
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        // 仅 1 个 block（容量 = blockSize 个 token），装不下 2*blockSize 的 prompt
        KVCacheManager kvMgr = new KVCacheManager(1, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, new int[0], SEED);

        Sequence seq = engine.addRequest("x".repeat(cfg.blockSize() * 2), greedy(4), null);
        engine.step();
        assertEquals(Sequence.Stage.ABORTED, seq.stage());
        assertTrue(seq.isFinished());
        assertFalse(engine.scheduler().hasWork(), "ABORT 后不应残留待办工作");
        assertEquals(1, kvMgr.freeBlocks());
    }
}
