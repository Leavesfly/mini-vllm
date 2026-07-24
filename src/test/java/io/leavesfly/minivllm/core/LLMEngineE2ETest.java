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
    void asyncServiceModeProcessesRequest() throws InterruptedException {
        // 服务模式：后台线程持续 step，验证 start()/addRequest()/stop() 全链路
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
}
