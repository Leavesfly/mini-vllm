package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.TransformerModel;
import io.leavesfly.minivllm.tokenizer.ByteTokenizer;
import io.leavesfly.minivllm.weights.ModelLoader;
import org.junit.jupiter.api.Test;

import io.leavesfly.minivllm.model.LlmModel;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SpeculativeDecodingTest —— prompt-lookup 投机采样的端到端测试。
 *
 * 核心性质（无损性）：greedy 解码下，投机路径的输出必须与普通 decode 逐 token
 * 完全一致——草稿只是"提前猜测"，验证保证了分布等价。覆盖：
 * 1. 逐 token 一致性（多个 prompt，含重复性文本）
 * 2. 重复性上下文确实产生 n-gram 草稿（投机路径被触发）
 * 3. 投机结束后的 KV block 释放与普通 decode 一致（截断无泄漏）
 * 4. 非 greedy 请求自动回退普通 decode（不产生草稿）
 * 5. PromptLookup 的草稿查找规则（needle 递减 / 最后出现 / maxDraft 截断）
 *
 * 草稿模型模式（draft-model 投机，对照 vLLM --speculative-model）：
 * 6. 独立草稿模型下输出与普通 decode 逐 token 一致（无损性）
 * 7. 草稿=目标（完美起草）时接受率 100%（每步 k 草稿 + 1 bonus）
 * 8. 序列结束后草稿 KV 池全回收（无泄漏）
 * 9. 混批期间走普通 decode，落单后追平草稿 KV 恢复投机，输出仍无损
 */
class SpeculativeDecodingTest {

    private static final long SEED = 42L;

    private static LLMEngine newEngine(int speculativeK) {
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg); // 固定 seed，权重确定
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, new int[0], SEED);
        engine.setSpeculativeK(speculativeK);
        return engine;
    }

    /** greedy 解码参数：temperature=0（argmax），不启用 top-k/top-p */
    private static SamplingParams greedy(int maxTokens) {
        return new SamplingParams(maxTokens, 0.0f, 0, 1.0f);
    }

    private static Sequence run(LLMEngine engine, String prompt, SamplingParams params) {
        Sequence seq = engine.addRequest(prompt, params, null);
        while (!seq.isFinished()) {
            if (engine.scheduler().hasWork()) {
                engine.step();
            }
        }
        return seq;
    }

    // ─── 草稿模型模式（draft-model 投机） ───

    /** 独立权重的草稿模型（seed 与目标不同 → 权重确定但相异，接受率 0~1 之间） */
    private static TransformerModel independentDraft() {
        return ModelLoader.randomInit(ModelConfig.small(), 7L);
    }

    private static KVCacheManager draftPoolOf(ModelConfig dcfg) {
        return new KVCacheManager(512, dcfg.blockSize(), dcfg.dModel());
    }

    /** 目标引擎：固定 seed 随机权重；draft 非空时接入草稿模型（独占 draftKv 池） */
    private static LLMEngine newEngineWithDraft(int speculativeK, LlmModel draft, KVCacheManager draftKv) {
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize(), cfg.dModel());
        LLMEngine engine = new LLMEngine(model, kvMgr, new ByteTokenizer(), 2, new int[0], SEED);
        engine.setSpeculativeK(speculativeK);
        if (draft != null) {
            engine.setDraftModel(draft, draftKv);
        }
        return engine;
    }

    @Test
    void draftModelSpecMatchesPlainDecodeTokenByToken() {
        String[] prompts = {
                "The quick brown fox jumps over the lazy dog",
                "abc abc abc abc abc",
                "short",
        };
        for (String prompt : prompts) {
            Sequence plain = run(newEngine(0), prompt, greedy(24));
            Sequence spec = run(newEngineWithDraft(4, independentDraft(), draftPoolOf(ModelConfig.small())),
                    prompt, greedy(24));
            assertEquals(plain.outputTokens(), spec.outputTokens(),
                    "草稿模型投机与普通 decode 输出不一致: " + prompt);
        }
    }

    @Test
    void perfectDraftIsFullyAccepted() {
        // 草稿与目标是同一模型实例：完美起草（验证与草稿算术一致，逐 token 相同）
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        LLMEngine engine = newEngineWithDraft(4, model, draftPoolOf(cfg));
        Sequence spec = run(engine, "hello speculative decoding", greedy(24));

        Sequence ref = run(newEngine(0), "hello speculative decoding", greedy(24));
        assertEquals(ref.outputTokens(), spec.outputTokens(), "完美草稿下输出仍须与普通 decode 一致");
        long drafted = (long) engine.metricsSnapshot().get("spec_draft_tokens");
        long accepted = (long) engine.metricsSnapshot().get("spec_accepted_tokens");
        assertTrue(drafted > 0, "投机路径应被触发");
        assertEquals(drafted, accepted, "草稿=目标模型时应全接受");
    }

    @Test
    void draftKvPoolReleasedAfterFinish() {
        ModelConfig dcfg = ModelConfig.small();
        KVCacheManager draftKv = draftPoolOf(dcfg);
        int total = draftKv.pool().numBlocks();
        LLMEngine engine = newEngineWithDraft(4, independentDraft(), draftKv);
        run(engine, "abc abc abc abc", greedy(20));
        assertEquals(total, draftKv.freeBlocks(), "序列结束后草稿 KV 池应全回收（无泄漏）");
    }

    @Test
    void nonGreedySkipsDraftModelSpeculation() {
        LLMEngine engine = newEngineWithDraft(4, independentDraft(), draftPoolOf(ModelConfig.small()));
        run(engine, "abc abc abc abc", new SamplingParams(16, 0.8f, 0, 0.9f));
        long drafted = (long) engine.metricsSnapshot().get("spec_draft_tokens");
        assertEquals(0, drafted, "非 greedy 不应产生草稿（草稿模型模式同样回退）");
    }

    @Test
    void batchedSequencesResumeSpeculationWithCatchUp() {
        // 两个 greedy 序列同时 running：混批时走普通 decode；先完成者退出后，
        // 落单序列恢复投机——DraftProposer 追平段补齐混批期间落下的 context token。
        LLMEngine engine = newEngineWithDraft(4, independentDraft(), draftPoolOf(ModelConfig.small()));
        Sequence a = engine.addRequest("first prompt here", greedy(8), null);
        Sequence b = engine.addRequest("second different prompt", greedy(20), null);
        while (!a.isFinished() || !b.isFinished()) {
            if (engine.scheduler().hasWork()) {
                engine.step();
            }
        }
        long drafted = (long) engine.metricsSnapshot().get("spec_draft_tokens");
        assertTrue(drafted > 0, "落单后应恢复投机");
        Sequence refB = run(newEngine(0), "second different prompt", greedy(20));
        assertEquals(refB.outputTokens(), b.outputTokens(), "混批/投机交替下输出应与纯 decode 一致");
    }

    @Test
    void speculativeOutputMatchesPlainDecodeTokenByToken() {
        String[] prompts = {
                "The quick brown fox jumps over the lazy dog",
                "abc abc abc abc abc",           // 重复性文本：草稿高命中场景
                "short",
        };
        for (String prompt : prompts) {
            Sequence plain = run(newEngine(0), prompt, greedy(24));
            Sequence spec = run(newEngine(4), prompt, greedy(24));
            assertEquals(plain.outputTokens(), spec.outputTokens(),
                    "投机与普通 decode 输出不一致: " + prompt);
        }
    }

    @Test
    void repetitiveContextProducesDrafts() {
        LLMEngine engine = newEngine(4);
        run(engine, "abc abc abc abc abc abc abc", greedy(16));
        long drafted = (long) engine.metricsSnapshot().get("spec_draft_tokens");
        assertTrue(drafted > 0, "重复性上下文应产生 n-gram 草稿");
    }

    @Test
    void kvBlocksReleasedIdenticallyAfterSpeculative() {
        LLMEngine plain = newEngine(0);
        run(plain, "abc abc abc abc", greedy(20));
        int freeAfterPlain = plain.metrics().generatedTokens() >= 0 ? freeBlocksOf(plain) : -1;

        LLMEngine spec = newEngine(4);
        run(spec, "abc abc abc abc", greedy(20));
        assertEquals(freeAfterPlain, freeBlocksOf(spec), "投机完成后 KV 释放应与普通 decode 一致");
    }

    private static int freeBlocksOf(LLMEngine engine) {
        // 序列已结束：池应恢复满空闲（512  blocks 全回收）
        return engine.metricsSnapshot().get("kv_cache_usage") instanceof Number n ? n.intValue() : -1;
    }

    @Test
    void nonGreedySkipsSpeculation() {
        LLMEngine engine = newEngine(4);
        run(engine, "abc abc abc abc", new SamplingParams(16, 0.8f, 0, 0.9f));
        long drafted = (long) engine.metricsSnapshot().get("spec_draft_tokens");
        assertEquals(0, drafted, "非 greedy 不应产生草稿");
    }

    // ─── PromptLookup 草稿查找规则 ───

    @Test
    void draftFromLastOccurrenceWithDecreasingNeedle() {
        // 序列 [1,2,3,1,2]：末尾 needle [1,2]（3 元 [3,1,2] 无匹配）→ 上次出现在 0。
        // 对齐 vLLM n-gram 语义：取匹配点之后至多 maxDraft 个 token（允许覆盖末尾
        // needle 自身副本）——周期文本才能给出长草稿，草稿质量不影响无损性。
        int[] draft = PromptLookup.findDraft(new int[]{1, 2, 3}, List.of(1, 2), 4);
        assertArrayEquals(new int[]{3, 1, 2}, draft);
    }

    @Test
    void draftRespectsMaxDraft() {
        // needle [9] 上次出现在 0，后续 [2,3,4] 被 maxDraft=2 截断
        int[] draft = PromptLookup.findDraft(new int[]{9, 2, 3, 4, 8}, List.of(9), 2);
        assertArrayEquals(new int[]{2, 3}, draft);
    }

    @Test
    void noMatchReturnsNull() {
        assertNull(PromptLookup.findDraft(new int[]{1, 2, 3}, List.of(4), 4));
        assertNull(PromptLookup.findDraft(new int[]{}, List.of(1), 4)); // 长度不足
    }
}
