package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sequence 与 Scheduler 单元测试 —— 验证请求生命周期与调度队列。
 */
class CoreTest {

    private static final int[] NO_EOS = new int[0];

    /** 构造测试用 Sequence：temperature=1、不启用 top-k/top-p */
    private static Sequence newSeq(int id, int[] prompt, int maxTokens, int[] eosTokens,
                                   int nLayer, Consumer<String> onToken) {
        return new Sequence(id, prompt, new SamplingParams(maxTokens, 1.0f, 0, 1.0f),
                eosTokens, nLayer, onToken);
    }

    // ========== Sequence 测试 ==========

    @Test
    void sequenceInitialState() {
        int[] prompt = {1, 2, 3};
        Sequence seq = newSeq(0, prompt, 10, NO_EOS, 2, null);
        assertEquals(Sequence.Stage.WAITING, seq.stage());
        assertEquals(3, seq.promptTokens().length);
        assertTrue(seq.outputTokens().isEmpty());
        assertEquals(3, seq.totalLen());
    }

    @Test
    void sequenceBlockTablesCreatedPerLayer() {
        int nLayer = 4;
        Sequence seq = newSeq(1, new int[]{1}, 10, NO_EOS, nLayer, null);
        assertEquals(nLayer, seq.blockTables().length);
        for (BlockTable bt : seq.blockTables()) {
            assertNotNull(bt);
            assertEquals(0, bt.numBlocks());
        }
    }

    @Test
    void sequenceIsFinishedByMaxTokens() {
        Sequence seq = newSeq(0, new int[]{1}, 3, NO_EOS, 2, null);
        assertFalse(seq.isFinished());
        seq.outputTokens().add(10);
        seq.outputTokens().add(20);
        assertFalse(seq.isFinished());
        seq.outputTokens().add(30);
        assertTrue(seq.isFinished()); // 达到 maxTokens=3
    }

    @Test
    void sequenceIsFinishedByEos() {
        int eos = 99;
        Sequence seq = newSeq(0, new int[]{1}, 100, new int[]{eos}, 2, null);
        seq.outputTokens().add(10);
        assertFalse(seq.isFinished());
        seq.outputTokens().add(eos);
        assertTrue(seq.isFinished()); // 命中 EOS
    }

    @Test
    void sequenceIsFinishedByStage() {
        Sequence seq = newSeq(0, new int[]{1}, 100, NO_EOS, 2, null);
        seq.setStage(Sequence.Stage.FINISHED);
        assertTrue(seq.isFinished());
    }

    @Test
    void sequenceIsFinishedByAborted() {
        Sequence seq = newSeq(0, new int[]{1}, 100, NO_EOS, 2, null);
        seq.setStage(Sequence.Stage.ABORTED);
        assertTrue(seq.isFinished());
    }

    @Test
    void sequenceTotalLen() {
        Sequence seq = newSeq(0, new int[]{1, 2, 3}, 10, NO_EOS, 2, null);
        assertEquals(3, seq.totalLen());
        seq.outputTokens().add(4);
        seq.outputTokens().add(5);
        assertEquals(5, seq.totalLen());
    }

    @Test
    void sequenceOnTokenCallback() {
        List<String> received = new ArrayList<>();
        Sequence seq = newSeq(0, new int[]{1}, 10, NO_EOS, 2, received::add);
        seq.onToken().accept("hello");
        seq.onToken().accept("world");
        assertEquals(List.of("hello", "world"), received);
    }

    // ========== Scheduler 测试 ==========

    @Test
    void schedulerAddToWaiting() {
        Scheduler scheduler = new Scheduler(8);
        Sequence seq = newSeq(0, new int[]{1}, 10, NO_EOS, 2, null);
        scheduler.add(seq);
        assertEquals(1, scheduler.waitingCount());
        assertTrue(scheduler.hasWork());
    }

    @Test
    void schedulerEmptyNoWork() {
        Scheduler scheduler = new Scheduler(8);
        assertFalse(scheduler.hasWork());
    }

    @Test
    void schedulerRunningList() {
        Scheduler scheduler = new Scheduler(4);
        assertEquals(0, scheduler.runningCount());
        Sequence seq = newSeq(0, new int[]{1}, 10, NO_EOS, 2, null);
        scheduler.addRunning(seq);
        assertTrue(scheduler.hasWork());
        assertEquals(1, scheduler.runningCount());
    }

    @Test
    void schedulerMaxNumSeqs() {
        Scheduler scheduler = new Scheduler(16);
        assertEquals(16, scheduler.maxNumSeqs());
    }

    @Test
    void schedulerWaitingIsConcurrentSafe() throws InterruptedException {
        Scheduler scheduler = new Scheduler(100);
        // 多线程添加请求
        Thread[] threads = new Thread[10];
        for (int t = 0; t < threads.length; t++) {
            final int tid = t;
            threads[t] = new Thread(() -> {
                for (int i = 0; i < 10; i++) {
                    scheduler.add(newSeq(tid * 10 + i, new int[]{1}, 10, NO_EOS, 2, null));
                }
            });
        }
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();
        assertEquals(100, scheduler.waitingCount());
    }

    // ========== 采样 penalty 测试（阶段五） ==========

    /** greedy 参数基础：temperature=0 → argmax，便于确定性验证 penalty 对 logits 的变换 */
    private static SamplingParams penaltyParams(float rep, float freq, Map<Integer, Float> bias) {
        return new SamplingParams(10, 0f, 0, 1f, rep, freq, 0f, bias);
    }

    @Test
    void logitBiasShiftsGreedyChoice() {
        DefaultSamplingStrategy strategy = new DefaultSamplingStrategy(1L);
        float[] logits = {1f, 2f, 0f};
        // 无偏置时 greedy 选 token 1
        assertEquals(1, strategy.sample(logits.clone(), penaltyParams(1f, 0f, null),
                new int[0], List.of()));
        // 对 token 2 加大正偏置后翻转 argmax
        assertEquals(2, strategy.sample(logits.clone(), penaltyParams(1f, 0f, Map.of(2, 100f)),
                new int[0], List.of()));
    }

    @Test
    void repetitionPenaltySuppressesHistoryToken() {
        DefaultSamplingStrategy strategy = new DefaultSamplingStrategy(1L);
        float[] logits = {1f, 2f, 0f};
        // token 1 在 prompt 中出现，r=3 → logit 2/3≈0.67 < 1 → argmax 转为 token 0
        assertEquals(0, strategy.sample(logits.clone(), penaltyParams(3f, 0f, null),
                new int[]{1}, List.of()));
    }

    @Test
    void frequencyPenaltyCountsOutputOccurrences() {
        DefaultSamplingStrategy strategy = new DefaultSamplingStrategy(1L);
        float[] logits = {1f, 2f, 0f};
        // token 1 已生成两次，freq=2 → 2 - 2×2 = -2 → argmax 转为 token 0
        assertEquals(0, strategy.sample(logits.clone(), penaltyParams(1f, 2f, null),
                new int[0], List.of(1, 1)));
    }

    @Test
    void cancelledSequenceReportsFinished() {
        Sequence seq = newSeq(0, new int[]{1}, 100, NO_EOS, 2, null);
        assertFalse(seq.isFinished());
        seq.cancel();
        assertTrue(seq.cancelled());
        assertTrue(seq.isFinished(), "取消后 isFinished 应立即为 true（等待方不阻塞）");
    }
}
