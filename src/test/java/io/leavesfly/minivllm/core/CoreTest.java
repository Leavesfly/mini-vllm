package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sequence 与 Scheduler 单元测试 —— 验证请求生命周期与调度队列。
 */
class CoreTest {

    // ========== Sequence 测试 ==========

    @Test
    void sequenceInitialState() {
        int[] prompt = {1, 2, 3};
        Sequence seq = new Sequence(0, prompt, 10, 1.0f, 0, 1.0f, -1, 2, null);
        assertEquals(Sequence.Stage.WAITING, seq.stage());
        assertEquals(3, seq.promptTokens().length);
        assertTrue(seq.outputTokens().isEmpty());
        assertEquals(3, seq.totalLen());
    }

    @Test
    void sequenceBlockTablesCreatedPerLayer() {
        int nLayer = 4;
        Sequence seq = new Sequence(1, new int[]{1}, 10, 1.0f, 0, 1.0f, -1, nLayer, null);
        assertEquals(nLayer, seq.blockTables().length);
        for (BlockTable bt : seq.blockTables()) {
            assertNotNull(bt);
            assertEquals(0, bt.numBlocks());
        }
    }

    @Test
    void sequenceIsFinishedByMaxTokens() {
        Sequence seq = new Sequence(0, new int[]{1}, 3, 1.0f, 0, 1.0f, -1, 2, null);
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
        Sequence seq = new Sequence(0, new int[]{1}, 100, 1.0f, 0, 1.0f, eos, 2, null);
        seq.outputTokens().add(10);
        assertFalse(seq.isFinished());
        seq.outputTokens().add(eos);
        assertTrue(seq.isFinished()); // 命中 EOS
    }

    @Test
    void sequenceIsFinishedByStage() {
        Sequence seq = new Sequence(0, new int[]{1}, 100, 1.0f, 0, 1.0f, -1, 2, null);
        seq.setStage(Sequence.Stage.FINISHED);
        assertTrue(seq.isFinished());
    }

    @Test
    void sequenceIsFinishedByAborted() {
        Sequence seq = new Sequence(0, new int[]{1}, 100, 1.0f, 0, 1.0f, -1, 2, null);
        seq.setStage(Sequence.Stage.ABORTED);
        assertTrue(seq.isFinished());
    }

    @Test
    void sequenceTotalLen() {
        Sequence seq = new Sequence(0, new int[]{1, 2, 3}, 10, 1.0f, 0, 1.0f, -1, 2, null);
        assertEquals(3, seq.totalLen());
        seq.outputTokens().add(4);
        seq.outputTokens().add(5);
        assertEquals(5, seq.totalLen());
    }

    @Test
    void sequenceOnTokenCallback() {
        List<String> received = new ArrayList<>();
        Sequence seq = new Sequence(0, new int[]{1}, 10, 1.0f, 0, 1.0f, -1, 2, received::add);
        seq.onToken().accept("hello");
        seq.onToken().accept("world");
        assertEquals(List.of("hello", "world"), received);
    }

    // ========== Scheduler 测试 ==========

    @Test
    void schedulerAddToWaiting() {
        Scheduler scheduler = new Scheduler(8);
        Sequence seq = new Sequence(0, new int[]{1}, 10, 1.0f, 0, 1.0f, -1, 2, null);
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
        Sequence seq = new Sequence(0, new int[]{1}, 10, 1.0f, 0, 1.0f, -1, 2, null);
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
                    scheduler.add(new Sequence(tid * 10 + i, new int[]{1}, 10,
                            1.0f, 0, 1.0f, -1, 2, null));
                }
            });
        }
        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();
        assertEquals(100, scheduler.waitingCount());
    }
}
