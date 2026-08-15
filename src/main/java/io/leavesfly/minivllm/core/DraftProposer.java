package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.LlmModel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * DraftProposer —— 独立草稿模型的投机起草器（对照 vLLM 的 draft-model 投机模式）。
 *
 * 学习要点：
 * 1. 与 prompt-lookup（从上下文查 n-gram）不同，这里用一个更小的同词表模型自回归
 *    起草 k 个 token：小模型前向便宜，草稿质量（接受率）通常远高于 n-gram 查找。
 * 2. 草稿模型持有自己的 KV cache（独立 KVCacheManager + 每层 BlockTable），与目标
 *    模型的 KV 完全隔离——两者块布局（kvDim/层数）不同，绝不能共享 block。
 * 3. 不变式：每轮起草前，草稿 KV 覆盖 context[0, coveredLen) 且 coveredLen = totalLen-1
 *    （最后一个 context token 尚未起草过）。起草时先把未见 context 段一次 prefill 追平
 *    （末位 logits 直接产出第一个草稿），再逐 token decode 出其余草稿；目标模型验证后
 *    按新上下文长度截断草稿 KV，被拒绝草稿占用的块随之释放——与目标侧回滚对称。
 * 4. 追平段设计同时覆盖两种间隙来源：多序列混批时本序列走普通 decode 落下的 token，
 *    以及被抢占重算后重新进入 decode 的序列（草稿状态在抢占后仍然有效，无需重建）。
 * 5. 仅 greedy 启用：草稿取 argmax，与目标模型验证的 argmax 比对保证无损加速。
 */
final class DraftProposer {

    /** 草稿模型（与目标模型同词表，权重独立） */
    private final LlmModel draft;
    /** 草稿模型独占的 KV 池 */
    private final KVCacheManager draftKv;
    /** 序列 id → 草稿侧状态（首次投机时延迟创建，序列结束时释放） */
    private final Map<Integer, DraftState> states = new HashMap<>();

    /** 单序列的草稿侧状态：各层 BlockTable + 已覆盖的 context 前缀长度 */
    private static final class DraftState {
        final BlockTable[] bts;
        /** 草稿 KV 已覆盖的 context 前缀长度（不变式：新一轮起草前 = totalLen - 1） */
        int coveredLen;

        DraftState(int nLayer) {
            bts = new BlockTable[nLayer];
            for (int i = 0; i < nLayer; i++) {
                bts[i] = new BlockTable();
            }
        }
    }

    DraftProposer(LlmModel draft, KVCacheManager draftKv) {
        this.draft = draft;
        this.draftKv = draftKv;
    }

    /**
     * 起草至多 k 个 token。
     * 流程：追平未见 context（prefill 末位 logits 即第一个草稿的分布）→ 逐 token
     * decode 出剩余草稿；草稿命中停止 token 时提前截断（其后的草稿不可能被用到）。
     *
     * @return 草稿 token（长度 ≤ k）；草稿池容量不足时返回 null（调用方回退普通 decode）
     */
    int[] propose(Sequence seq, int k) {
        DraftState st = states.computeIfAbsent(seq.id(),
                id -> new DraftState(draft.config().nLayer()));
        int totalLen = seq.totalLen();
        for (BlockTable bt : st.bts) {
            if (!draftKv.ensureCapacity(bt, totalLen + k)) {
                return null; // 草稿池不足：本步放弃投机（状态保留，下步容量富余时重试）
            }
        }
        if (st.coveredLen >= totalLen) {
            // 防御：末位 token 的 logits 不可得，退一位重算（同位置 KV 覆写是幂等的）
            st.coveredLen = totalLen - 1;
        }
        int[] ctx = seq.contextTokens();
        // 追平段含最后一个 context token：prefill 返回的末位 logits 直接作为首个草稿分布
        int[] seg = Arrays.copyOfRange(ctx, st.coveredLen, totalLen);
        float[] logits = draft.prefillLogits(seg, draftKv, st.bts, st.coveredLen);
        st.coveredLen = totalLen;

        int[] draftTokens = new int[k];
        for (int j = 0; j < k; j++) {
            int t = argmax(logits);
            draftTokens[j] = t;
            if (isStopToken(seq, t)) {
                return Arrays.copyOf(draftTokens, j + 1);
            }
            if (j == k - 1) {
                break; // 最后一个草稿的 KV 不急着算：被接受后由下一轮追平段补齐
            }
            logits = draft.decodeLogits(t, st.coveredLen, draftKv, st.bts);
            st.coveredLen++;
        }
        return draftTokens;
    }

    /**
     * 验证完成后的草稿 KV 同步：newTotalLen 为目标模型的新上下文长度。
     * 不变式要求 coveredLen = newTotalLen - 1：超出部分是被拒绝草稿的 KV，截断释放；
     * 不足部分（草稿全被接受、bonus token 尚未起草）留给下一轮追平段补齐。
     */
    void sync(Sequence seq, int newTotalLen) {
        DraftState st = states.get(seq.id());
        if (st == null) {
            return;
        }
        int required = newTotalLen - 1;
        if (st.coveredLen > required) {
            for (BlockTable bt : st.bts) {
                draftKv.truncateTo(bt, required);
            }
            st.coveredLen = required;
        }
    }

    /** 序列结束/中止：释放草稿侧全部 KV block */
    void free(Sequence seq) {
        DraftState st = states.remove(seq.id());
        if (st != null) {
            for (BlockTable bt : st.bts) {
                draftKv.free(bt);
            }
        }
    }

    /** 观测/测试用：草稿 KV 池空闲 block 数 */
    int freeBlocks() {
        return draftKv.freeBlocks();
    }

    private static boolean isStopToken(Sequence seq, int token) {
        for (int eos : seq.eosTokens()) {
            if (token == eos) {
                return true;
            }
        }
        return false;
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }
}
