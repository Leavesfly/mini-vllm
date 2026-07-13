package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Tensor;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.weights.ModelLoader;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * TransformerModel 单元测试 —— 验证 GPT-2/GPT-3 前向与 PyTorch 风格对外接口。
 *
 * 核心保证：
 * 1. PyTorch 风格 forward() 输出形状为 [seqLen, vocabSize]。
 * 2. forward() 最后一行 logits 与推理引擎 prefill+logits 路径数值一致（dense 层共用同一注意力核心）。
 * 3. GPT-3 交替稀疏注意力模型可正常前向，且逐层稀疏性符合 isSparseLayer 约定。
 */
class TransformerModelTest {

    private static BlockTable[] newBlockTables(ModelConfig cfg, KVCacheManager kv, int promptLen) {
        BlockTable[] bts = new BlockTable[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            bts[i] = new BlockTable();
            assertTrue(kv.ensureCapacity(bts[i], promptLen));
        }
        return bts;
    }

    @Test
    void forwardReturnsLogitsWithCorrectShape() {
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        int[] prompt = {5, 9, 13, 2, 7};

        Tensor logits = model.forward(prompt);
        assertEquals(2, logits.dim());
        assertEquals(prompt.length, logits.shape[0]);
        assertEquals(cfg.vocabSize, logits.shape[1]);
        for (float v : logits.data) {
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    void forwardMatchesPrefillPathForDenseModel() {
        ModelConfig cfg = ModelConfig.small(); // dense (GPT-2 风格)
        TransformerModel model = ModelLoader.randomInit(cfg);
        int[] prompt = {5, 9, 13, 2, 7, 100, 42};

        // 引擎路径：prefillLogits（走 PagedAttention KV cache，直接返回最后位置 logits）
        KVCacheManager kv = new KVCacheManager(256, cfg.blockSize, cfg.dModel);
        BlockTable[] bts = newBlockTables(cfg, kv, prompt.length);
        float[] logitsPrefill = model.prefillLogits(prompt, kv, bts, 0);

        // PyTorch 风格路径：forward 取最后一行
        float[] logitsForward = model.forwardLastLogits(prompt);

        assertEquals(logitsPrefill.length, logitsForward.length);
        assertArrayEquals(logitsPrefill, logitsForward, 1e-4f,
                "PyTorch 风格 forward 应与 prefill 路径数值一致（dense 层共用注意力核心）");
    }

    @Test
    void forwardTensorOverloadEqualsIntArrayOverload() {
        ModelConfig cfg = ModelConfig.small();
        TransformerModel model = ModelLoader.randomInit(cfg);
        int[] prompt = {1, 2, 3, 4};

        Tensor a = model.forward(prompt);
        Tensor b = model.forward(new Tensor(new float[]{1f, 2f, 3f, 4f}, 4));
        assertArrayEquals(a.data, b.data, 0f);
    }

    @Test
    void gpt3SparseModelRunsAndAlternatesLayers() {
        ModelConfig cfg = ModelConfig.gpt3Nano();
        assertTrue(cfg.useSparseAttention);
        // 交替：偶数层 dense，奇数层 sparse
        assertFalse(cfg.isSparseLayer(0));
        assertTrue(cfg.isSparseLayer(1));
        assertFalse(cfg.isSparseLayer(2));

        TransformerModel model = ModelLoader.randomInit(cfg);
        assertEquals(cfg.nLayer, model.numLayers());
        assertTrue(model.numParameters() > 0);

        int[] prompt = {10, 20, 30, 40, 50, 60};
        Tensor logits = model.forward(prompt);
        assertEquals(prompt.length, logits.shape[0]);
        assertEquals(cfg.vocabSize, logits.shape[1]);
        for (float v : logits.data) {
            assertTrue(Float.isFinite(v));
        }
    }

    @Test
    void sparseForwardMatchesSparsePrefill() {
        ModelConfig cfg = ModelConfig.gpt3Nano();
        TransformerModel model = ModelLoader.randomInit(cfg);
        // 用超过 sparseWindow 的序列，确保稀疏窗口被真正触发
        int seqLen = cfg.sparseWindow + 5;
        int[] prompt = new int[seqLen];
        for (int i = 0; i < seqLen; i++) prompt[i] = (i * 7 + 3) % cfg.vocabSize;

        KVCacheManager kv = new KVCacheManager(512, cfg.blockSize, cfg.dModel);
        BlockTable[] bts = newBlockTables(cfg, kv, seqLen);
        float[] logitsPrefill = model.prefillLogits(prompt, kv, bts, 0);

        float[] logitsForward = model.forwardLastLogits(prompt);
        assertArrayEquals(logitsPrefill, logitsForward, 1e-4f,
                "稀疏模型下 forward 与 prefill 也应一致（两条路径遵守同一窗口约束）");
    }
}
