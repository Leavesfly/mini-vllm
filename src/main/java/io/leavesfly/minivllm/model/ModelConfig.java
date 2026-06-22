package io.leavesfly.minivllm.model;

/**
 * 模型超参数配置。
 *
 * 学习要点：
 * 1. dModel 是隐藏层维度，nHead 是注意力头数，headDim = dModel / nHead。
 * 2. dFfn 是 FFN 中间层维度，GPT-2 通常取 4 * dModel。
 * 3. blockSize 是 PagedAttention 的物理块大小（每个 block 容纳的 token 数）。
 * 4. 学习项目用小尺寸即可跑通，例如 dModel=64, nHead=4, nLayer=2, vocabSize=256。
 */
public final class ModelConfig {

    public int vocabSize = 256;
    public int dModel = 64;
    public int nHead = 4;
    public int nLayer = 2;
    public int dFfn = 256;     // 4 * dModel
    public int blockSize = 16; // PagedAttention 块大小
    public int maxSeqLen = 512;
    public float layerNormEps = 1e-5f;
    public boolean tieWordEmbeddings = true; // GPT-2 词嵌入与 lm_head 共享权重

    /** 每个注意力头的维度 */
    public int headDim() {
        return dModel / nHead;
    }

    /** 便捷构造（学习用小模型） */
    public static ModelConfig small() {
        return new ModelConfig();
    }

    @Override
    public String toString() {
        return "ModelConfig{vocab=" + vocabSize + ", d=" + dModel + ", h=" + nHead
                + ", L=" + nLayer + ", ffn=" + dFfn + ", blk=" + blockSize + "}";
    }
}
