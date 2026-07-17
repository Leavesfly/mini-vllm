package io.leavesfly.minivllm.model;

/**
 * RotaryEmbedding —— 旋转位置编码（RoPE），Qwen / LLaMA 系列的位置编码方案。
 *
 * 学习要点：
 * 1. 核心思想：把每个注意力头的 headDim 维向量按"前半/后半"配对（half-split，
 *    即 GPT-NeoX 风格，Qwen 的实际实现），每对 (x[i], x[i+half]) 在二维平面上
 *    旋转 angle = pos * invFreq[i] 度：
 *      out[i]      = x[i]*cos - x[i+half]*sin
 *      out[i+half] = x[i+half]*cos + x[i]*sin
 *    注意不是 GPT-J 的交错式（interleaved）配对——这是与 HF 对齐时最常见的坑。
 * 2. invFreq[i] = theta^(-2i/headDim)，i ∈ [0, headDim/2)；Qwen3 theta=1000000。
 * 3. 相对位置特性：Q·K 的点积只依赖相对位置 (m-n)，因此 RoPE 只作用 Q/K，不作用 V。
 * 4. cos/sin 表按位置预计算 [maxSeqLen, headDim/2]，推理时查表即可。
 */
public final class RotaryEmbedding {

    private final int headDim;
    private final int halfDim;
    private final int maxSeqLen;
    private final float[] cos; // [maxSeqLen, halfDim] 行优先
    private final float[] sin; // [maxSeqLen, halfDim] 行优先

    public RotaryEmbedding(int headDim, int maxSeqLen, float theta) {
        this.headDim = headDim;
        this.halfDim = headDim / 2;
        this.maxSeqLen = maxSeqLen;
        this.cos = new float[maxSeqLen * halfDim];
        this.sin = new float[maxSeqLen * halfDim];
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double invFreq = 1.0 / Math.pow(theta, 2.0 * i / headDim);
                double angle = pos * invFreq;
                cos[pos * halfDim + i] = (float) Math.cos(angle);
                sin[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }
    }

    public int headDim() {
        return headDim;
    }

    public int maxSeqLen() {
        return maxSeqLen;
    }

    /**
     * 对位于 offset 的一个头向量（长度 headDim）就地施加位置 pos 的旋转。
     */
    public void applyInPlace(float[] x, int offset, int pos) {
        if (pos < 0 || pos >= maxSeqLen) {
            throw new IllegalArgumentException("RoPE 位置越界: " + pos + " (max=" + maxSeqLen + ")");
        }
        int row = pos * halfDim;
        for (int i = 0; i < halfDim; i++) {
            float c = cos[row + i];
            float s = sin[row + i];
            float x1 = x[offset + i];
            float x2 = x[offset + i + halfDim];
            x[offset + i] = x1 * c - x2 * s;
            x[offset + i + halfDim] = x2 * c + x1 * s;
        }
    }

    /** 查询用：位置 pos、第 i 对的 cos 值（测试对齐用） */
    public float cosAt(int pos, int i) {
        return cos[pos * halfDim + i];
    }

    /** 查询用：位置 pos、第 i 对的 sin 值（测试对齐用） */
    public float sinAt(int pos, int i) {
        return sin[pos * halfDim + i];
    }
}
