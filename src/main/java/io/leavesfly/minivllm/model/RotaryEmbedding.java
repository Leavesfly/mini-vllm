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
 * 5. 长上下文外推（可选）：llama3 风格频率缩放——按波长分三段处理 invFreq，
 *    波长短于 origLen/highFreq 不变（高频保局部细节），长于 origLen/lowFreq 除以
 *    factor（低频扩上下文），中间平滑插值（对齐 HF transformers 的 llama3 实现）。
 */
public final class RotaryEmbedding {

    private final int headDim;
    private final int halfDim;
    private final int maxSeqLen;
    private final float[] cos; // [maxSeqLen, halfDim] 行优先
    private final float[] sin; // [maxSeqLen, halfDim] 行优先

    public RotaryEmbedding(int headDim, int maxSeqLen, float theta) {
        this(headDim, maxSeqLen, theta, null);
    }

    /**
     * @param llama3Scaling llama3 频率缩放参数 [factor, lowFreqFactor, highFreqFactor,
     *                      originalMaxPosEmbeddings]；null 表示不缩放
     */
    public RotaryEmbedding(int headDim, int maxSeqLen, float theta, float[] llama3Scaling) {
        this.headDim = headDim;
        this.halfDim = headDim / 2;
        this.maxSeqLen = maxSeqLen;
        this.cos = new float[maxSeqLen * halfDim];
        this.sin = new float[maxSeqLen * halfDim];
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfDim; i++) {
                double invFreq = 1.0 / Math.pow(theta, 2.0 * i / headDim);
                if (llama3Scaling != null) {
                    invFreq = applyLlama3Scaling(invFreq, llama3Scaling);
                }
                double angle = pos * invFreq;
                cos[pos * halfDim + i] = (float) Math.cos(angle);
                sin[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }
    }

    /** llama3 频率缩放：短波长不变、长波长除 factor、中间平滑过渡（HF 对齐公式） */
    private static double applyLlama3Scaling(double invFreq, float[] s) {
        double factor = s[0];
        double lowWaveLen = s[3] / s[1];   // originalMaxLen / lowFreqFactor
        double highWaveLen = s[3] / s[2];  // originalMaxLen / highFreqFactor
        double waveLen = 2.0 * Math.PI / invFreq;
        if (waveLen < highWaveLen) {
            return invFreq;
        }
        if (waveLen > lowWaveLen) {
            return invFreq / factor;
        }
        double smooth = (s[3] / waveLen - s[1]) / (s[2] - s[1]);
        return (1 - smooth) * invFreq / factor + smooth * invFreq;
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
