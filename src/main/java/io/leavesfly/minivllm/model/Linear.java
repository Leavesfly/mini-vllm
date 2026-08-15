package io.leavesfly.minivllm.model;

import io.leavesfly.minivllm.math.Matmul;

import java.nio.ShortBuffer;

/**
 * 线性层 y = x·Wᵀ + b —— TransformerModel 中最常见的算子（Q/K/V/O 投影、FFN、lm_head 都是它）。
 *
 * 学习要点：
 * 1. 权重布局与 PyTorch nn.Linear 一致：W 形状 [outFeatures, inFeatures]（行优先），
 *    计算 y = x·Wᵀ + b，即 y[i] = Σ_p x[p] * W[i*in+p] + b[i]。
 * 2. 零依赖下没有 cuBLAS，矩阵乘退化为 Matmul.matVec 的逐行点积。
 * 3. 权重支持五种常驻方式（互斥，其余为 null）：
 *    - weight(F32)：默认，随机初始化 / GPT-2 路径使用；
 *    - weightBf16(BF16 位)：Qwen3 真实权重可直接以 bf16 常驻，内存/带宽减半；
 *    - weightInt8(INT8 量化)：per-row 对称量化，带宽为 bf16 的一半，decode 理论再提速 60-80%；
 *    - weightInt4(INT4 量化)：per-group 对称量化（Q4_0 风格打包），带宽为 int8 的一半；
 *    - weightMmapBf16(mmap 映射视图)：权重不落堆、由 OS 页缓存按需调页，
 *      物理内存小于模型体积也能运行；点积前逐行 bulk 读进线程私有缓冲再走 SIMD 内核。
 */
public final class Linear {

    private final float[] weight;        // [outFeatures, inFeatures] 行优先；其它模式为 null
    private final short[] weightBf16;    // bf16 位版权重；其它模式为 null
    private final byte[] weightInt8;     // int8 量化权重；其它模式为 null
    private final float[] scaleInt8;     // int8 per-row 缩放因子 [outFeatures]；非 int8 模式为 null
    private final byte[] weightInt4;     // int4 packed 量化权重（Q4_0 布局）；其它模式为 null
    private final float[] scaleInt4;     // int4 per-group 缩放因子 [outFeatures * groups]；非 int4 为 null
    private final int int4Group;         // int4 分组大小（元素数）；非 int4 为 0
    private final ShortBuffer weightMmapBf16; // mmap bf16 视图（绝对位置访问）；非 mmap 为 null
    private final float[] bias;          // [outFeatures]，可为 null
    private final int inFeatures;
    private final int outFeatures;

    public Linear(float[] weight, float[] bias, int inFeatures, int outFeatures) {
        this(weight, null, null, null, null, null, 0, null, bias, inFeatures, outFeatures);
    }

    private Linear(float[] weight, short[] weightBf16, byte[] weightInt8, float[] scaleInt8,
                   byte[] weightInt4, float[] scaleInt4, int int4Group, ShortBuffer weightMmapBf16,
                   float[] bias, int inFeatures, int outFeatures) {
        this.weight = weight;
        this.weightBf16 = weightBf16;
        this.weightInt8 = weightInt8;
        this.scaleInt8 = scaleInt8;
        this.weightInt4 = weightInt4;
        this.scaleInt4 = scaleInt4;
        this.int4Group = int4Group;
        this.weightMmapBf16 = weightMmapBf16;
        this.bias = bias;
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }


    // ─── 访问器 ───

    public float[] weight() {
        return weight;
    }

    public short[] weightBf16() {
        return weightBf16;
    }

    public byte[] weightInt8() {
        return weightInt8;
    }

    public float[] scaleInt8() {
        return scaleInt8;
    }

    public byte[] weightInt4() {
        return weightInt4;
    }

    public float[] scaleInt4() {
        return scaleInt4;
    }

    /** int4 分组大小（元素数）；非 int4 模式为 0 */
    public int int4Group() {
        return int4Group;
    }

    /** mmap bf16 权重视图（绝对位置访问）；非 mmap 模式为 null */
    public ShortBuffer weightMmapBf16() {
        return weightMmapBf16;
    }

    public float[] bias() {
        return bias;
    }

    public int inFeatures() {
        return inFeatures;
    }

    public int outFeatures() {
        return outFeatures;
    }

    /** 是否为 bf16 常驻权重 */
    public boolean isBf16() {
        return weightBf16 != null;
    }

    /** 是否为 int8 量化权重 */
    public boolean isInt8() {
        return weightInt8 != null;
    }

    /** 是否为 int4 量化权重 */
    public boolean isInt4() {
        return weightInt4 != null;
    }

    /** 是否为 mmap bf16 权重（不落堆，OS 页缓存按需调页） */
    public boolean isMmapBf16() {
        return weightMmapBf16 != null;
    }

    /**
     * 计算第 row 行权重与输入 x 的点积（封装 int4/int8/bf16/mmap/f32 分派）。
     * 用于 SwiGLU 等融合路径，避免外部重复判断权重类型。
     */
    public float dotRow(float[] x, int xOff, int row) {
        if (weightInt4 != null) {
            return Matmul.dotInt4(weightInt4, row * (inFeatures / 2), x, xOff, inFeatures,
                    scaleInt4, row * (inFeatures / int4Group), int4Group);
        }
        int wOff = row * inFeatures;
        if (weightInt8 != null) {
            return Matmul.dotInt8(weightInt8, wOff, x, xOff, inFeatures, scaleInt8[row]);
        } else if (weightBf16 != null) {
            return Matmul.dotBf16(weightBf16, wOff, x, xOff, inFeatures);
        } else if (weightMmapBf16 != null) {
            return Matmul.dotBf16Mapped(weightMmapBf16, wOff, x, xOff, inFeatures);
        } else {
            return Matmul.dot(weight, wOff, x, xOff, inFeatures);
        }
    }

    /** 单向量前向：x[inFeatures] -> y[outFeatures] */
    public float[] forward(float[] x) {
        float[] y;
        if (weightInt4 != null) {
            y = Matmul.matVecInt4(weightInt4, scaleInt4, x, outFeatures, inFeatures, int4Group);
        } else if (weightInt8 != null) {
            y = Matmul.matVecInt8(weightInt8, scaleInt8, x, outFeatures, inFeatures);
        } else if (weightBf16 != null) {
            y = Matmul.matVecBf16(weightBf16, x, outFeatures, inFeatures);
        } else if (weightMmapBf16 != null) {
            y = Matmul.matVecBf16Mapped(weightMmapBf16, x, outFeatures, inFeatures);
        } else {
            y = Matmul.matVec(weight, x, outFeatures, inFeatures);
        }
        if (bias != null) {
            for (int i = 0; i < outFeatures; i++) {
                y[i] += bias[i];
            }
        }
        return y;
    }

    /**
     * 批量前向：x[m, inFeatures] -> y[m, outFeatures]
     * prefill 阶段对整段 prompt 一次性投影时使用。
     *
     * 性能：按输出通道 o 单次并行（而非逐行 m 次 matVec 分发），
     * 每个权重行 weight[o] 只读一次并复用到全部 m 个输入行，缓存友好、降线程调度开销。
     *
     * 量化权重（int4/int8）的关键优化：先把本行反量化为 f32 行缓冲（每个输出通道仅一次），
     * 再对 m 个输入行走普通 f32 点积。若逐输入行调 dotInt4/dotInt8，解包/加宽链会对同一
     * 权重行重复执行 m 次——prefill chunk 动辄数百行，解包开销被放大数百倍（实测 int4
     * prefill 慢 10× 以上）。反量化一次后，m 个点积全部走最快的 f32 SIMD 路径。
     * 代价：量化路径结果与 matVec（量化域点积）存在 1-ulp 量级的浮点重排差异，
     * prefill/decode 间不再 bitwise 一致（推理场景可接受，e2e greedy 对拍验证无损）。
     */
    public float[] forwardBatch(float[] x, int m) {
        float[] y = new float[m * outFeatures];
        boolean int4 = weightInt4 != null;
        boolean int8 = weightInt8 != null;
        boolean bf16 = weightBf16 != null;
        boolean mmap = weightMmapBf16 != null;
        Matmul.parallelRows(outFeatures, o -> {
            int wOff = o * inFeatures;
            float b = bias != null ? bias[o] : 0f;
            if (int4 || int8) {
                // 每输出通道一个线程私有 f32 行缓冲（lambda 内分配，无共享）
                float[] wRow = int4 ? dequantInt4Row(o) : dequantInt8Row(o);
                for (int i = 0; i < m; i++) {
                    y[i * outFeatures + o] = Matmul.dot(wRow, 0, x, i * inFeatures, inFeatures) + b;
                }
                return;
            }
            if (mmap) {
                // 每输出通道读一次行进线程私有缓冲，复用到全部 m 个输入行（同量化路径的摊销思路）
                short[] wRow = Matmul.mmapRowBuffer(inFeatures);
                weightMmapBf16.get(wOff, wRow, 0, inFeatures);
                for (int i = 0; i < m; i++) {
                    y[i * outFeatures + o] = Matmul.dotBf16(wRow, 0, x, i * inFeatures, inFeatures) + b;
                }
                return;
            }
            for (int i = 0; i < m; i++) {
                float dot = bf16
                        ? Matmul.dotBf16(weightBf16, wOff, x, i * inFeatures, inFeatures)
                        : Matmul.dot(weight, wOff, x, i * inFeatures, inFeatures);
                y[i * outFeatures + o] = dot + b;
            }
        });
        return y;
    }

    /**
     * 第 o 行权重的 f32 反量化拷贝（包私有，int4/int8 批量前向专用）。
     * 供 {@link SwiGluFfn#forwardBatch} 等融合路径在每输出通道上解包一次、
     * 摊薄到全部输入行；f32/bf16 路径不应调用（dotRow 已是 SIMD 最优）。
     */
    float[] dequantRow(int o) {
        return weightInt4 != null ? dequantInt4Row(o) : dequantInt8Row(o);
    }

    /** 反量化一行 INT4 权重为 f32：w = (nibble-8) × scale[group]（Q4_0 半字节打包布局） */
    private float[] dequantInt4Row(int o) {
        float[] wRow = new float[inFeatures];
        int half = int4Group / 2;
        int groups = inFeatures / int4Group;
        int packBase = o * (inFeatures / 2);
        int scaleBase = o * groups;
        for (int g = 0; g < groups; g++) {
            float s = scaleInt4[scaleBase + g];
            int elemBase = g * int4Group;
            int pk = packBase + g * half;
            for (int j = 0; j < half; j++) {
                int packed = weightInt4[pk + j] & 0xFF;
                wRow[elemBase + j] = ((packed & 0xF) - 8) * s;        // 低 nibble → 组内前半
                wRow[elemBase + half + j] = ((packed >>> 4) - 8) * s; // 高 nibble → 组内后半
            }
        }
        return wRow;
    }

    /** 反量化一行 INT8 权重为 f32：w = q × scale */
    private float[] dequantInt8Row(int o) {
        float[] wRow = new float[inFeatures];
        int wOff = o * inFeatures;
        float s = scaleInt8[o];
        for (int i = 0; i < inFeatures; i++) {
            wRow[i] = weightInt8[wOff + i] * s;
        }
        return wRow;
    }

    /** 无偏置线性层便捷构造（F32 权重） */
    public static Linear of(float[] weight, int inFeatures, int outFeatures) {
        return new Linear(weight, null, null, null, null, null, 0, null, null, inFeatures, outFeatures);
    }

    /** 无偏置线性层便捷构造（BF16 位权重常驻） */
    public static Linear ofBf16(short[] weightBf16, int inFeatures, int outFeatures) {
        return new Linear(null, weightBf16, null, null, null, null, 0, null, null, inFeatures, outFeatures);
    }

    /** 无偏置线性层便捷构造（INT8 量化权重 + per-row scale） */
    public static Linear ofInt8(byte[] weightInt8, float[] scaleInt8, int inFeatures, int outFeatures) {
        return new Linear(null, null, weightInt8, scaleInt8, null, null, 0, null, null, inFeatures, outFeatures);
    }

    /**
     * 无偏置线性层便捷构造（INT4 量化权重 + per-group scale）。
     * 要求 inFeatures % groupSize == 0；packed 权重每行 inFeatures/2 字节。
     */
    public static Linear ofInt4(byte[] weightInt4, float[] scaleInt4, int groupSize,
                                int inFeatures, int outFeatures) {
        return new Linear(null, null, null, null, weightInt4, scaleInt4, groupSize,
                null, null, inFeatures, outFeatures);
    }

    /** 无偏置线性层便捷构造（mmap bf16 视图，权重不落堆） */
    public static Linear ofMmapBf16(ShortBuffer weightMmapBf16, int inFeatures, int outFeatures) {
        return new Linear(null, null, null, null, null, null, 0, weightMmapBf16, null,
                inFeatures, outFeatures);
    }

    /** 参数量（PyTorch: weight.numel() + bias.numel()；按逻辑形状计，量化存储不影响计数） */
    public long numParameters() {
        return (long) inFeatures * outFeatures + (bias == null ? 0L : bias.length);
    }
}
