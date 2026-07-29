package io.leavesfly.minivllm.math;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * DotKernel —— 点积计算内核抽象（阶段五性能优化）。
 *
 * 学习要点：
 * 1. decode 阶段的热点是把权重矩阵的每一行与激活向量做点积（matVec），
 *    点积的向量化程度直接决定 token 生成速度。
 * 2. 策略模式：运行时探测 jdk.incubator.vector 模块是否可用，
 *    可用则用 {@link VectorDotKernel}（SIMD 分块累加），否则回退 {@link ScalarDotKernel}。
 * 3. 两种内核的浮点累加顺序不同，结果不保证 bitwise 一致（推理场景可接受）。
 * 4. 编译需 --add-modules jdk.incubator.vector（pom 已配置）；
 *    运行时未加该模块则自动回退标量，功能不受影响。
 */
interface DotKernel {

    /** 计算 a[aOff..aOff+len) 与 b[bOff..bOff+len) 的点积 */
    float dot(float[] a, int aOff, float[] b, int bOff, int len);

    /**
     * BF16 权重 × F32 激活 的点积：a 为 bf16 位（short），b 为 f32。
     * a[i] 的 bf16 位左移 16 位即得 f32（与 {@link Bf16#bf16ToFloat} 一致），
     * 与"先整体转 f32 再点积"算术等价，但权重只占一半内存/带宽。
     */
    float dotBf16(short[] a, int aOff, float[] b, int bOff, int len);
    
    /**
     * INT8 量化权重 × F32 激活 的点积：w 为 signed byte（-127~127），scale 为行缩放因子。
     * 结果 = scale * Σ(w[i] * x[i])。权重每元素仅 1 字节，带宽为 bf16 的一半。
     */
    float dotInt8(byte[] w, int wOff, float[] x, int xOff, int len, float scale);

    /**
     * 加权累加：dst[dOff+i] += w * src[sOff+i]，i∈[0,len)。
     * 用于 attention 的 V 加权求和，向量化后可显著加速长上下文 decode。
     */
    void axpy(float w, float[] src, int sOff, float[] dst, int dOff, int len);

    /** 内核名称（启动日志用） */
    String name();
}

/**
 * 标量点积内核：朴素循环，依赖 HotSpot 自动向量化，作为兜底实现。
 */
final class ScalarDotKernel implements DotKernel {

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public float dotBf16(short[] a, int aOff, float[] b, int bOff, int len) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            // bf16 位 -> f32：低 16 位左移 16（符号/零扩展均可，高位被移出）
            float w = Float.intBitsToFloat(a[aOff + i] << 16);
            sum += w * b[bOff + i];
        }
        return sum;
    }

    @Override
    public float dotInt8(byte[] w, int wOff, float[] x, int xOff, int len, float scale) {
        float sum = 0f;
        for (int i = 0; i < len; i++) {
            sum += (float) w[wOff + i] * x[xOff + i];
        }
        return sum * scale;
    }

    @Override
    public void axpy(float w, float[] src, int sOff, float[] dst, int dOff, int len) {
        for (int i = 0; i < len; i++) {
            dst[dOff + i] += w * src[sOff + i];
        }
    }

    @Override
    public String name() {
        return "scalar";
    }
}

/**
 * Vector API SIMD 点积内核。
 *
 * 按硬件优选物种（SPECIES_PREFERRED）的 lane 宽度分块：
 * 主循环用 fma（乘加融合）累积到向量累加器，最后 reduceLanes 归约，
 * 不足一个 lane 宽度的尾部用标量处理。
 *
 * 性能关键：多累加器展开（4 路）。FMA 指令延迟 ~3-4 周期，单累加器存在
 * 串行依赖链（每次 fma 必须等上一次完成），单核吞吐仅峰值的 ~25%；
 * 4 个独立累加器可填满流水线，在 Apple Silicon（NEON 128-bit）上接近 4× 单核提速。
 */
final class VectorDotKernel implements DotKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    /**
     * 与 SPECIES 同 lane 数的 short/int 物种，用于 bf16 -> f32 加宽。
     * short 取"半位宽"形状（元素 16 位，lane 数与 float 一致，不会越界读取），
     * int 取与 float 同形状（元素 32 位）；S2I 为同 lane 数的扩展转换（part=0）。
     */
    private static final VectorSpecies<Short> SHORT_SPECIES =
            VectorSpecies.of(short.class, VectorShape.forBitSize(SPECIES.vectorBitSize() / 2));
    private static final VectorSpecies<Integer> INT_SPECIES =
            VectorSpecies.of(int.class, SPECIES.vectorShape());
    /**
     * int8 路径的 byte/short 物种：一次加载 2N 个 byte（N = float lane 数）。
     * byte 取 vectorBitSize/2 形状（最小 64-bit，避开 32-bit 非法形状陷阱），
     * B2S 加宽到满宽 short（2N lanes），再 S2I 分 part 0/1 得两组 int，I2F 转 float。
     * 注意：不可用 B2F 一步转换（非硬件直接支持，JIT 会退化为逐 lane 标量循环，
     * 实测慢 ~50×）；B2S/S2I/I2F 均有对应硬件指令，全链路寄存器内完成。
     */
    private static final VectorSpecies<Byte> BYTE_SPECIES =
            VectorSpecies.of(byte.class, VectorShape.forBitSize(SPECIES.vectorBitSize() / 2));
    private static final VectorSpecies<Short> SHORT_FULL_SPECIES =
            VectorSpecies.of(short.class, SPECIES.vectorShape());

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        int i = 0;
        int lanes = SPECIES.length();
        int step = lanes * 4;
        // 4 路独立累加器：打破 FMA 延迟链，填满流水线
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        FloatVector acc2 = FloatVector.zero(SPECIES);
        FloatVector acc3 = FloatVector.zero(SPECIES);
        for (int bound4 = len - len % step; i < bound4; i += step) {
            acc0 = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
            acc1 = FloatVector.fromArray(SPECIES, a, aOff + i + lanes)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes), acc1);
            acc2 = FloatVector.fromArray(SPECIES, a, aOff + i + lanes * 2)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes * 2), acc2);
            acc3 = FloatVector.fromArray(SPECIES, a, aOff + i + lanes * 3)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes * 3), acc3);
        }
        for (; i < SPECIES.loopBound(len); i += lanes) {
            acc0 = FloatVector.fromArray(SPECIES, a, aOff + i)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
        }
        float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public float dotBf16(short[] a, int aOff, float[] b, int bOff, int len) {
        int i = 0;
        int lanes = SPECIES.length();
        int step = lanes * 4;
        // 4 路独立累加器（同 dot）：bf16 转换链（加载/S2I/LSHL）与 FMA 交错填满流水线
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        FloatVector acc2 = FloatVector.zero(SPECIES);
        FloatVector acc3 = FloatVector.zero(SPECIES);
        for (int bound4 = len - len % step; i < bound4; i += step) {
            acc0 = loadBf16(a, aOff + i).fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
            acc1 = loadBf16(a, aOff + i + lanes)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes), acc1);
            acc2 = loadBf16(a, aOff + i + lanes * 2)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes * 2), acc2);
            acc3 = loadBf16(a, aOff + i + lanes * 3)
                    .fma(FloatVector.fromArray(SPECIES, b, bOff + i + lanes * 3), acc3);
        }
        for (; i < SPECIES.loopBound(len); i += lanes) {
            acc0 = loadBf16(a, aOff + i).fma(FloatVector.fromArray(SPECIES, b, bOff + i), acc0);
        }
        float sum = acc0.add(acc1).add(acc2.add(acc3)).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += Float.intBitsToFloat(a[aOff + i] << 16) * b[bOff + i];
        }
        return sum;
    }

    /** 加载 lanes 个 bf16 位并加宽为 f32 向量：short -> int（S2I）-> 左移 16 -> 重解释 float */
    private static FloatVector loadBf16(short[] a, int off) {
        ShortVector sv = ShortVector.fromArray(SHORT_SPECIES, a, off);
        IntVector iv = (IntVector) sv.convertShape(VectorOperators.S2I, INT_SPECIES, 0);
        return iv.lanewise(VectorOperators.LSHL, 16).reinterpretAsFloats();
    }

    @Override
    public float dotInt8(byte[] w, int wOff, float[] x, int xOff, int len, float scale) {
        // 全向量化路径：一次加载 2N 个 byte，B2S 加宽到满宽 short，再 S2I 分 part 0/1
        // 得两组 int，I2F 转 float 后 FMA（各步均有硬件指令，勿用会退化标量的 B2F）。
        // 转换全在寄存器内完成，无标量中转缓冲；双累加器打破 FMA 延迟链。
        // 带宽收益：权重每元素仅 1 字节（bf16 的一半）。
        int i = 0;
        int lanes = SPECIES.length();
        int step = lanes * 2; // 每轮处理 2N 元素（一次 byte 向量加载）
        FloatVector acc0 = FloatVector.zero(SPECIES);
        FloatVector acc1 = FloatVector.zero(SPECIES);
        for (int bound2 = len - len % step; i < bound2; i += step) {
            ByteVector bv = ByteVector.fromArray(BYTE_SPECIES, w, wOff + i);
            ShortVector sv = (ShortVector) bv.convertShape(VectorOperators.B2S, SHORT_FULL_SPECIES, 0);
            FloatVector f0 = (FloatVector) ((IntVector) sv
                    .convertShape(VectorOperators.S2I, INT_SPECIES, 0))
                    .convert(VectorOperators.I2F, 0);
            FloatVector f1 = (FloatVector) ((IntVector) sv
                    .convertShape(VectorOperators.S2I, INT_SPECIES, 1))
                    .convert(VectorOperators.I2F, 0);
            acc0 = f0.fma(FloatVector.fromArray(SPECIES, x, xOff + i), acc0);
            acc1 = f1.fma(FloatVector.fromArray(SPECIES, x, xOff + i + lanes), acc1);
        }
        float sum = acc0.add(acc1).reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += (float) w[wOff + i] * x[xOff + i];
        }
        return sum * scale;
    }

    @Override
    public void axpy(float w, float[] src, int sOff, float[] dst, int dOff, int len) {
        int i = 0;
        int lanes = SPECIES.length();
        int bound = SPECIES.loopBound(len);
        FloatVector wv = FloatVector.broadcast(SPECIES, w);
        for (; i < bound; i += lanes) {
            FloatVector sv = FloatVector.fromArray(SPECIES, src, sOff + i);
            FloatVector dv = FloatVector.fromArray(SPECIES, dst, dOff + i);
            sv.fma(wv, dv).intoArray(dst, dOff + i);
        }
        for (; i < len; i++) {
            dst[dOff + i] += w * src[sOff + i];
        }
    }

    @Override
    public String name() {
        return "vector(" + SPECIES.length() + " lanes)";
    }
}
