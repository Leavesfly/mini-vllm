package io.leavesfly.minivllm.math;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
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
 */
final class VectorDotKernel implements DotKernel {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    @Override
    public float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        int i = 0;
        FloatVector acc = FloatVector.zero(SPECIES);
        for (; i < SPECIES.loopBound(len); i += SPECIES.length()) {
            FloatVector va = FloatVector.fromArray(SPECIES, a, aOff + i);
            FloatVector vb = FloatVector.fromArray(SPECIES, b, bOff + i);
            acc = va.fma(vb, acc);
        }
        float sum = acc.reduceLanes(VectorOperators.ADD);
        for (; i < len; i++) {
            sum += a[aOff + i] * b[bOff + i];
        }
        return sum;
    }

    @Override
    public String name() {
        return "vector(" + SPECIES.length() + " lanes)";
    }
}
