package io.leavesfly.minivllm.math;

import java.util.Random;

/**
 * 数组工具类 —— 零依赖下的基础数值运算集合。
 *
 * 学习要点：在没有任何科学计算库的情况下，所有张量运算都退化为对一维 float[] 的逐元素操作。
 * 这正是理解“深度学习底层在算什么”的最佳途径。
 */
public final class ArrayUtil {

    private ArrayUtil() {
    }

    /** 创建长度为 n 的全零数组 */
    public static float[] zeros(int n) {
        return new float[n];
    }

    /** 用值 v 填充数组 */
    public static void fill(float[] a, float v) {
        for (int i = 0; i < a.length; i++) {
            a[i] = v;
        }
    }

    /** 拷贝前 len 个元素到新数组 */
    public static float[] copyOf(float[] a, int len) {
        float[] r = new float[len];
        System.arraycopy(a, 0, r, 0, Math.min(len, a.length));
        return r;
    }

    /** 求和 */
    public static float sum(float[] a) {
        float s = 0f;
        for (float v : a) {
            s += v;
        }
        return s;
    }

    /** 求最大值 */
    public static float max(float[] a) {
        float m = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > m) {
                m = a[i];
            }
        }
        return m;
    }

    /** 就地 exp，返回自身便于链式调用 */
    public static float[] expInPlace(float[] a) {
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) Math.exp(a[i]);
        }
        return a;
    }

    /** 就地缩放 */
    public static void scaleInPlace(float[] a, float s) {
        for (int i = 0; i < a.length; i++) {
            a[i] *= s;
        }
    }

    /** 就地逐元素加法：dst[i] += src[i]（残差连接的核心操作） */
    public static void addInPlace(float[] dst, float[] src) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] += src[i];
        }
    }

    /** 点积 */
    public static float dot(float[] a, float[] b) {
        float s = 0f;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /** 打印数组（调试用） */
    public static String toString(float[] a, int limit) {
        int n = Math.min(a.length, limit);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", a[i]));
        }
        if (a.length > limit) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }

    /**
     * Box-Muller 生成正态分布随机数组（GPT-2/GPT-3 风格初始化用）。
     * 每次生成一对正态随机数，充分利用三角函数计算。
     *
     * @param rnd 随机数生成器
     * @param n   数组长度
     * @param std 标准差（GPT-2 标准初始化用 0.02）
     * @return 长度为 n 的正态分布随机数组
     */
    public static float[] randNormal(Random rnd, int n, float std) {
        float[] r = new float[n];
        for (int i = 0; i < n; i += 2) {
            double u1 = Math.max(rnd.nextDouble(), 1e-12);
            double u2 = rnd.nextDouble();
            double radius = Math.sqrt(-2.0 * Math.log(u1));
            r[i] = (float) (radius * Math.cos(2 * Math.PI * u2) * std);
            if (i + 1 < n) {
                r[i + 1] = (float) (radius * Math.sin(2 * Math.PI * u2) * std);
            }
        }
        return r;
    }
}
