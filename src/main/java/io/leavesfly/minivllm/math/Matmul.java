package io.leavesfly.minivllm.math;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.IntConsumer;

/**
 * 矩阵乘法 —— TransformerModel 中最核心、计算量最大的算子。
 *
 * 性能优化（阶段五）：
 * 1. 点积内核 {@link DotKernel}：运行时探测 jdk.incubator.vector，可用则 SIMD 加速，
 *    否则标量（HotSpot 也会自动向量化）。
 * 2. 多线程并行：matVec / matmul 按输出行分块到固定线程池，decode 阶段 memory-bound
 *    的逐行点积可获接近物理核数的加速比。
 * 3. 行分块无跨行依赖，并行结果与串行 bitwise 一致。
 *
 * 学习要点：
 * 1. C[m,n] = A[m,k] · B[k,n]，三重循环是 GEMM 的最朴素实现。
 * 2. 真实框架会调用 BLAS/MKL/cuBLAS 做分块、向量化、并行化，但数学本质就是这个循环。
 * 3. matVec（y = x·Wᵀ）是 decode 的热点：每生成一个 token 都要算一次，
 *    权重 W 按 [out, in] 行优先存储，每行一个点积，天然适合行并行 + SIMD。
 */
public final class Matmul {

    /** 选定的点积内核（向量优先，标量兜底） */
    public static final DotKernel KERNEL;
    /** 并行线程池（守护线程，按物理核数） */
    private static final int CORES = Math.max(1, Runtime.getRuntime().availableProcessors());
    private static final ExecutorService POOL = Executors.newFixedThreadPool(CORES, r -> {
        Thread t = new Thread(r, "mini-vllm-matmul");
        t.setDaemon(true);
        return t;
    });
    /** 输出行数 >= 该阈值才并行（避免小矩阵的线程调度开销） */
    private static final int PARALLEL_THRESHOLD = 1024;

    static {
        DotKernel k;
        try {
            Class.forName("jdk.incubator.vector.FloatVector");
            k = new VectorDotKernel();
        } catch (Throwable t) {
            k = new ScalarDotKernel();
        }
        KERNEL = k;
    }

    private Matmul() {
    }

    /** 内核名称（便于启动日志） */
    public static String kernelName() {
        return KERNEL.name();
    }

    /** CPU 核数 */
    public static int cores() {
        return CORES;
    }

    /**
     * 按 [0, total) 的行区间分块并行执行 rowTask。
     * total 小于阈值时串行，避免线程调度开销。
     */
    public static void parallelRows(int total, IntConsumer rowTask) {
        if (total < PARALLEL_THRESHOLD || CORES == 1) {
            for (int i = 0; i < total; i++) {
                rowTask.accept(i);
            }
            return;
        }
        int t = Math.min(CORES, total);
        int per = (total + t - 1) / t;
        List<Future<?>> futures = new ArrayList<>(t);
        for (int i = 0; i < t; i++) {
            int from = i * per;
            int to = Math.min(from + per, total);
            if (from >= to) {
                break;
            }
            futures.add(POOL.submit(() -> {
                for (int r = from; r < to; r++) {
                    rowTask.accept(r);
                }
            }));
        }
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * 2D 矩阵乘：C = A(m,k) · B(k,n)
     * 输入为一维行优先数组。并行按输出行 m 分块。
     */
    public static float[] matmul(float[] a, int m, int k,
                                 float[] b, int k2, int n) {
        if (k != k2) {
            throw new IllegalArgumentException("内部维度不匹配: " + k + " vs " + k2);
        }
        float[] c = new float[m * n];
        parallelRows(m, i -> {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                float aip = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += aip * b[bRow + j];
                }
            }
        });
        return c;
    }

    /**
     * Tensor 版矩阵乘，A:[m,k] B:[k,n] -> C:[m,n]
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        int m = a.shape[0], k = a.shape[1];
        int k2 = b.shape[0], n = b.shape[1];
        float[] c = matmul(a.data, m, k, b.data, k2, n);
        return new Tensor(c, m, n);
    }

    /**
     * 向量与矩阵乘：y = x(1,k) · W(k,n) -> y(n)
     * W 行优先 [k,n]（即 [in,out]），y[j] = Σ_p x[p]*W[p*n+j]。
     * 注意布局与 {@link #matVec}（W 为 [out,in]）互为转置，调用时需留意权重存放约定。
     * 非热点路径，保留标量实现。
     */
    public static float[] vecMat(float[] x, float[] w, int k, int n) {
        float[] y = new float[n];
        for (int p = 0; p < k; p++) {
            float xp = x[p];
            int wRow = p * n;
            for (int j = 0; j < n; j++) {
                y[j] += xp * w[wRow + j];
            }
        }
        return y;
    }

    /**
     * 矩阵与向量乘：y(m) = W(m,k) · x(k)
     * W 行优先 [m,k]，y[i] = dot(W 行 i, x)。并行按输出行 m 分块。
     */
    public static float[] matVec(float[] w, float[] x, int m, int k) {
        float[] y = new float[m];
        parallelRows(m, i -> y[i] = KERNEL.dot(w, i * k, x, 0, k));
        return y;
    }
}
