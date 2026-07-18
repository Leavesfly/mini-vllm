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
    /** 并行线程池大小：默认物理核数，可用 -Dmatmul.threads 覆盖（1 即纯串行，便于对比） */
    private static final int CORES = resolveCores();
    private static final ExecutorService POOL = Executors.newFixedThreadPool(CORES, r -> {
        Thread t = new Thread(r, "mini-vllm-matmul");
        t.setDaemon(true);
        return t;
    });
    /** 输出行数 >= 该阈值才并行（避免小矩阵的线程调度开销） */
    private static final int PARALLEL_THRESHOLD = 1024;

    static {
        // 内核选择优先级：-Dmatmul.kernel=scalar|vector 显式指定 > 运行时探测 Vector API
        String forced = System.getProperty("matmul.kernel");
        DotKernel k;
        if ("scalar".equalsIgnoreCase(forced)) {
            k = new ScalarDotKernel();
        } else {
            try {
                Class.forName("jdk.incubator.vector.FloatVector");
                k = new VectorDotKernel();
            } catch (Throwable t) {
                k = new ScalarDotKernel();
            }
        }
        KERNEL = k;
    }

    /** 解析线程数：-Dmatmul.threads 优先，否则取可用处理器数，下限 1 */
    private static int resolveCores() {
        int def = Math.max(1, Runtime.getRuntime().availableProcessors());
        Integer override = Integer.getInteger("matmul.threads");
        if (override != null && override > 0) {
            return override;
        }
        return def;
    }

    private Matmul() {
    }

    /** 内核名称（便于启动日志） */
    public static String kernelName() {
        return KERNEL.name();
    }

    /** 一行诊断信息（内核 / 线程 / 并行阈值），供启动时打印 */
    public static String diagnostics() {
        return "matmul kernel=" + KERNEL.name() + ", threads=" + CORES
                + ", parallelThreshold=" + PARALLEL_THRESHOLD;
    }

    /** CPU 核数 */
    public static int cores() {
        return CORES;
    }

    /**
     * 公共点积入口：a[aOff..aOff+len) · b[bOff..bOff+len)。
     * DotKernel 接口为包私有，跨包（model 等）需经此方法复用 SIMD／标量内核。
     */
    public static float dot(float[] a, int aOff, float[] b, int bOff, int len) {
        return KERNEL.dot(a, aOff, b, bOff, len);
    }

    /**
     * 公共 BF16 点积入口：a 为 bf16 位（short 权重），b 为 f32 激活。
     * 与 {@link #dot} 算术等价，但权重只占一半内存/带宽（decode 内存受限时更快）。
     */
    public static float dotBf16(short[] a, int aOff, float[] b, int bOff, int len) {
        return KERNEL.dotBf16(a, aOff, b, bOff, len);
    }

    /**
     * 公共 INT8 点积入口：w 为 signed byte 量化权重，x 为 f32 激活，scale 为行缩放因子。
     * 结果 = scale * Σ(w[i] * x[i])。权重仅 1 字节/元素，带宽为 bf16 的一半。
     */
    public static float dotInt8(byte[] w, int wOff, float[] x, int xOff, int len, float scale) {
        return KERNEL.dotInt8(w, wOff, x, xOff, len, scale);
    }

    /**
     * 加权累加：dst[dOff+i] += w * src[sOff+i]。
     * 用于 attention 的 V 加权求和，SIMD 向量化实现。
     */
    public static void axpy(float w, float[] src, int sOff, float[] dst, int dOff, int len) {
        KERNEL.axpy(w, src, sOff, dst, dOff, len);
    }

    /**
     * 按 [0, total) 的行区间分块并行执行 rowTask。
     * total 小于阈值时串行，避免线程调度开销。
     */
    public static void parallelRows(int total, IntConsumer rowTask) {
        parallelRows(total, PARALLEL_THRESHOLD, rowTask);
    }

    /**
     * 带自定义阈值的行分块并行。
     * total < threshold 或单核时串行。用于"列数少但每列开销大"的场景
     *（如多头注意力：nHead 仅十几个，但长上下文时每头计算量大，值得并行）。
     *
     * 性能优化：当前线程参与第 0 块计算（而非空等），充分利用调用方 CPU 时间。
     */
    public static void parallelRows(int total, int threshold, IntConsumer rowTask) {
        if (total < threshold || CORES == 1) {
            for (int i = 0; i < total; i++) {
                rowTask.accept(i);
            }
            return;
        }
        int t = Math.min(CORES, total);
        int per = (total + t - 1) / t;
        // 分发 chunks 1..t-1 到线程池，当前线程处理 chunk 0
        List<Future<?>> futures = new ArrayList<>(t - 1);
        for (int i = 1; i < t; i++) {
            int from = i * per;
            int to = Math.min(from + per, total);
            if (from >= to) break;
            futures.add(POOL.submit(() -> {
                for (int r = from; r < to; r++) {
                    rowTask.accept(r);
                }
            }));
        }
        // 当前线程处理第 0 块（不浪费等待时间）
        int end0 = Math.min(per, total);
        for (int r = 0; r < end0; r++) {
            rowTask.accept(r);
        }
        // 等待其余线程完成
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

    /**
     * BF16 权重版矩阵与向量乘：y(m) = W(m,k) · x(k)，W 为 bf16 位（short）行优先 [m,k]。
     * 与 {@link #matVec} 完全同构，仅权重存储为 bf16——decode/prefill 的投影与 lm_head 复用此路径。
     */
    public static float[] matVecBf16(short[] w, float[] x, int m, int k) {
        float[] y = new float[m];
        parallelRows(m, i -> y[i] = KERNEL.dotBf16(w, i * k, x, 0, k));
        return y;
    }

    /**
     * INT8 量化权重版矩阵与向量乘：y(m) = W(m,k) · x(k)。
     * W 为 signed byte 行优先 [m,k]，scale[m] 为每行缩放因子。
     * y[i] = scale[i] * Σ(w[i*k+j] * x[j])。带宽为 bf16 的一半，decode 理论提速 60-80%。
     */
    public static float[] matVecInt8(byte[] w, float[] scale, float[] x, int m, int k) {
        float[] y = new float[m];
        parallelRows(m, i -> y[i] = KERNEL.dotInt8(w, i * k, x, 0, k, scale[i]));
        return y;
    }
}
