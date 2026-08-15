package io.leavesfly.minivllm.math;

import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
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
     * 公共 INT4 点积入口（per-group 对称量化，Q4_0 风格打包）。
     * w 为 packed 4-bit 权重（每字节两元素，wOff 为字节偏移），scales 为 per-group 缩放因子，
     * sOff 为该权重行在 scales 中的起始组下标。要求 len % groupSize == 0。
     * 权重仅 0.5 字节/元素，带宽为 int8 的一半。
     */
    public static float dotInt4(byte[] w, int wOff, float[] x, int xOff, int len,
                                float[] scales, int sOff, int groupSize) {
        return KERNEL.dotInt4(w, wOff, x, xOff, len, scales, sOff, groupSize);
    }

    /**
     * 加权累加：dst[dOff+i] += w * src[sOff+i]。
     * 用于 attention 的 V 加权求和，SIMD 向量化实现。
     */
    public static void axpy(float w, float[] src, int sOff, float[] dst, int dOff, int len) {
        KERNEL.axpy(w, src, sOff, dst, dOff, len);
    }

    /**
     * 堆外点积入口：a 为堆上 f32 激活，b 为 direct ByteBuffer（堆外 KV block），
     * bByteOff 为字节偏移（= 元素下标 × 4）。与 {@link #dot} 算术等价。
     */
    public static float dot(float[] a, int aOff, java.nio.ByteBuffer b, int bByteOff, int len) {
        return KERNEL.dotOffHeap(a, aOff, b, bByteOff, len);
    }

    /**
     * 堆外加权累加入口：dst[dOff+i] += w * src 第 (sByteOff+i*4) 字节处的 float。
     * KV cache 堆外化后 attention 的 V 加权求和直接读堆外数据，无需拷回堆上。
     */
    public static void axpy(float w, java.nio.ByteBuffer src, int sByteOff, float[] dst, int dOff, int len) {
        KERNEL.axpyOffHeap(w, src, sByteOff, dst, dOff, len);
    }

    /**
     * INT8 量化行的加权累加：dst[dOff+i] += w * scale * src[sOff+i]，src 为 signed byte。
     * KV cache INT8 量化后在量化域直接做 V 加权求和，避免整行反量化物化为 f32。
     */
    public static void axpyInt8(float w, byte[] src, int sOff, float[] dst, int dOff, int len, float scale) {
        KERNEL.axpyInt8(w, src, sOff, dst, dOff, len, scale);
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
     * 性能优化：
     * 1. 当前线程同样参与领取分块（而非空等），充分利用调用方 CPU 时间。
     * 2. 动态分块（原子计数器领取）而非静态均分：异构大小核（如 Apple Silicon
     *    的 P/E 核）下静态均分会让快核算完后空等慢核，动态领取小块使快核自然
     *    多干活，尾部等待时间接近于单块耗时。块内行连续，缓存友好性不变。
     */
    public static void parallelRows(int total, int threshold, IntConsumer rowTask) {
        if (total < threshold || CORES == 1) {
            for (int i = 0; i < total; i++) {
                rowTask.accept(i);
            }
            return;
        }
        int t = Math.min(CORES, total);
        // 块大小：每线程平均领 4 块，兼顾负载均衡粒度与领取开销（领取次数至多 4t，原子开销可忽略）
        int chunk = Math.max(1, (total + t * 4 - 1) / (t * 4));
        AtomicInteger cursor = new AtomicInteger(0);
        Runnable worker = () -> {
            int from;
            while ((from = cursor.getAndAdd(chunk)) < total) {
                int to = Math.min(from + chunk, total);
                for (int r = from; r < to; r++) {
                    rowTask.accept(r);
                }
            }
        };
        // 分发 t-1 个 worker 到线程池，当前线程也作为一个 worker 参与领取
        List<Future<?>> futures = new ArrayList<>(t - 1);
        for (int i = 1; i < t; i++) {
            futures.add(POOL.submit(worker));
        }
        worker.run();
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
                                 float[] b, int kB, int n) {
        if (k != kB) {
            throw new IllegalArgumentException("内部维度不匹配: " + k + " vs " + kB);
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
        int m = a.shape()[0], k = a.shape()[1];
        int k2 = b.shape()[0], n = b.shape()[1];
        float[] c = matmul(a.data(), m, k, b.data(), k2, n);
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

    // ─── mmap 权重内核：行拷贝 + 复用 SIMD 点积 ───

    /**
     * 线程私有行缓冲（只增不缩）：mmap 权重每次点积前把一行 bulk 读进此缓冲，
     * 再走堆内 SIMD 内核。避免逐行分配 short[] 的 GC 压力。
     * 约束：调用方必须在下次取用前消费完本次写入（禁止嵌套复用）。
     */
    private static final ThreadLocal<short[]> MMAP_ROW_BUF = ThreadLocal.withInitial(() -> new short[0]);

    /** 取线程私有行缓冲（容量不足时扩容），供 mmap 路径“读一行、复用多次”的场景 */
    public static short[] mmapRowBuffer(int minLen) {
        short[] buf = MMAP_ROW_BUF.get();
        if (buf.length < minLen) {
            buf = new short[minLen];
            MMAP_ROW_BUF.set(buf);
        }
        return buf;
    }

    /**
     * mmap bf16 权重行与 f32 激活的点积：与 {@link #dotBf16} 算术完全一致，
     * 仅权重来源从堆内 short[] 换成映射视图——bulk 读行进线程私有缓冲后转调 SIMD 内核。
     * 绝对位置 get 不移动 position，多线程可并发直读同一视图。
     */
    public static float dotBf16Mapped(ShortBuffer w, int wOff, float[] x, int xOff, int len) {
        short[] row = mmapRowBuffer(len);
        w.get(wOff, row, 0, len);
        return KERNEL.dotBf16(row, 0, x, xOff, len);
    }

    /** mmap bf16 权重版矩阵与向量乘：结构同 {@link #matVecBf16}，权重来自映射视图 */
    public static float[] matVecBf16Mapped(ShortBuffer w, float[] x, int m, int k) {
        float[] y = new float[m];
        parallelRows(m, i -> y[i] = dotBf16Mapped(w, i * k, x, 0, k));
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

    /**
     * INT4 量化权重版矩阵与向量乘：y(m) = W(m,k) · x(k)。
     * W 为 packed 4-bit 行优先（每行 k/2 字节），scales 为 per-group 因子 [m * (k/groupSize)]。
     * 带宽为 int8 的一半，decode 理论再提速近 2×。要求 k % groupSize == 0。
     */
    public static float[] matVecInt4(byte[] w, float[] scales, float[] x, int m, int k, int groupSize) {
        float[] y = new float[m];
        int groupsPerRow = k / groupSize;
        int packedRowBytes = k / 2;
        parallelRows(m, i -> y[i] = KERNEL.dotInt4(w, i * packedRowBytes, x, 0, k,
                scales, i * groupsPerRow, groupSize));
        return y;
    }
}
