package io.leavesfly.minivllm.math;

/**
 * 矩阵乘法 —— TransformerModel 中最核心、计算量最大的算子。
 *
 * 学习要点：
 * 1. C[m,n] = A[m,k] · B[k,n]，三重循环是 GEMM 的最朴素实现。
 * 2. 真实框架会调用 BLAS/MKL/cuBLAS 做分块、向量化、并行化，但数学本质就是这个循环。
 * 3. 循环顺序 (i→p→j) 对 CPU 缓存友好程度不同，这里采用朴素版本，重在清晰。
 */
public final class Matmul {

    private Matmul() {
    }

    /**
     * 2D 矩阵乘：C = A(m,k) · B(k,n)
     * 输入为一维行优先数组。
     */
    public static float[] matmul(float[] a, int m, int k,
                                 float[] b, int k2, int n) {
        if (k != k2) {
            throw new IllegalArgumentException("内部维度不匹配: " + k + " vs " + k2);
        }
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                float aip = a[i * k + p];
                if (aip == 0f) continue; // 稀疏跳过，小优化
                int bRow = p * n;
                int cRow = i * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += aip * b[bRow + j];
                }
            }
        }
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
     * 常用于单 token 的线性层：logits = hidden · lmHead。
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
     * 注意：这里 W 按 [m,k] 行优先存储，与上面的 vecMat 的 [k,n] 布局不同，
     * 调用时需注意权重存放约定。
     */
    public static float[] matVec(float[] w, float[] x, int m, int k) {
        float[] y = new float[m];
        for (int i = 0; i < m; i++) {
            float s = 0f;
            int row = i * k;
            for (int p = 0; p < k; p++) {
                s += w[row + p] * x[p];
            }
            y[i] = s;
        }
        return y;
    }
}
