package io.leavesfly.minivllm.math;

/**
 * 张量类 —— 用一维 float[] + 形状描述来模拟多维张量。
 *
 * 学习要点：
 * 1. 真实框架(PyTorch/TensorRT)的张量在内存中也是一维连续存储，靠 shape + stride 描述维度。
 * 2. 这里采用行优先(row-major)布局：对于 shape [rows, cols]，元素 (i,j) 位于 data[i*cols + j]。
 * 3. 不实现自动求导——推理引擎只需要前向计算。
 */
public final class Tensor {

    /** 底层一维数据，行优先存储 */
    private final float[] data;
    /** 形状，例如 {seqLen, dModel} */
    private final int[] shape;

    public Tensor(float[] data, int... shape) {
        if (data == null) throw new IllegalArgumentException("data 不能为 null");
        int expect = 1;
        for (int s : shape) {
            if (s <= 0) throw new IllegalArgumentException("shape 各维必须 > 0，得到 " + s);
            expect *= s;
        }
        if (expect != data.length && shape.length > 0) {
            throw new IllegalArgumentException(
                    "shape 体积 " + expect + " 与数据长度 " + data.length + " 不一致");
        }
        this.data = data;
        this.shape = shape;
    }

    /** 底层一维数据（行优先存储） */
    public float[] data() { return data; }

    /** 形状数组 */
    public int[] shape() { return shape; }

    /** 创建全零张量 */
    public static Tensor zeros(int... shape) {
        int n = 1;
        for (int s : shape) {
            n *= s;
        }
        return new Tensor(new float[n], shape);
    }

    /** 元素总数 */
    public int size() {
        return data.length;
    }

    /** 维度数 */
    public int dim() {
        return shape.length;
    }

    // ---------- 2D 访问（shape=[rows, cols]） ----------

    /** 取第 i 行第 j 列 */
    public float get2d(int i, int j) {
        return data[i * shape[1] + j];
    }

    /** 设第 i 行第 j 列 */
    public void set2d(int i, int j, float v) {
        data[i * shape[1] + j] = v;
    }

    /** 取第 i 行（返回的是拷贝） */
    public float[] row(int i) {
        int cols = shape[1];
        float[] r = new float[cols];
        System.arraycopy(data, i * cols, r, 0, cols);
        return r;
    }

    // ---------- 3D 访问（shape=[a, b, c]） ----------

    /** 取 [i,j,k] */
    public float get3d(int i, int j, int k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    /** 设 [i,j,k] */
    public void set3d(int i, int j, int k, float v) {
        data[(i * shape[1] + j) * shape[2] + k] = v;
    }

    /** 行优先扁平索引（任意维度） */
    public int flatIndex(int... idx) {
        int off = 0;
        for (int d = 0; d < idx.length; d++) {
            off = off * shape[d] + idx[d];
        }
        return off;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Tensor(shape=[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(shape[i]);
        }
        sb.append("], data=").append(ArrayUtil.toString(data, 8)).append(")");
        return sb.toString();
    }
}
