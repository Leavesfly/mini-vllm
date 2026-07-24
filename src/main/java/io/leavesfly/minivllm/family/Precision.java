package io.leavesfly.minivllm.family;

/**
 * Precision —— 权重常驻内存的精度。
 *
 * 学习要点：
 * 1. F32：加载后全部转成 float，计算最快但内存占用最大（每参数 4 字节）。
 * 2. BF16：权重以 bf16 位模式常驻（每参数 2 字节），计算时在内核内转 f32，
 *    内存减半、带宽减半，是 CPU 推理大模型的常用折中。
 * 3. INT8：对称量化（每参数 1 字节 + per-row scale），预留给后续扩展。
 */
public enum Precision {
    F32,
    BF16,
    INT8
}
