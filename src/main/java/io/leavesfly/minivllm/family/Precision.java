package io.leavesfly.minivllm.family;

/**
 * Precision —— 权重常驻内存的精度。
 *
 * 学习要点：
 * 1. F32：加载后全部转成 float，计算最快但内存占用最大（每参数 4 字节）。
 * 2. BF16：权重以 bf16 位模式常驻（每参数 2 字节），计算时在内核内转 f32，
 *    内存减半、带宽减半，是 CPU 推理大模型的常用折中。
 * 3. INT8：对称量化（每参数 1 字节 + per-row scale），decode 带宽再减半。
 * 4. INT4：per-group 对称量化（约 0.5 字节/参数 + per-group scale），带宽为 int8 的一半；
 *    Embedding 保持 INT8（查表随机访问，对量化误差不敏感）。
 * 5. MMAP：bf16 磁盘精度 + 内存映射驻留（对照 llama.cpp 默认 mmap）：权重不落堆，
 *    由 OS 页缓存按需调页，物理内存小于模型体积也能运行；要求磁盘张量为 BF16。
 *
 * 实测备注（Qwen3-0.6B，Apple Silicon，decode）：bf16 ≈35 / int8 ≈25 / int4 ≈4 tok/s——
 * 小模型未打满内存带宽瓶颈，量化解包链（B2S→S2I→I2F）的算力开销是净损失。
 * 量化的价值在大模型：4B+ 必须量化才能装下内存，且带宽瓶颈真实存在。
 */
public enum Precision {
    F32,
    BF16,
    INT8,
    INT4,
    MMAP
}
