package io.leavesfly.minivllm.weights;

/**
 * Bf16 —— BF16 / F16 半精度浮点到 F32 的转换。
 *
 * 学习要点：
 * 1. BF16（bfloat16）：1 符号位 + 8 指数位 + 7 尾数位，与 F32 指数范围相同，
 *    只是截断尾数——所以转 F32 只需左移 16 位，这是它成为 LLM 权重主流格式的原因。
 * 2. F16（IEEE half）：1 符号位 + 5 指数位 + 10 尾数位，转换需处理
 *    次正规数（exp=0）与无穷/NaN（exp=31），按位重建 F32。
 * 3. HuggingFace 发布的 Qwen3 权重为 BF16；safetensors 中每个元素占 2 字节 little-endian。
 */
public final class Bf16 {

    private Bf16() {
    }

    /** 单个 BF16（16 位 int）-> F32 */
    public static float bf16ToFloat(int bits) {
        return Float.intBitsToFloat(bits << 16);
    }

    /** 单个 F16（16 位 int）-> F32 */
    public static float f16ToFloat(int h) {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int frac = h & 0x3FF;
        int f32Bits;
        if (exp == 0) {
            // 次正规数：value = frac / 1024 * 2^-14，用算术方式构造
            float v = (float) (frac / 1024.0 * Math.pow(2, -14));
            return sign == 1 ? -v : v;
        } else if (exp == 31) {
            f32Bits = (sign << 31) | 0x7F800000 | (frac << 13); // inf / NaN
        } else {
            // 正规数：exp 偏置 15 -> 127，尾数左移 13 位
            f32Bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
        }
        return Float.intBitsToFloat(f32Bits);
    }

}
