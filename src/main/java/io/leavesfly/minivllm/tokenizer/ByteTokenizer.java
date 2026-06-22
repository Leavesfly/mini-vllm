package io.leavesfly.minivllm.tokenizer;

import java.nio.charset.StandardCharsets;

/**
 * 字节级分词器 —— 每个 UTF-8 字节作为一个 token，词表固定 256。
 *
 * 学习要点：
 * 1. 这是"字节级"分词的最简形式：任意文本都能编码，不会出现未知字符。
 * 2. GPT-2 的 byte-level BPE 在此基础上再做合并，词表扩到 50257；这里不做合并，纯学习用。
 * 3. 词表 256 正好与 ModelConfig 默认 vocabSize=256 对齐，配合随机初始化模型即可跑通。
 */
public final class ByteTokenizer implements SimpleTokenizer {

    @Override
    public int[] encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            ids[i] = bytes[i] & 0xFF; // byte→无符号 0..255
        }
        return ids;
    }

    @Override
    public String decode(int[] ids) {
        byte[] bytes = new byte[ids.length];
        for (int i = 0; i < ids.length; i++) {
            bytes[i] = (byte) ids[i];
        }
        return new String(bytes, StandardCharsets.UTF_8);
    }

    @Override
    public int vocabSize() {
        return 256;
    }
}
