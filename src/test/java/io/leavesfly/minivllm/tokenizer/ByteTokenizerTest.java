package io.leavesfly.minivllm.tokenizer;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ByteTokenizer 单元测试 —— 验证字节级分词的编解码。
 */
class ByteTokenizerTest {

    private final ByteTokenizer tokenizer = new ByteTokenizer();

    @Test
    void vocabSizeIs256() {
        assertEquals(256, tokenizer.vocabSize());
    }

    @Test
    void encodeAscii() {
        int[] ids = tokenizer.encode("Hi");
        assertArrayEquals(new int[]{'H', 'i'}, ids);
    }

    @Test
    void decodeAscii() {
        int[] ids = {'H', 'e', 'l', 'l', 'o'};
        assertEquals("Hello", tokenizer.decode(ids));
    }

    @Test
    void encodeDecodeRoundTrip() {
        String text = "Hello, World! 123";
        int[] ids = tokenizer.encode(text);
        String decoded = tokenizer.decode(ids);
        assertEquals(text, decoded);
    }

    @Test
    void encodeUtf8MultiByteChar() {
        // 中文字符需多字节 UTF-8 编码
        String text = "你";
        int[] ids = tokenizer.encode(text);
        // "你" = UTF-8 0xE4 0xBD 0xA0 → 3 bytes
        assertEquals(3, ids.length);
        // 每个 id 在 0-255 范围内
        for (int id : ids) {
            assertTrue(id >= 0 && id < 256);
        }
    }

    @Test
    void encodeDecodeUtf8RoundTrip() {
        String text = "你好世界";
        int[] ids = tokenizer.encode(text);
        String decoded = tokenizer.decode(ids);
        assertEquals(text, decoded);
    }

    @Test
    void emptyStringEncodesToEmpty() {
        int[] ids = tokenizer.encode("");
        assertEquals(0, ids.length);
    }

    @Test
    void encodeAllByteValues() {
        // 确保 0-255 都是有效 token id
        for (int b = 0; b < 256; b++) {
            int[] ids = {b};
            String s = tokenizer.decode(ids);
            assertNotNull(s);
        }
    }
}
