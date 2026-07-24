package io.leavesfly.minivllm.tokenizer;

import io.leavesfly.minivllm.json.SimpleJson;
import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * BpeTokenizer 对齐测试 —— 与 HF Qwen3-0.6B tokenizer 输出逐 id 对比。
 *
 * 参考数据：src/test/resources/qwen3/tokenizer_cases.txt（由 tools/dump_tokenizer_refs.py 生成）。
 * 词表文件：默认取 HF 缓存 ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/&lt;hash&gt;/，
 * 可用 -Dqwen3.dir=/path/to/model 指定；找不到词表时跳过（不影响离线 CI）。
 */
class BpeTokenizerTest {

    private static Path resolveModelDir() {
        String prop = System.getProperty("qwen3.dir");
        if (prop != null) {
            return Path.of(prop);
        }
        Path snapshots = Path.of(System.getProperty("user.home"),
                ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        if (Files.isDirectory(snapshots)) {
            try (var stream = Files.list(snapshots)) {
                return stream.filter(Files::isDirectory).findFirst().orElse(null);
            } catch (IOException e) {
                return null;
            }
        }
        return null;
    }

    private static BpeTokenizer loadTokenizer() throws IOException {
        Path dir = resolveModelDir();
        assumeTrue(dir != null && Files.exists(dir.resolve("vocab.json")),
                "未找到 Qwen3-0.6B 词表，跳过对齐测试");
        return BpeTokenizer.fromModelDir(dir);
    }

    // ===================== 与 HF 逐 id 对齐 =====================

    @Test
    void alignWithHuggingFace() throws IOException {
        BpeTokenizer tok = loadTokenizer();
        // 词表 151643 个 BPE token + 26 个 added token（151643..151668）
        assertEquals(151669, tok.vocabSize());

        int cases = 0;
        try (BufferedReader reader = resourceReader("/qwen3/tokenizer_cases.txt")) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                Map<String, Object> c = SimpleJson.parseObject(line);
                String text = (String) c.get("text");
                @SuppressWarnings("unchecked")
                List<Object> idObjs = (List<Object>) c.get("ids");
                int[] expect = new int[idObjs.size()];
                for (int i = 0; i < expect.length; i++) {
                    expect[i] = ((Number) idObjs.get(i)).intValue();
                }
                int[] actual = tok.encode(text);
                assertArrayEquals(expect, actual, "encode 不一致: " + abbrev(text));
                cases++;
            }
        }
        assertTrue(cases >= 16, "参考用例数量不足: " + cases);
    }

    // ===================== round-trip =====================

    @Test
    void roundTrip() throws IOException {
        BpeTokenizer tok = loadTokenizer();
        String[] samples = {
                "Hello, world!",
                "你好，世界！",
                "emoji 😀🎉 test",
                "def main():\n    print('hi')",
                "  spaces  and\ttabs\nnewlines",
                "日本語のテキスト",
        };
        for (String s : samples) {
            assertEquals(s, tok.decode(tok.encode(s)), "round-trip 失败: " + abbrev(s));
        }
    }

    // ===================== special tokens =====================

    @Test
    void specialTokens() throws IOException {
        BpeTokenizer tok = loadTokenizer();
        assertEquals(151643, tok.specialId("<|endoftext|>"));
        assertEquals(151644, tok.specialId("<|im_start|>"));
        assertEquals(151645, tok.specialId("<|im_end|>"));
        // special token 不进 BPE，直接映射 id
        int[] ids = tok.encode("<|im_start|>user\nhi<|im_end|>");
        assertEquals(151644, ids[0]);
        assertEquals(151645, ids[ids.length - 1]);
    }

    // ===================== 增量解码（跨 token UTF-8 边界） =====================

    @Test
    void incrementalDecoder() throws IOException {
        BpeTokenizer tok = loadTokenizer();
        String text = "你好，世界！emoji 😀";
        int[] ids = tok.encode(text);
        IncrementalDecoder dec = tok.incrementalDecoder();
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(dec.accept(id));
        }
        sb.append(dec.flush());
        assertEquals(text, sb.toString());
    }

    // ===================== byte-level 映射表（不依赖词表，始终运行） =====================

    @Test
    void byteLevelTables() {
        // 可打印字节映射为自身
        assertEquals('A', ByteLevelBpe.BYTE_TO_CHAR['A']);
        assertEquals('~', ByteLevelBpe.BYTE_TO_CHAR['~']);
        // 空格 0x20 -> Ġ(0x120)，换行 0x0A -> Ċ(0x10A)（GPT-2 公开映射）
        assertEquals('\u0120', ByteLevelBpe.BYTE_TO_CHAR[0x20]);
        assertEquals('\u010A', ByteLevelBpe.BYTE_TO_CHAR[0x0A]);
        // 反查表
        assertEquals(0x20, ByteLevelBpe.charToByte('\u0120'));
        assertEquals('A', ByteLevelBpe.charToByte('A'));
        // 全部 256 个字节往返一致
        for (int b = 0; b < 256; b++) {
            assertEquals(b, ByteLevelBpe.charToByte(ByteLevelBpe.BYTE_TO_CHAR[b]));
        }
    }

    // ===================== 工具 =====================

    private static BufferedReader resourceReader(String name) {
        InputStream in = BpeTokenizerTest.class.getResourceAsStream(name);
        assertNotNull(in, "缺少测试资源: " + name);
        return new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
    }

    private static String abbrev(String s) {
        String t = s.replace("\n", "\\n");
        return t.length() <= 40 ? t : t.substring(0, 40) + "...";
    }
}
