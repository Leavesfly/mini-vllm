package io.leavesfly.minivllm.tokenizer;

import io.leavesfly.minivllm.json.SimpleJson;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * ChatMLTemplateTest —— 两个 ChatML 模板与 HF chat_template 的逐字符对齐。
 *
 * 参考数据：src/test/resources/qwen3/chat_template_cases.json，由
 * tools/dump_chat_template_refs.py 用 jinja2 渲染两份真实模板生成（不需要模型权重，可离线跑）。
 *
 * 重点覆盖同为 qwen3 架构、思考模式约定却相反的两处差异：
 *   1. 生成起点：Qwen3 官方不预填 think 开启标记（模型自己生成），MiniMind3 由模板预填
 *   2. 历史 assistant 消息：Qwen3 官方丢弃上一轮思考，MiniMind3 重建 think 块
 */
class ChatMLTemplateTest {

    private static Map<String, Object> cases;

    /** 以下 messages 与 tools/dump_chat_template_refs.py 中的用例一一对应 */
    private static final List<ChatTemplate.Message> SINGLE =
            List.of(new ChatTemplate.Message("user", "你好"));

    private static final List<ChatTemplate.Message> MULTI = List.of(
            new ChatTemplate.Message("user", "问1"),
            // 上一轮的原始输出：开启标记来自模板预填，所以只剩闭合标记
            new ChatTemplate.Message("assistant", "思考A\n</think>\n\n答案A"),
            new ChatTemplate.Message("user", "问2"));

    private static final List<ChatTemplate.Message> PLAIN_HISTORY = List.of(
            new ChatTemplate.Message("user", "问1"),
            new ChatTemplate.Message("assistant", "纯答案"),
            new ChatTemplate.Message("user", "问2"));

    @BeforeAll
    static void loadCases() throws IOException {
        try (InputStream in = ChatMLTemplateTest.class
                .getResourceAsStream("/qwen3/chat_template_cases.json")) {
            assertNotNull(in, "缺少参考数据（先运行 tools/dump_chat_template_refs.py）");
            cases = SimpleJson.parseObject(new String(in.readAllBytes(), StandardCharsets.UTF_8));
        }
    }

    private static String ref(String name) {
        Object expected = cases.get(name);
        assertNotNull(expected, "参考数据缺少用例: " + name);
        return (String) expected;
    }

    // ===================== MiniMind3：思考模式由模板预填开启标记 =====================

    @Test
    void miniMindThinkingPrefillsThinkOpen() {
        assertEquals(ref("mini/single/think"),
                new MiniMind3ChatMLTemplate().render(SINGLE, true),
                "MiniMind3 思考模式必须预填 think 开启标记，否则模型只会吐出闭合标记");
    }

    @Test
    void miniMindNoThinkingPrefillsEmptyThinkBlock() {
        assertEquals(ref("mini/single/nothink"),
                new MiniMind3ChatMLTemplate().render(SINGLE, false));
    }

    @Test
    void miniMindRebuildsHistoryThinkBlock() {
        assertEquals(ref("mini/multi/think"),
                new MiniMind3ChatMLTemplate().render(MULTI, true));
    }

    @Test
    void miniMindKeepsEmptyThinkBlockForPlainHistory() {
        assertEquals(ref("mini/plainhist/think"),
                new MiniMind3ChatMLTemplate().render(PLAIN_HISTORY, true));
    }

    // ===================== Qwen3 官方：思考模式不预填 =====================

    @Test
    void qwen3ThinkingLeavesThinkOpenToModel() {
        assertEquals(ref("qwen/single/think"),
                new Qwen3ChatMLTemplate().render(SINGLE, true));
    }

    @Test
    void qwen3NoThinkingPrefillsEmptyThinkBlock() {
        assertEquals(ref("qwen/single/nothink"),
                new Qwen3ChatMLTemplate().render(SINGLE, false));
    }

    @Test
    void qwen3DropsHistoryReasoning() {
        assertEquals(ref("qwen/multi/think"),
                new Qwen3ChatMLTemplate().render(MULTI, true));
    }

    // ===================== 思考/答案切分 =====================

    @Test
    void splitsReasoningAndAnswerWithoutThinkOpen() {
        ThinkParts parts = ThinkParts.split("思考A\n</think>\n\n答案A");
        assertEquals("思考A", parts.reasoning());
        assertEquals("答案A", parts.answer());
    }

    @Test
    void splitsReasoningAndAnswerWhenModelEmitsThinkOpen() {
        ThinkParts parts = ThinkParts.split("<think>\n思考A\n</think>\n\n答案A");
        assertEquals("思考A", parts.reasoning());
        assertEquals("答案A", parts.answer());
    }

    @Test
    void treatsTextWithoutThinkCloseAsPureAnswer() {
        ThinkParts parts = ThinkParts.split("纯答案");
        assertEquals("", parts.reasoning());
        assertEquals("纯答案", parts.answer());
    }
}
