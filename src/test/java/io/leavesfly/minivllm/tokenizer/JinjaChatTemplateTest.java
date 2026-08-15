package io.leavesfly.minivllm.tokenizer;

import io.leavesfly.minivllm.json.SimpleJson;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * JinjaChatTemplateTest —— jinja 子集解释器的单元测试。
 *
 * 两层验证：
 * 1. 语法特性单测：变量/空白控制/if/for/loop/set/namespace/切片/测试/过滤器/方法/内置函数
 * 2. 真实模板对齐（金标准）：渲染 Qwen3 官方 chat_template，与 HF jinja2 参考输出
 *    （src/test/resources/qwen3/chat_template_cases.json，tools/dump_chat_template_refs.py 生成）
 *    逐字符一致——覆盖 namespace、负步长切片、is string、带参 strip 等几乎全部特性。
 */
class JinjaChatTemplateTest {

    private static String render(String tpl, List<ChatTemplate.Message> messages, boolean thinking) {
        return new JinjaChatTemplate(tpl).render(messages, thinking);
    }

    private static List<ChatTemplate.Message> single(String text) {
        return List.of(new ChatTemplate.Message("user", text));
    }

    // ─── 语法特性 ───

    @Test
    void variableOutputAndWhitespaceControl() {
        assertEquals("你好", render("{{ messages[0].content }}", single("你好"), false));
        // {{- -}} 两侧空白剥离
        assertEquals("abc", render("a  {{- 'b' -}}  c", single("x"), false));
        // {%  %} 语句不输出；语句间文本原样保留
        assertEquals("xy", render("x{% if true %}y{% endif %}", single("x"), false));
    }

    @Test
    void ifElifElse() {
        String tpl = "{% if messages|length > 2 %}many{% elif messages|length == 2 %}two{% else %}one{% endif %}";
        assertEquals("one", render(tpl, single("a"), false));
        assertEquals("two", render(tpl, List.of(
                new ChatTemplate.Message("user", "a"), new ChatTemplate.Message("user", "b")), false));
    }

    @Test
    void forLoopWithLoopVarAndElse() {
        String tpl = "{% for m in messages %}{{ loop.index0 }}:{{ m.role }}{{ ';' if not loop.last }}{% else %}empty{% endfor %}";
        assertEquals("0:user;1:assistant", render(tpl, List.of(
                new ChatTemplate.Message("user", "a"), new ChatTemplate.Message("assistant", "b")), false));
        assertEquals("empty", render(tpl, List.of(), false));
    }

    @Test
    void setAndNamespace() {
        String tpl = "{% set ns = namespace(count=0) %}"
                + "{% for m in messages %}{% set ns.count = ns.count + 1 %}{% endfor %}"
                + "{% set total = ns.count * 2 %}{{ total }}";
        assertEquals("4", render(tpl, List.of(
                new ChatTemplate.Message("user", "a"), new ChatTemplate.Message("user", "b")), false));
    }

    @Test
    void sliceAndNegativeIndex() {
        // [::-1] 反转（Qwen3 模板用它倒序找最近 user 消息）
        String tpl = "{% for m in messages[::-1] %}{{ m.role }};{% endfor %}{{ messages[-1].role }}";
        assertEquals("user;assistant;user;user",
                render(tpl, List.of(new ChatTemplate.Message("user", "1"),
                        new ChatTemplate.Message("assistant", "2"),
                        new ChatTemplate.Message("user", "3")), false));
    }

    @Test
    void testsAndMembership() {
        assertEquals("Y", render("{% if bogus is not defined %}Y{% endif %}", single("a"), false));
        assertEquals("S", render("{% if messages[0].content is string %}S{% endif %}", single("a"), false));
        assertEquals("F", render("{% if enable_thinking is defined and enable_thinking is false %}F{% endif %}",
                single("a"), false));
        assertEquals("in", render("{% if '</think>' in 'a</think>b' %}in{% endif %}", single("a"), false));
        assertEquals("ni", render("{% if 'x' not in 'abc' %}ni{% endif %}", single("a"), false));
    }

    @Test
    void stringMethodsAndFilters() {
        // Qwen3 模板的 reasoning 提取链：split / rstrip / lstrip / strip 带字符参数
        String tpl = "{{ ' 思考\n'.split('考')[0].rstrip('\\n').strip() }}"
                + "|{{ 'a,b'.split(',')|length }}"
                + "|{{ {'k': 1} | tojson }}"
                + "|{{ missing | default('dft') }}";
        assertEquals("思|2|{\"k\": 1}|dft", render(tpl, single("a"), false));
    }

    @Test
    void tojsonEscapesHtmlChars() {
        // jinja2 htmlsafe：' 与 < 转为 unicode 转义；非 ASCII 不转义（ensure_ascii=False）
        assertEquals("{\"s\": \"\\u003c\\u0027中\"}",
                render("{{ {'s': \"<'中\"} | tojson }}", single("a"), false));
    }

    @Test
    void ternaryAndOperators() {
        assertEquals("yes", render("{{ 'yes' if messages|length > 0 else 'no' }}", single("a"), false));
        assertEquals("3", render("{{ 1 + 6 % 4 }}", single("a"), false));
        assertEquals("ab1", render("{{ 'a' ~ 'b' ~ 1 }}", single("a"), false));
    }

    @Test
    void raiseExceptionPropagates() {
        JinjaChatTemplate tpl = new JinjaChatTemplate("{% if messages %}ok{% else %}{{ raise_exception('empty!') }}{% endif %}");
        JinjaChatTemplate.TemplateException e = assertThrows(JinjaChatTemplate.TemplateException.class,
                () -> tpl.render(List.of(), false));
        assertTrue(e.getMessage().contains("empty!"));
    }

    // ─── 金标准：Qwen3 官方模板与 HF 参考逐字符对齐 ───

    @Test
    void qwen3OfficialTemplateAlignsWithHfReference() throws IOException {
        Path modelDir = Path.of("models/Qwen3-0.6B");
        Path casesFile = Path.of("src/test/resources/qwen3/chat_template_cases.json");
        assumeTrue(Files.isDirectory(modelDir) && Files.isRegularFile(casesFile),
                "缺少 Qwen3-0.6B 模型目录或参考用例，跳过");
        Map<String, Object> cases = SimpleJson.parseObject(Files.readString(casesFile));
        JinjaChatTemplate tpl = JinjaChatTemplate.fromModelDir(modelDir);

        List<ChatTemplate.Message> single = single("你好");
        List<ChatTemplate.Message> multi = List.of(
                new ChatTemplate.Message("user", "问1"),
                new ChatTemplate.Message("assistant", "思考A\n</think>\n\n答案A"),
                new ChatTemplate.Message("user", "问2"));

        assertEquals(cases.get("qwen/single/think"), tpl.render(single, true), "qwen/single/think");
        assertEquals(cases.get("qwen/single/nothink"), tpl.render(single, false), "qwen/single/nothink");
        assertEquals(cases.get("qwen/multi/think"), tpl.render(multi, true), "qwen/multi/think");
    }

    // ─── 金标准：Llama-3.2 官方模板与 jinja2 参考逐字符对齐 ───
    // 覆盖 {# 注释 #}、strftime_now、切片赋值（messages = messages[1:]）、
    // 字符串键下标（messages[0]['role']）、trim 过滤器、is mapping/iterable 等特性。

    @Test
    void llama32OfficialTemplateAlignsWithJinja2Reference() throws IOException {
        Path modelDir = Path.of("models/Llama-3.2-1B-Instruct");
        Path casesFile = Path.of("src/test/resources/llama3/chat_template_cases.json");
        assumeTrue(Files.isDirectory(modelDir) && Files.isRegularFile(casesFile),
                "缺少 Llama-3.2 模型目录或参考用例，跳过");
        Map<String, Object> cases = SimpleJson.parseObject(Files.readString(casesFile));
        JinjaChatTemplate tpl = JinjaChatTemplate.fromModelDir(modelDir);

        for (String name : new String[]{"llama/single", "llama/multi"}) {
            @SuppressWarnings("unchecked")
            Map<String, Object> c = (Map<String, Object>) cases.get(name);
            @SuppressWarnings("unchecked")
            List<Map<String, String>> msgs = (List<Map<String, String>>) c.get("messages");
            List<ChatTemplate.Message> messages = msgs.stream()
                    .map(m -> new ChatTemplate.Message(m.get("role"), m.get("content")))
                    .toList();
            // strftime_now 产生当天日期：参考与实现两侧归一化后比对
            assertEquals(normalizeDate((String) c.get("expected")),
                    normalizeDate(tpl.render(messages, false)), name);
        }
    }

    /** "Today Date: 13 Aug 2026" 中的日期部分替换为占位符（消除运行日差异） */
    private static String normalizeDate(String s) {
        return s.replaceAll("Today Date: \\d{2} \\w{3} \\d{4}", "Today Date: <DATE>");
    }
}
