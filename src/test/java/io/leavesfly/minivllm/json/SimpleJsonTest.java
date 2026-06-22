package io.leavesfly.minivllm.json;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SimpleJson 单元测试 —— 验证 JSON 解析与序列化。
 */
class SimpleJsonTest {

    // ========== 解析测试 ==========

    @Test
    void parseString() {
        Object result = SimpleJson.parse("\"hello\"");
        assertEquals("hello", result);
    }

    @Test
    void parseStringWithEscapes() {
        Object result = SimpleJson.parse("\"line1\\nline2\\ttab\\\"quote\\\"\"");
        assertEquals("line1\nline2\ttab\"quote\"", result);
    }

    @Test
    void parseNumber() {
        Object result = SimpleJson.parse("42");
        assertEquals(42.0, result);
    }

    @Test
    void parseNegativeNumber() {
        Object result = SimpleJson.parse("-3.14");
        assertEquals(-3.14, (Double) result, 1e-10);
    }

    @Test
    void parseBooleanTrue() {
        assertEquals(Boolean.TRUE, SimpleJson.parse("true"));
    }

    @Test
    void parseBooleanFalse() {
        assertEquals(Boolean.FALSE, SimpleJson.parse("false"));
    }

    @Test
    void parseNull() {
        assertNull(SimpleJson.parse("null"));
    }

    @Test
    void parseEmptyObject() {
        Map<String, Object> m = SimpleJson.parseObject("{}");
        assertTrue(m.isEmpty());
    }

    @Test
    void parseSimpleObject() {
        String json = "{\"name\":\"mini-vllm\",\"version\":1}";
        Map<String, Object> m = SimpleJson.parseObject(json);
        assertEquals("mini-vllm", m.get("name"));
        assertEquals(1.0, m.get("version"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void parseNestedObject() {
        String json = "{\"model\":\"gpt\",\"config\":{\"layers\":2,\"dim\":64}}";
        Map<String, Object> m = SimpleJson.parseObject(json);
        assertEquals("gpt", m.get("model"));
        Map<String, Object> config = (Map<String, Object>) m.get("config");
        assertEquals(2.0, config.get("layers"));
        assertEquals(64.0, config.get("dim"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void parseArray() {
        Object result = SimpleJson.parse("[1,2,3]");
        List<Object> arr = (List<Object>) result;
        assertEquals(3, arr.size());
        assertEquals(1.0, arr.get(0));
        assertEquals(3.0, arr.get(2));
    }

    @Test
    @SuppressWarnings("unchecked")
    void parseOpenAiRequest() {
        String json = "{\"model\":\"mini-vllm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true,\"max_tokens\":16}";
        Map<String, Object> m = SimpleJson.parseObject(json);
        assertEquals("mini-vllm", m.get("model"));
        assertEquals(Boolean.TRUE, m.get("stream"));
        assertEquals(16.0, m.get("max_tokens"));
        List<Object> messages = (List<Object>) m.get("messages");
        assertEquals(1, messages.size());
        Map<String, Object> msg = (Map<String, Object>) messages.get(0);
        assertEquals("user", msg.get("role"));
        assertEquals("Hello", msg.get("content"));
    }

    @Test
    void parseWithWhitespace() {
        String json = "  { \"key\" : \"value\" , \"num\" : 42 }  ";
        Map<String, Object> m = SimpleJson.parseObject(json);
        assertEquals("value", m.get("key"));
        assertEquals(42.0, m.get("num"));
    }

    // ========== 序列化测试 ==========

    @Test
    void stringifyNull() {
        assertEquals("null", SimpleJson.stringify(null));
    }

    @Test
    void stringifyString() {
        assertEquals("\"hello\"", SimpleJson.stringify("hello"));
    }

    @Test
    void stringifyStringWithSpecialChars() {
        String result = SimpleJson.stringify("a\"b\nc");
        assertEquals("\"a\\\"b\\nc\"", result);
    }

    @Test
    void stringifyNumber() {
        assertEquals("42", SimpleJson.stringify(42));
        assertEquals("3.14", SimpleJson.stringify(3.14));
    }

    @Test
    void stringifyBoolean() {
        assertEquals("true", SimpleJson.stringify(true));
        assertEquals("false", SimpleJson.stringify(false));
    }

    @Test
    void stringifyMap() {
        Map<String, Object> m = Map.of("key", "value");
        String json = SimpleJson.stringify(m);
        assertTrue(json.contains("\"key\""));
        assertTrue(json.contains("\"value\""));
    }

    @Test
    void stringifyList() {
        List<Object> list = List.of(1, "two", true);
        String json = SimpleJson.stringify(list);
        assertEquals("[1,\"two\",true]", json);
    }

    // ========== 往返测试 ==========

    @Test
    @SuppressWarnings("unchecked")
    void roundTripObject() {
        String original = "{\"model\":\"test\",\"stream\":false}";
        Map<String, Object> parsed = SimpleJson.parseObject(original);
        String serialized = SimpleJson.stringify(parsed);
        Map<String, Object> reparsed = SimpleJson.parseObject(serialized);
        assertEquals(parsed.get("model"), reparsed.get("model"));
        assertEquals(parsed.get("stream"), reparsed.get("stream"));
    }

    @Test
    void parseUnicodeEscape() {
        String json = "\"\\u0041\""; // \u0041 = 'A'
        assertEquals("A", SimpleJson.parse(json));
    }
}
