package io.leavesfly.minivllm.json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 极简 JSON 解析与序列化 —— 零依赖下手写，供 SafetensorsLoader 与 API 层使用。
 *
 * 学习要点：
 * 1. 递归下降解析器：根据当前字符分派到 object/array/string/number/bool/null 的解析。
 * 2. 这里只做"够用"的子集：不支持注释、不支持科学计数法尾随逗号等，但覆盖 safetensors header 与 OpenAI 请求/响应所需。
 * 3. 数字统一用 Double 表示，需要 int/long 处由调用方转换。
 */
public final class SimpleJson {

    private SimpleJson() {
    }

    // ===================== 解析 =====================

    public static Object parse(String s) {
        Parser p = new Parser(s);
        p.skipWs();
        Object v = p.parseValue();
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseObject(String s) {
        return (Map<String, Object>) parse(s);
    }

    // ===================== 序列化 =====================

    public static String stringify(Object o) {
        StringBuilder sb = new StringBuilder();
        writeValue(sb, o);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static void writeValue(StringBuilder sb, Object o) {
        if (o == null) {
            sb.append("null");
        } else if (o instanceof String) {
            writeString(sb, (String) o);
        } else if (o instanceof Number || o instanceof Boolean) {
            sb.append(o.toString());
        } else if (o instanceof Map) {
            writeObject(sb, (Map<String, Object>) o);
        } else if (o instanceof List) {
            writeArray(sb, (List<Object>) o);
        } else {
            writeString(sb, o.toString());
        }
    }

    private static void writeObject(StringBuilder sb, Map<String, Object> m) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : m.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            writeString(sb, e.getKey());
            sb.append(':');
            writeValue(sb, e.getValue());
        }
        sb.append('}');
    }

    private static void writeArray(StringBuilder sb, List<Object> a) {
        sb.append('[');
        for (int i = 0; i < a.size(); i++) {
            if (i > 0) sb.append(',');
            writeValue(sb, a.get(i));
        }
        sb.append(']');
    }

    private static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }

    // ===================== 递归下降解析器 =====================

    private static final class Parser {
        private final String s;
        private int pos;

        Parser(String s) {
            this.s = s;
        }

        void skipWs() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        Object parseValue() {
            skipWs();
            char c = s.charAt(pos);
            switch (c) {
                case '{': return parseObject();
                case '[': return parseArray();
                case '"': return parseString();
                case 't': case 'f': return parseBool();
                case 'n': return parseNull();
                default: return parseNumber();
            }
        }

        Map<String, Object> parseObject() {
            Map<String, Object> m = new LinkedHashMap<>();
            pos++; // 跳过 '{'
            skipWs();
            if (s.charAt(pos) == '}') {
                pos++;
                return m;
            }
            while (true) {
                skipWs();
                String key = parseString();
                skipWs();
                if (s.charAt(pos) != ':') throw err("期望 ':'");
                pos++;
                Object val = parseValue();
                m.put(key, val);
                skipWs();
                char c = s.charAt(pos);
                if (c == ',') {
                    pos++;
                    continue;
                }
                if (c == '}') {
                    pos++;
                    break;
                }
                throw err("期望 ',' 或 '}'");
            }
            return m;
        }

        List<Object> parseArray() {
            List<Object> a = new ArrayList<>();
            pos++; // 跳过 '['
            skipWs();
            if (s.charAt(pos) == ']') {
                pos++;
                return a;
            }
            while (true) {
                a.add(parseValue());
                skipWs();
                char c = s.charAt(pos);
                if (c == ',') {
                    pos++;
                    continue;
                }
                if (c == ']') {
                    pos++;
                    break;
                }
                throw err("期望 ',' 或 ']'");
            }
            return a;
        }

        String parseString() {
            if (s.charAt(pos) != '"') throw err("期望字符串");
            pos++;
            StringBuilder sb = new StringBuilder();
            while (pos < s.length()) {
                char c = s.charAt(pos++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c == '\\') {
                    char esc = s.charAt(pos++);
                    switch (esc) {
                        case '"': sb.append('"'); break;
                        case '\\': sb.append('\\'); break;
                        case '/': sb.append('/'); break;
                        case 'n': sb.append('\n'); break;
                        case 'r': sb.append('\r'); break;
                        case 't': sb.append('\t'); break;
                        case 'u':
                            sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
                            pos += 4;
                            break;
                        default: sb.append(esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw err("字符串未闭合");
        }

        Double parseNumber() {
            int start = pos;
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                    pos++;
                } else {
                    break;
                }
            }
            return Double.parseDouble(s.substring(start, pos));
        }

        Boolean parseBool() {
            if (s.startsWith("true", pos)) {
                pos += 4;
                return Boolean.TRUE;
            }
            if (s.startsWith("false", pos)) {
                pos += 5;
                return Boolean.FALSE;
            }
            throw err("非法布尔值");
        }

        Object parseNull() {
            if (s.startsWith("null", pos)) {
                pos += 4;
                return null;
            }
            throw err("非法 null");
        }

        RuntimeException err(String msg) {
            return new RuntimeException("JSON 解析错误 @" + pos + ": " + msg);
        }
    }
}
