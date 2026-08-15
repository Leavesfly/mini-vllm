package io.leavesfly.minivllm.tokenizer;

import io.leavesfly.minivllm.json.SimpleJson;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * JinjaChatTemplate —— 直接渲染模型自带的 jinja 对话模板（子集解释器，零依赖）。
 *
 * 学习要点：
 * 1. HF 生态的对话格式权威来源是模型仓库的 chat_template（tokenizer_config.json 字段
 *    或 chat_template.jinja 文件），vLLM/transformers 用 jinja2 渲染它。每接入一个新
 *    模型手写一个模板类不可持续——实现 jinja 子集解释器后，Llama3/Qwen3 等模板
 *    开箱即用，与 HF apply_chat_template 逐字符对齐。
 * 2. 支持的语法子集（按 Qwen3/Llama3 官方模板实测需求划定）：
 *      语句：{{ expr }}（含 {{- -}} 空白控制）、{% if/elif/else %}、
 *            {% for x in iter %}（loop.index0/index/first/last/length + {% else %}）、
 *            {% set name = expr %} / {% set ns.attr = expr %}（namespace 可变属性）
 *      表达式：字面量（字符串/数字/true/false/none）、变量、a[i]、a[::-1] 切片（含负步长）、
 *            a.b 属性、and/or/not、==/!=/&lt;/&lt;=/&gt;/&gt;=、in/not in、+ - ~ * / // %、
 *            三元 cond-expr（a if c else b）、is defined/none/string/number/false/true 等测试、
 *            tojson/length/trim/default 过滤器、strip/split/startswith/items/get 等方法、
 *            namespace()/raise_exception()/strftime_now() 内置函数
 * 3. 环境语义对齐 HF 参考渲染：trim_blocks=False, lstrip_blocks=False（Qwen3/Llama3
 *    模板全程用 {&#37;- -&#37;} 显式控制空白，不依赖环境开关）；tojson 对齐
 *    jinja2 的 htmlsafe 行为（&lt; &gt; &amp; ' 转为 unicode 转义序列）且 ensure_ascii=False。
 * 4. 未定义变量按 ChainableUndefined 处理：取值/属性/下标得 Undefined 单例，
 *    布尔判定为 false、打印为空串、迭代为空——模板里的 tools/tool_calls 等
 *    可选分支自然跳过。
 */
public final class JinjaChatTemplate implements ChatTemplate {

    /** 模板渲染失败（含 raise_exception） */
    public static final class TemplateException extends RuntimeException {
        public TemplateException(String message) {
            super(message);
        }
    }

    /** 未定义值单例（ChainableUndefined 语义） */
    private static final class Undefined {
        static final Undefined INSTANCE = new Undefined();

        private Undefined() {
        }

        @Override
        public String toString() {
            return "";
        }
    }

    private final RootNode root;
    private final String bosToken;
    private final String eosToken;

    public JinjaChatTemplate(String source) {
        this(source, null, null);
    }

    public JinjaChatTemplate(String source, String bosToken, String eosToken) {
        this.root = new Parser(lex(source)).parse();
        this.bosToken = bosToken;
        this.eosToken = eosToken;
    }

    /**
     * 从模型目录加载模板：优先 chat_template.jinja 文件，其次 tokenizer_config.json
     * 的 chat_template 字段；同时读出 bos_token/eos_token 供模板引用。
     */
    public static JinjaChatTemplate fromModelDir(Path modelDir) throws IOException {
        String source = null;
        String bos = null;
        String eos = null;
        Path jinja = modelDir.resolve("chat_template.jinja");
        if (Files.isRegularFile(jinja)) {
            source = Files.readString(jinja);
        }
        Path cfgPath = modelDir.resolve("tokenizer_config.json");
        if (Files.isRegularFile(cfgPath)) {
            Map<String, Object> cfg = SimpleJson.parseObject(Files.readString(cfgPath));
            if (source == null && cfg.get("chat_template") instanceof String s) {
                source = s;
            }
            bos = tokenText(cfg.get("bos_token"));
            eos = tokenText(cfg.get("eos_token"));
        }
        if (source == null) {
            throw new IOException("模型目录缺少对话模板（chat_template.jinja 或 tokenizer_config.json 的 chat_template）: "
                    + modelDir);
        }
        return new JinjaChatTemplate(source, bos, eos);
    }

    /** bos_token/eos_token 字段兼容字符串与 {"content": ...} 对象两种形态 */
    private static String tokenText(Object v) {
        if (v instanceof String s) {
            return s;
        }
        if (v instanceof Map<?, ?> m && m.get("content") instanceof String s) {
            return s;
        }
        return null;
    }

    // ─── ChatTemplate 接口 ───

    @Override
    public String render(List<Message> messages, boolean enableThinking) {
        Map<String, Object> globals = new HashMap<>();
        List<Map<String, Object>> msgMaps = new ArrayList<>(messages.size());
        for (Message m : messages) {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role", m.role());
            map.put("content", m.content());
            msgMaps.add(map);
        }
        globals.put("messages", msgMaps);
        globals.put("add_generation_prompt", true); // 本接口语义：以 assistant 生成起点结尾
        globals.put("enable_thinking", enableThinking);
        // Llama-3.2 模板检测 strftime_now is defined：放入哨兵值即视为已定义（调用走内置函数）
        globals.put("strftime_now", Boolean.TRUE);
        if (bosToken != null) {
            globals.put("bos_token", bosToken);
        }
        if (eosToken != null) {
            globals.put("eos_token", eosToken);
        }
        StringBuilder out = new StringBuilder();
        execNodes(root.children(), new Scope(globals), out);
        return out.toString();
    }

    // ═══════════════ Lexer ═══════════════

    private static final int T_TEXT = 0;
    private static final int T_VAR = 1;   // {{ ... }}
    private static final int T_STMT = 2;  // {% ... %}

    private record LexTok(int kind, String text) {
    }

    /** 切分为 TEXT/VAR/STMT 序列，处理 {{- -}} {&#37;- -&#37;} 空白控制与 {# 注释 #} */
    private static List<LexTok> lex(String src) {
        // jinja2 keep_trailing_newline=False（默认）：模板末尾的一个换行被剥离
        if (src.endsWith("\r\n")) {
            src = src.substring(0, src.length() - 2);
        } else if (src.endsWith("\n")) {
            src = src.substring(0, src.length() - 1);
        }
        List<LexTok> toks = new ArrayList<>();
        boolean stripNext = false; // 上一个 tag 以 -}} / -%} 结尾：本段 TEXT 去首空白
        int i = 0;
        int n = src.length();
        while (i < n) {
            int varAt = src.indexOf("{{", i);
            int stmtAt = src.indexOf("{%", i);
            int cmtAt = src.indexOf("{#", i);
            int tagAt = minPos(minPos(varAt, stmtAt), cmtAt);
            String text = src.substring(i, tagAt < 0 ? n : tagAt);
            if (stripNext) {
                text = lstrip(text);
                stripNext = false;
            }
            if (tagAt < 0) {
                if (!text.isEmpty()) {
                    toks.add(new LexTok(T_TEXT, text));
                }
                break;
            }
            boolean isVar = tagAt == varAt && varAt >= 0;
            int bodyStart = tagAt + 2;
            boolean stripLeft = bodyStart < n && src.charAt(bodyStart) == '-';
            if (stripLeft) {
                bodyStart++;
                text = rstrip(text); // {{- / {%- / {#- 左侧去尾空白
            }
            if (!text.isEmpty()) {
                toks.add(new LexTok(T_TEXT, text));
            }
            String closer = tagAt == cmtAt && cmtAt >= 0 ? "#}" : (isVar ? "}}" : "%}");
            int end = src.indexOf(closer, bodyStart);
            if (end < 0) {
                throw new TemplateException("未闭合的标签: " + src.substring(tagAt, Math.min(n, tagAt + 8)));
            }
            int bodyEnd = end;
            if (bodyEnd > bodyStart && src.charAt(bodyEnd - 1) == '-') {
                bodyEnd--;
                stripNext = true; // -}} / -%} / -#} 右侧去首空白
            }
            if ("#}".equals(closer)) {
                i = end + 2; // 注释：不产出 token
                continue;
            }
            toks.add(new LexTok(isVar ? T_VAR : T_STMT, src.substring(bodyStart, bodyEnd).trim()));
            i = end + 2;
        }
        return toks;
    }

    private static int minPos(int a, int b) {
        if (a < 0) {
            return b;
        }
        return b < 0 ? a : Math.min(a, b);
    }

    private static String lstrip(String s) {
        int i = 0;
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
        return s.substring(i);
    }

    private static String rstrip(String s) {
        int i = s.length();
        while (i > 0 && Character.isWhitespace(s.charAt(i - 1))) {
            i--;
        }
        return s.substring(0, i);
    }

    // ═══════════════ AST ═══════════════

    private sealed interface Node {
    }

    private record TextNode(String text) implements Node {
    }

    private record OutputNode(Expr expr) implements Node {
    }

    private record IfNode(List<Expr> conds, List<List<Node>> bodies) implements Node {
        // conds.size() == bodies.size() - 1（末位为 else 分支）或 == bodies.size()
    }

    private record ForNode(String var, Expr iterable, List<Node> body, List<Node> elseBody) implements Node {
    }

    private record SetNode(String name, String nsAttr, Expr value) implements Node {
        // nsAttr != null 表示 {% set ns.attr = ... %}（name 为 namespace 变量名）
    }

    private sealed interface Expr {
    }

    private record Literal(Object value) implements Expr {
    }

    private record Name(String id) implements Expr {
    }

    private record GetItem(Expr obj, Expr index) implements Expr {
    }

    private record SliceExpr(Expr obj, Expr start, Expr stop, Expr step) implements Expr {
    }

    private record GetAttr(Expr obj, String attr) implements Expr {
    }

    private record BinOp(String op, Expr left, Expr right) implements Expr {
    }

    private record UnOp(String op, Expr expr) implements Expr {
    }

    private record TestExpr(Expr expr, String test, boolean negated) implements Expr {
    }

    private record FilterExpr(Expr expr, String filter, List<Expr> args) implements Expr {
    }

    private record CallExpr(Expr func, List<Expr> args, Map<String, Expr> kwargs) implements Expr {
    }

    private record CondExpr(Expr then, Expr cond, Expr otherwise) implements Expr {
    }

    private record ListLit(List<Expr> items) implements Expr {
    }

    private record DictLit(List<Expr> keys, List<Expr> values) implements Expr {
    }

    // ═══════════════ Parser ═══════════════

    /** 块结构 + 表达式两级解析（表达式为递归下降） */
    private static final class Parser {
        private final List<LexTok> toks;
        private int pos;

        Parser(List<LexTok> toks) {
            this.toks = toks;
        }

        RootNode parse() {
            List<Node> nodes = parseNodes();
            if (pos < toks.size()) {
                throw new TemplateException("意外的标签: " + toks.get(pos).text());
            }
            return new RootNode(nodes);
        }

        /** 解析节点序列；遇到 elif/else/endif/endfor 停（由调用方消费） */
        private List<Node> parseNodes() {
            List<Node> nodes = new ArrayList<>();
            while (pos < toks.size()) {
                LexTok t = toks.get(pos);
                if (t.kind == T_TEXT) {
                    nodes.add(new TextNode(t.text));
                    pos++;
                } else if (t.kind == T_VAR) {
                    nodes.add(new OutputNode(new ExprParser(t.text).parseFull()));
                    pos++;
                } else {
                    String head = headWord(t.text);
                    switch (head) {
                        case "if" -> nodes.add(parseIf(t.text.substring(2).trim()));
                        case "for" -> nodes.add(parseFor(t.text.substring(3).trim()));
                        case "set" -> nodes.add(parseSet(t.text.substring(3).trim()));
                        default -> {
                            return nodes; // elif/else/endif/endfor 等交还调用方
                        }
                    }
                }
            }
            return nodes;
        }

        private List<Node> parseNodesAndConsume(String expectHead) {
            List<Node> nodes = parseNodes();
            if (pos >= toks.size() || !headWord(cur()).equals(expectHead)) {
                throw new TemplateException("缺少 {% " + expectHead + " %}（当前: "
                        + (pos < toks.size() ? cur() : "EOF") + "）");
            }
            pos++;
            return nodes;
        }

        private String cur() {
            return toks.get(pos).text();
        }

        private static String headWord(String stmt) {
            int sp = 0;
            while (sp < stmt.length() && !Character.isWhitespace(stmt.charAt(sp))) {
                sp++;
            }
            return stmt.substring(0, sp);
        }

        /** if 已在队首（text 为条件部分）；消费到 endif 为止 */
        private IfNode parseIf(String firstCond) {
            pos++; // 消费 if
            List<Expr> conds = new ArrayList<>();
            List<List<Node>> bodies = new ArrayList<>();
            conds.add(new ExprParser(firstCond).parseFull());
            bodies.add(parseNodes());
            while (pos < toks.size()) {
                String head = headWord(cur());
                if (head.equals("elif")) {
                    conds.add(new ExprParser(cur().substring(4).trim()).parseFull());
                    pos++;
                    bodies.add(parseNodes());
                } else if (head.equals("else")) {
                    pos++;
                    bodies.add(parseNodesAndConsume("endif"));
                    return new IfNode(conds, bodies);
                } else if (head.equals("endif")) {
                    pos++;
                    return new IfNode(conds, bodies);
                } else {
                    throw new TemplateException("if 块内意外标签: " + cur());
                }
            }
            throw new TemplateException("if 缺少 endif");
        }

        private ForNode parseFor(String header) {
            pos++; // 消费 for
            int inIdx = header.indexOf(" in ");
            if (inIdx < 0) {
                throw new TemplateException("for 语法应为 {% for x in iterable %}: " + header);
            }
            String var = header.substring(0, inIdx).trim();
            if (var.contains(",")) {
                throw new TemplateException("暂不支持 for 多变量解包: " + header);
            }
            Expr iterable = new ExprParser(header.substring(inIdx + 4).trim()).parseFull();
            List<Node> body = parseNodes();
            List<Node> elseBody = List.of();
            if (pos < toks.size() && headWord(cur()).equals("else")) {
                pos++;
                elseBody = parseNodesAndConsume("endfor");
            } else if (pos < toks.size() && headWord(cur()).equals("endfor")) {
                pos++;
            } else {
                throw new TemplateException("for 缺少 endfor");
            }
            return new ForNode(var, iterable, body, elseBody);
        }

        private SetNode parseSet(String body) {
            pos++; // 消费 set
            int eq = body.indexOf('=');
            if (eq < 0) {
                throw new TemplateException("set 语法应为 {% set name = expr %}: " + body);
            }
            String target = body.substring(0, eq).trim();
            Expr value = new ExprParser(body.substring(eq + 1).trim()).parseFull();
            int dot = target.indexOf('.');
            if (dot >= 0) {
                return new SetNode(target.substring(0, dot).trim(), target.substring(dot + 1).trim(), value);
            }
            return new SetNode(target, null, value);
        }
    }

    private record RootNode(List<Node> children) implements Node {
    }

    // ═══════════════ 表达式解析（递归下降） ═══════════════

    /**
     * 优先级（jinja2，自低向高）：or → and → not → 比较(==/!=/&lt;.../in/is) →
     * 加减(+-) → 拼接(~) → 乘除(* / // %) → 一元(-/+) → 后缀(. [] | 调用) → 原子。
     * 三元 cond-expr 在 or 之上再包一层。
     */
    private static final class ExprParser {
        private final List<String> t;
        private int p;

        ExprParser(String src) {
            this.t = tokenize(src);
        }

        Expr parseFull() {
            Expr e = parseCond();
            if (p < t.size()) {
                throw new TemplateException("表达式解析残留: " + String.join(" ", t.subList(p, t.size())));
            }
            return e;
        }

        // 词法：字符串字面量 / 数字 / 名字与关键字 / 运算符
        private static List<String> tokenize(String s) {
            List<String> out = new ArrayList<>();
            int i = 0;
            int n = s.length();
            while (i < n) {
                char c = s.charAt(i);
                if (Character.isWhitespace(c)) {
                    i++;
                } else if (c == '\'' || c == '"') {
                    StringBuilder sb = new StringBuilder();
                    int j = i + 1;
                    while (j < n && s.charAt(j) != c) {
                        if (s.charAt(j) == '\\' && j + 1 < n) {
                            sb.append(unescape(s.charAt(j + 1)));
                            j += 2;
                        } else {
                            sb.append(s.charAt(j++));
                        }
                    }
                    if (j >= n) {
                        throw new TemplateException("未闭合的字符串字面量");
                    }
                    out.add("S:" + sb);
                    i = j + 1;
                } else if (Character.isDigit(c) || (c == '.' && i + 1 < n && Character.isDigit(s.charAt(i + 1)))) {
                    int j = i;
                    while (j < n && (Character.isDigit(s.charAt(j)) || s.charAt(j) == '.')) {
                        j++;
                    }
                    out.add("N:" + s.substring(i, j));
                    i = j;
                } else if (Character.isJavaIdentifierStart(c) || c == '_') {
                    int j = i;
                    while (j < n && (Character.isJavaIdentifierPart(s.charAt(j)) || s.charAt(j) == '_')) {
                        j++;
                    }
                    out.add("I:" + s.substring(i, j));
                    i = j;
                } else {
                    String two = i + 1 < n ? s.substring(i, i + 2) : "";
                    if (two.equals("==") || two.equals("!=") || two.equals("<=") || two.equals(">=")
                            || two.equals("//") || two.equals("**")) {
                        out.add(two);
                        i += 2;
                    } else {
                        out.add(String.valueOf(c));
                        i++;
                    }
                }
            }
            return out;
        }

        private static char unescape(char c) {
            return switch (c) {
                case 'n' -> '\n';
                case 't' -> '\t';
                case 'r' -> '\r';
                case '\\' -> '\\';
                case '\'' -> '\'';
                case '"' -> '"';
                default -> c;
            };
        }

        private String peek() {
            return p < t.size() ? t.get(p) : null;
        }

        private String next() {
            return p < t.size() ? t.get(p++) : null;
        }

        private boolean at(String tok) {
            return tok.equals(peek());
        }

        private boolean atName(String id) {
            return ("I:" + id).equals(peek());
        }

        private boolean eat(String tok) {
            if (at(tok)) {
                p++;
                return true;
            }
            return false;
        }

        private void expect(String tok) {
            if (!eat(tok)) {
                throw new TemplateException("期望 " + tok + " 实际 " + peek());
            }
        }

        // a if cond else b
        private Expr parseCond() {
            Expr then = parseOr();
            if (atName("if")) {
                next();
                Expr cond = parseOr();
                Expr other = null;
                if (atName("else")) {
                    next();
                    other = parseCond();
                }
                return new CondExpr(then, cond, other);
            }
            return then;
        }

        private Expr parseOr() {
            Expr l = parseAnd();
            while (atName("or")) {
                next();
                l = new BinOp("or", l, parseAnd());
            }
            return l;
        }

        private Expr parseAnd() {
            Expr l = parseNot();
            while (atName("and")) {
                next();
                l = new BinOp("and", l, parseNot());
            }
            return l;
        }

        private Expr parseNot() {
            if (atName("not")) {
                next();
                return new UnOp("not", parseNot());
            }
            return parseCompare();
        }

        private Expr parseCompare() {
            Expr l = parseMath1();
            while (true) {
                String tk = peek();
                if (tk != null && switch (tk) {
                    case "==", "!=", "<", "<=", ">", ">=" -> true;
                    default -> false;
                }) {
                    next();
                    l = new BinOp(tk, l, parseMath1());
                } else if (atName("in")) {
                    next();
                    l = new BinOp("in", l, parseMath1());
                } else if (atName("not") && atName("in", 1)) {
                    next();
                    next();
                    l = new BinOp("not in", l, parseMath1());
                } else if (atName("is")) {
                    next();
                    boolean neg = atName("not");
                    if (neg) {
                        next();
                    }
                    String testName = nameToken(next());
                    l = new TestExpr(l, testName, neg);
                } else {
                    return l;
                }
            }
        }

        private boolean atName(String id, int ahead) {
            return p + ahead < t.size() && ("I:" + id).equals(t.get(p + ahead));
        }

        private static String nameToken(String tok) {
            if (tok == null || !tok.startsWith("I:")) {
                throw new TemplateException("期望名字，实际: " + tok);
            }
            return tok.substring(2);
        }

        private Expr parseMath1() {
            Expr l = parseConcat();
            while (at("+") || at("-")) {
                l = new BinOp(next(), l, parseConcat());
            }
            return l;
        }

        private Expr parseConcat() {
            Expr l = parseMath2();
            while (at("~")) {
                next();
                l = new BinOp("~", l, parseMath2());
            }
            return l;
        }

        private Expr parseMath2() {
            Expr l = parseUnary();
            while (at("*") || at("/") || at("//") || at("%")) {
                l = new BinOp(next(), l, parseUnary());
            }
            return l;
        }

        private Expr parseUnary() {
            if (eat("-")) {
                return new UnOp("-", parseUnary());
            }
            if (eat("+")) {
                return parseUnary();
            }
            return parsePostfix(parsePrimary());
        }

        private Expr parsePostfix(Expr e) {
            while (true) {
                if (eat(".")) {
                    String attr = nameToken(next());
                    if (at("(")) {
                        e = new CallExpr(new GetAttr(e, attr), parseCallArgs().positional(), null);
                    } else {
                        e = new GetAttr(e, attr);
                    }
                } else if (eat("[")) {
                    // 下标或切片：a[i] / a[:] / a[::] / a[1:2:3]
                    Expr first = at(":") ? null : parseCond();
                    if (eat(":")) {
                        Expr stop = at("]") || at(":") ? null : parseCond();
                        Expr step = null;
                        if (eat(":")) {
                            step = at("]") ? null : parseCond();
                        }
                        expect("]");
                        e = new SliceExpr(e, first, stop, step);
                    } else {
                        expect("]");
                        e = new GetItem(e, first);
                    }
                } else if (at("|")) {
                    next();
                    String fname = nameToken(next());
                    List<Expr> args = new ArrayList<>();
                    if (at("(")) {
                        args = parseCallArgs().positional();
                    }
                    e = new FilterExpr(e, fname, args);
                } else if (at("(") && e instanceof Name) {
                    CallArgs callArgs = parseCallArgs();
                    e = new CallExpr(e, callArgs.positional(), callArgs.kwargs());
                } else {
                    return e;
                }
            }
        }

        private record CallArgs(List<Expr> positional, Map<String, Expr> kwargs) {
        }

        /** 解析 (a, b, k=v, ...) 实参表：name=expr 形态入 kwargs，其余按位置 */
        private CallArgs parseCallArgs() {
            expect("(");
            List<Expr> positional = new ArrayList<>();
            Map<String, Expr> kwargs = new LinkedHashMap<>();
            while (!at(")")) {
                // 关键字参数判定：名字后紧跟单个 =（== 已在词法层合并为单独 token，不会误伤）
                if (peek() != null && peek().startsWith("I:") && p + 1 < t.size()
                        && t.get(p + 1).equals("=")) {
                    String key = nameToken(next());
                    next(); // '='
                    kwargs.put(key, parseCond());
                } else {
                    positional.add(parseCond());
                }
                if (!eat(",")) {
                    break;
                }
            }
            expect(")");
            return new CallArgs(positional, kwargs);
        }

        private Expr parsePrimary() {
            String tk = next();
            if (tk == null) {
                throw new TemplateException("表达式意外结束");
            }
            if (tk.startsWith("S:")) {
                return new Literal(tk.substring(2));
            }
            if (tk.startsWith("N:")) {
                return new Literal(Double.parseDouble(tk.substring(2)));
            }
            if (tk.equals("(")) {
                Expr e = parseCond();
                expect(")");
                return e;
            }
            if (tk.equals("[")) {
                List<Expr> items = new ArrayList<>();
                while (!at("]")) {
                    items.add(parseCond());
                    if (!eat(",")) {
                        break;
                    }
                }
                expect("]");
                return new ListLit(items);
            }
            if (tk.equals("{")) {
                List<Expr> keys = new ArrayList<>();
                List<Expr> values = new ArrayList<>();
                while (!at("}")) {
                    keys.add(parseCond());
                    expect(":");
                    values.add(parseCond());
                    if (!eat(",")) {
                        break;
                    }
                }
                expect("}");
                return new DictLit(keys, values);
            }
            if (tk.startsWith("I:")) {
                String id = tk.substring(2);
                return switch (id) {
                    case "true", "True" -> new Literal(Boolean.TRUE);
                    case "false", "False" -> new Literal(Boolean.FALSE);
                    case "none", "None", "null" -> new Literal(null);
                    default -> new Name(id);
                };
            }
            throw new TemplateException("意外的 token: " + tk);
        }
    }

    // ═══════════════ 解释执行 ═══════════════

    /** 作用域栈：for 压一层（循环变量与循环内 set 局部），if 不压 */
    private static final class Scope {
        private final Deque<Map<String, Object>> stack = new ArrayDeque<>();

        Scope(Map<String, Object> globals) {
            stack.push(globals);
        }

        Object lookup(String name) {
            for (Map<String, Object> m : stack) {
                if (m.containsKey(name)) {
                    return m.get(name);
                }
            }
            return Undefined.INSTANCE;
        }

        void set(String name, Object value) {
            stack.peek().put(name, value);
        }

        void push() {
            stack.push(new HashMap<>());
        }

        void pop() {
            stack.pop();
        }
    }

    private void execNodes(List<Node> nodes, Scope scope, StringBuilder out) {
        for (Node node : nodes) {
            if (node instanceof TextNode t) {
                out.append(t.text());
            } else if (node instanceof OutputNode o) {
                out.append(stringify(eval(o.expr(), scope)));
            } else if (node instanceof IfNode ifn) {
                List<Expr> conds = ifn.conds();
                List<List<Node>> bodies = ifn.bodies();
                boolean done = false;
                for (int i = 0; i < conds.size(); i++) {
                    if (truthy(eval(conds.get(i), scope))) {
                        execNodes(bodies.get(i), scope, out);
                        done = true;
                        break;
                    }
                }
                if (!done && bodies.size() > conds.size()) {
                    execNodes(bodies.get(bodies.size() - 1), scope, out);
                }
            } else if (node instanceof ForNode f) {
                List<Object> items = iterableOf(eval(f.iterable(), scope));
                if (items.isEmpty()) {
                    execNodes(f.elseBody(), scope, out);
                } else {
                    scope.push();
                    try {
                        int len = items.size();
                        for (int i = 0; i < len; i++) {
                            scope.set(f.var(), items.get(i));
                            Map<String, Object> loop = new HashMap<>();
                            loop.put("index0", (double) i);
                            loop.put("index", (double) (i + 1));
                            loop.put("first", i == 0);
                            loop.put("last", i == len - 1);
                            loop.put("length", (double) len);
                            scope.set("loop", loop);
                            execNodes(f.body(), scope, out);
                        }
                    } finally {
                        scope.pop();
                    }
                }
            } else if (node instanceof SetNode s) {
                Object value = eval(s.value(), scope);
                if (s.nsAttr() == null) {
                    scope.set(s.name(), value);
                } else {
                    Object ns = scope.lookup(s.name());
                    if (!(ns instanceof Map)) {
                        throw new TemplateException(s.name() + " 不是 namespace 对象");
                    }
                    @SuppressWarnings("unchecked")
                    Map<String, Object> nsMap = (Map<String, Object>) ns;
                    nsMap.put(s.nsAttr(), value);
                }
            } else if (node instanceof RootNode r) {
                execNodes(r.children(), scope, out);
            }
        }
    }

    private Object eval(Expr e, Scope scope) {
        if (e instanceof Literal lit) {
            return lit.value();
        }
        if (e instanceof Name name) {
            return scope.lookup(name.id());
        }
        if (e instanceof GetItem gi) {
            return getItem(eval(gi.obj(), scope), eval(gi.index(), scope));
        }
        if (e instanceof SliceExpr sl) {
            return slice(eval(sl.obj(), scope),
                    sl.start() == null ? null : eval(sl.start(), scope),
                    sl.stop() == null ? null : eval(sl.stop(), scope),
                    sl.step() == null ? null : eval(sl.step(), scope));
        }
        if (e instanceof GetAttr ga) {
            return getAttr(eval(ga.obj(), scope), ga.attr());
        }
        if (e instanceof BinOp b) {
            return evalBinOp(b, scope);
        }
        if (e instanceof UnOp u) {
            Object v = eval(u.expr(), scope);
            return switch (u.op()) {
                case "not" -> !truthy(v);
                case "-" -> -num(v);
                default -> throw new TemplateException("未知一元运算: " + u.op());
            };
        }
        if (e instanceof TestExpr te) {
            return applyTest(eval(te.expr(), scope), te.test(), te.negated());
        }
        if (e instanceof FilterExpr fe) {
            return applyFilter(eval(fe.expr(), scope), fe.filter(), evalArgs(fe.args(), scope));
        }
        if (e instanceof CallExpr ce) {
            return evalCall(ce, scope);
        }
        if (e instanceof CondExpr cond) {
            return truthy(eval(cond.cond(), scope))
                    ? eval(cond.then(), scope)
                    : cond.otherwise() == null ? Undefined.INSTANCE : eval(cond.otherwise(), scope);
        }
        if (e instanceof ListLit ll) {
            List<Object> items = new ArrayList<>(ll.items().size());
            for (Expr item : ll.items()) {
                items.add(eval(item, scope));
            }
            return items;
        }
        if (e instanceof DictLit dl) {
            Map<String, Object> map = new LinkedHashMap<>();
            for (int i = 0; i < dl.keys().size(); i++) {
                map.put(stringify(eval(dl.keys().get(i), scope)), eval(dl.values().get(i), scope));
            }
            return map;
        }
        throw new TemplateException("未知表达式节点: " + e.getClass().getSimpleName());
    }

    private List<Object> evalArgs(List<Expr> args, Scope scope) {
        List<Object> out = new ArrayList<>(args.size());
        for (Expr a : args) {
            out.add(eval(a, scope));
        }
        return out;
    }

    private Object evalBinOp(BinOp b, Scope scope) {
        // and/or 短路，返回操作数（python 语义）
        if (b.op().equals("and")) {
            Object l = eval(b.left(), scope);
            return truthy(l) ? eval(b.right(), scope) : l;
        }
        if (b.op().equals("or")) {
            Object l = eval(b.left(), scope);
            return truthy(l) ? l : eval(b.right(), scope);
        }
        Object l = eval(b.left(), scope);
        Object r = eval(b.right(), scope);
        return switch (b.op()) {
            case "==" -> valuesEqual(l, r);
            case "!=" -> !valuesEqual(l, r);
            case "<" -> compare(l, r) < 0;
            case "<=" -> compare(l, r) <= 0;
            case ">" -> compare(l, r) > 0;
            case ">=" -> compare(l, r) >= 0;
            case "in" -> contains(r, l);
            case "not in" -> !contains(r, l);
            case "+" -> plus(l, r);
            case "-" -> num(l) - num(r);
            case "*" -> num(l) * num(r);
            case "/" -> num(l) / num(r);
            case "//" -> Math.floor(num(l) / num(r));
            case "%" -> num(l) % num(r);
            case "~" -> stringify(l) + stringify(r);
            default -> throw new TemplateException("未知二元运算: " + b.op());
        };
    }

    private Object evalCall(CallExpr ce, Scope scope) {
        if (ce.func() instanceof Name name) {
            List<Expr> argExprs = ce.args();
            switch (name.id()) {
                case "namespace" -> {
                    // namespace(k=v, ...)：实参在 parser 中以 kwargs 收集（见 parseKwargs）
                    Map<String, Object> ns = new HashMap<>();
                    if (ce.kwargs() != null) {
                        for (Map.Entry<String, Expr> en : ce.kwargs().entrySet()) {
                            ns.put(en.getKey(), eval(en.getValue(), scope));
                        }
                    }
                    return ns;
                }
                case "raise_exception" -> throw new TemplateException(
                        stringify(eval(argExprs.get(0), scope)));
                case "strftime_now" -> {
                    String fmt = stringify(eval(argExprs.get(0), scope));
                    return strftime(fmt);
                }
                case "loop" -> throw new TemplateException("loop(...) 递归引用暂不支持");
                default -> throw new TemplateException("未支持的函数: " + name.id());
            }
        }
        if (ce.func() instanceof GetAttr ga) {
            // 方法调用：obj.attr(args)
            Object recv = eval(ga.obj(), scope);
            return callMethod(recv, ga.attr(), evalArgs(ce.args(), scope));
        }
        throw new TemplateException("不可调用的表达式");
    }

    // ─── 值操作语义 ───

    private static boolean truthy(Object v) {
        if (v == null || v == Undefined.INSTANCE) {
            return false;
        }
        if (v instanceof Boolean b) {
            return b;
        }
        if (v instanceof Number num) {
            return num.doubleValue() != 0;
        }
        if (v instanceof String s) {
            return !s.isEmpty();
        }
        if (v instanceof List<?> l) {
            return !l.isEmpty();
        }
        if (v instanceof Map<?, ?> m) {
            return !m.isEmpty();
        }
        return true;
    }

    private static double num(Object v) {
        if (v instanceof Number num) {
            return num.doubleValue();
        }
        if (v instanceof Boolean b) {
            return b ? 1 : 0;
        }
        throw new TemplateException("期望数字，实际: " + v);
    }

    private static String stringify(Object v) {
        if (v == null) {
            return "None";
        }
        if (v == Undefined.INSTANCE) {
            return "";
        }
        if (v instanceof Double d) {
            long lv = d.longValue();
            return d == lv ? Long.toString(lv) : d.toString();
        }
        if (v instanceof Boolean b) {
            return b ? "True" : "False"; // python 风格
        }
        if (v instanceof List || v instanceof Map) {
            return toJson(v); // 近似：模板不会打印复合结构（真需要时用 tojson）
        }
        return v.toString();
    }

    private static boolean valuesEqual(Object l, Object r) {
        if (l == Undefined.INSTANCE && r == Undefined.INSTANCE) {
            return true;
        }
        if (l == Undefined.INSTANCE || r == Undefined.INSTANCE) {
            return false;
        }
        if (l == null || r == null) {
            return l == null && r == null;
        }
        if (l instanceof Number a && r instanceof Number b) {
            return a.doubleValue() == b.doubleValue();
        }
        return l.equals(r);
    }

    private static int compare(Object l, Object r) {
        if (l instanceof Number a && r instanceof Number b) {
            return Double.compare(a.doubleValue(), b.doubleValue());
        }
        if (l instanceof String a && r instanceof String b) {
            return a.compareTo(b);
        }
        throw new TemplateException("不可比较: " + l + " vs " + r);
    }

    private static boolean contains(Object container, Object item) {
        if (container instanceof String s && item instanceof String sub) {
            return s.contains(sub);
        }
        if (container instanceof List<?> l) {
            for (Object o : l) {
                if (valuesEqual(o, item)) {
                    return true;
                }
            }
            return false;
        }
        if (container instanceof Map<?, ?> m && item instanceof String k) {
            return m.containsKey(k);
        }
        return false;
    }

    private static Object plus(Object l, Object r) {
        if (l instanceof String a && r instanceof String b) {
            return a + b;
        }
        if (l instanceof List<?> a && r instanceof List<?> b) {
            List<Object> out = new ArrayList<>(a);
            out.addAll(b);
            return out;
        }
        return num(l) + num(r);
    }

    private static Object getItem(Object obj, Object index) {
        if (obj == Undefined.INSTANCE) {
            return Undefined.INSTANCE;
        }
        // Map 的键可以是字符串（messages[0]['role']），必须先于数字索引判断
        if (obj instanceof Map<?, ?> m) {
            return m.containsKey(index) ? m.get(index) : Undefined.INSTANCE;
        }
        int i = (int) num(index);
        if (obj instanceof List<?> l) {
            int idx = i < 0 ? l.size() + i : i;
            return idx >= 0 && idx < l.size() ? l.get(idx) : Undefined.INSTANCE;
        }
        if (obj instanceof String s) {
            int idx = i < 0 ? s.length() + i : i;
            return idx >= 0 && idx < s.length() ? String.valueOf(s.charAt(idx)) : Undefined.INSTANCE;
        }
        throw new TemplateException("不支持下标访问: " + obj);
    }

    /** 切片：String 与 List 均支持 [start:stop:step]（含负索引与负步长） */
    private static Object slice(Object obj, Object startV, Object stopV, Object stepV) {
        int step = stepV == null ? 1 : (int) num(stepV);
        if (step == 0) {
            throw new TemplateException("切片步长不能为 0");
        }
        if (obj instanceof String s) {
            return sliceList(new ArrayList<>(s.codePoints().boxed().map(cp -> (Object) cp).toList()),
                    startV, stopV, step, true);
        }
        if (obj instanceof List<?> l) {
            return sliceList(new ArrayList<>(l), startV, stopV, step, false);
        }
        throw new TemplateException("不支持切片: " + obj);
    }

    private static Object sliceList(List<Object> items, Object startV, Object stopV, int step, boolean asString) {
        int n = items.size();
        int start = startV == null ? (step > 0 ? 0 : n - 1) : clampIdx((int) num(startV), n, step);
        // stop 为不含边界：step<0 缺省取 -1（循环条件 i > -1 即含索引 0，对齐 python [::-1]）；
        // 显式负值经 clampIdx 转 n+i（如 -1 → n-1，条件 i > n-1 即不含末元素）
        int stop = stopV == null ? (step > 0 ? n : -1) : clampIdx((int) num(stopV), n, step);
        StringBuilder sb = asString ? new StringBuilder() : null;
        List<Object> out = asString ? null : new ArrayList<>();
        for (int i = start; step > 0 ? i < stop : i > stop; i += step) {
            if (asString) {
                sb.appendCodePoint((Integer) items.get(i));
            } else {
                out.add(items.get(i));
            }
        }
        return asString ? sb.toString() : out;
    }

    private static int clampIdx(int i, int n, int step) {
        int idx = i < 0 ? n + i : i;
        if (step > 0) {
            return Math.max(0, Math.min(n, idx));
        }
        return Math.max(-1, Math.min(n - 1, idx));
    }

    private static Object getAttr(Object obj, String attr) {
        if (obj == Undefined.INSTANCE) {
            return Undefined.INSTANCE; // chainable
        }
        if (obj instanceof Map<?, ?> m) {
            return m.containsKey(attr) ? m.get(attr) : Undefined.INSTANCE;
        }
        throw new TemplateException("不支持属性访问 ." + attr + " on " + obj);
    }

    private static List<Object> iterableOf(Object v) {
        if (v == Undefined.INSTANCE || v == null) {
            return List.of();
        }
        if (v instanceof List<?> l) {
            return new ArrayList<>(l);
        }
        if (v instanceof Map<?, ?> m) {
            return new ArrayList<>(m.keySet()); // jinja 迭代 Map 得 keys
        }
        if (v instanceof String s) {
            List<Object> out = new ArrayList<>();
            s.codePoints().forEach(cp -> out.add(new String(Character.toChars(cp))));
            return out;
        }
        return List.of();
    }

    // ─── 测试 / 过滤器 / 方法 ───

    private static Object applyTest(Object v, String test, boolean negated) {
        boolean result = switch (test) {
            case "defined" -> v != Undefined.INSTANCE;
            case "undefined" -> v == Undefined.INSTANCE;
            case "none" -> v == null;
            case "string" -> v instanceof String;
            case "number" -> v instanceof Number;
            case "boolean" -> v instanceof Boolean;
            case "true" -> Boolean.TRUE.equals(v);
            case "false" -> Boolean.FALSE.equals(v);
            case "iterable" -> v instanceof List || v instanceof Map || v instanceof String;
            case "mapping" -> v instanceof Map;
            case "sequence" -> v instanceof List || v instanceof String;
            case "even" -> v instanceof Number num && num.doubleValue() % 2 == 0;
            case "odd" -> v instanceof Number num && num.doubleValue() % 2 != 0;
            default -> throw new TemplateException("未支持的测试: is " + test);
        };
        return negated != result;
    }

    private static Object applyFilter(Object v, String filter, List<Object> args) {
        return switch (filter) {
            case "tojson" -> toJson(v);
            case "length", "count" -> v == Undefined.INSTANCE ? 0.0
                    : v instanceof String s ? (double) s.codePointCount(0, s.length())
                    : v instanceof List<?> l ? (double) l.size()
                    : v instanceof Map<?, ?> m ? (double) m.size() : 0.0;
            case "trim" -> stringify(v).strip();
            case "lower" -> stringify(v).toLowerCase();
            case "upper" -> stringify(v).toUpperCase();
            case "default", "d" -> v == Undefined.INSTANCE ? (args.isEmpty() ? "" : args.get(0)) : v;
            case "list" -> iterableOf(v);
            case "first" -> iterableOf(v).isEmpty() ? Undefined.INSTANCE : iterableOf(v).get(0);
            case "last" -> {
                List<Object> items = iterableOf(v);
                yield items.isEmpty() ? Undefined.INSTANCE : items.get(items.size() - 1);
            }
            case "join" -> {
                String sep = args.isEmpty() ? "" : stringify(args.get(0));
                StringBuilder sb = new StringBuilder();
                List<Object> items = iterableOf(v);
                for (int i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        sb.append(sep);
                    }
                    sb.append(stringify(items.get(i)));
                }
                yield sb.toString();
            }
            default -> throw new TemplateException("未支持的过滤器: " + filter);
        };
    }

    private static Object callMethod(Object recv, String name, List<Object> args) {
        if (recv == Undefined.INSTANCE) {
            return Undefined.INSTANCE;
        }
        if (recv instanceof String s) {
            return switch (name) {
                case "strip" -> args.isEmpty() ? s.strip() : stripChars(s, stringify(args.get(0)), true, true);
                case "lstrip" -> args.isEmpty() ? lstrip(s) : stripChars(s, stringify(args.get(0)), true, false);
                case "rstrip" -> args.isEmpty() ? rstrip(s) : stripChars(s, stringify(args.get(0)), false, true);
                case "startswith" -> s.startsWith(stringify(args.get(0)));
                case "endswith" -> s.endsWith(stringify(args.get(0)));
                case "lower" -> s.toLowerCase();
                case "upper" -> s.toUpperCase();
                case "split" -> {
                    if (args.isEmpty()) {
                        yield new ArrayList<>(List.of(s.strip().split("\\s+")));
                    }
                    String sep = stringify(args.get(0));
                    List<Object> parts = new ArrayList<>();
                    for (String part : s.split(java.util.regex.Pattern.quote(sep), -1)) {
                        parts.add(part);
                    }
                    yield parts;
                }
                case "join" -> {
                    StringBuilder sb = new StringBuilder();
                    List<Object> items = iterableOf(args.get(0));
                    for (int i = 0; i < items.size(); i++) {
                        if (i > 0) {
                            sb.append(s);
                        }
                        sb.append(stringify(items.get(i)));
                    }
                    yield sb.toString();
                }
                case "replace" -> s.replace(stringify(args.get(0)), stringify(args.get(1)));
                case "capitalize" -> s.isEmpty() ? s
                        : s.substring(0, 1).toUpperCase() + s.substring(1).toLowerCase();
                case "title" -> java.util.Arrays.stream(s.split(" "))
                        .map(wd -> wd.isEmpty() ? wd : wd.substring(0, 1).toUpperCase() + wd.substring(1))
                        .reduce((a, b) -> a + " " + b).orElse("");
                default -> throw new TemplateException("字符串不支持方法: " + name);
            };
        }
        if (recv instanceof Map<?, ?> m) {
            return switch (name) {
                case "get" -> m.containsKey(args.get(0)) ? m.get(args.get(0))
                        : (args.size() > 1 ? args.get(1) : null);
                case "items" -> {
                    List<Object> pairs = new ArrayList<>();
                    for (Map.Entry<?, ?> en : m.entrySet()) {
                        pairs.add(List.of(en.getKey(), en.getValue()));
                    }
                    yield pairs;
                }
                case "keys" -> new ArrayList<>(m.keySet());
                case "values" -> new ArrayList<>(m.values());
                default -> throw new TemplateException("Map 不支持方法: " + name);
            };
        }
        throw new TemplateException("不支持方法调用 ." + name + " on " + recv);
    }

    private static String stripChars(String s, String chars, boolean left, boolean right) {
        int start = 0;
        int end = s.length();
        if (left) {
            while (start < end && chars.indexOf(s.charAt(start)) >= 0) {
                start++;
            }
        }
        if (right) {
            while (end > start && chars.indexOf(s.charAt(end - 1)) >= 0) {
                end--;
            }
        }
        return s.substring(start, end);
    }

    // ─── 内置工具 ───

    /** tojson：对齐 jinja2 htmlsafe（< > & ' 转义）+ ensure_ascii=False */
    private static String toJson(Object v) {
        StringBuilder sb = new StringBuilder();
        writeJson(v, sb);
        return sb.toString();
    }

    private static void writeJson(Object v, StringBuilder sb) {
        if (v == null || v == Undefined.INSTANCE) {
            sb.append("null");
        } else if (v instanceof String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\t' -> sb.append("\\t");
                    case '\r' -> sb.append("\\r");
                    case '<' -> sb.append("\\u003c");
                    case '>' -> sb.append("\\u003e");
                    case '&' -> sb.append("\\u0026");
                    case '\'' -> sb.append("\\u0027");
                    default -> sb.append(c);
                }
            }
            sb.append('"');
        } else if (v instanceof Boolean b) {
            sb.append(b ? "true" : "false");
        } else if (v instanceof Number num) {
            double d = num.doubleValue();
            long lv = (long) d;
            sb.append(d == lv ? Long.toString(lv) : Double.toString(d));
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                writeJson(l.get(i), sb);
            }
            sb.append(']');
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> en : m.entrySet()) {
                if (!first) {
                    sb.append(", ");
                }
                first = false;
                writeJson(String.valueOf(en.getKey()), sb);
                sb.append(": ");
                writeJson(en.getValue(), sb);
            }
            sb.append('}');
        } else {
            writeJson(v.toString(), sb);
        }
    }

    /** strftime_now 的格式子集：%Y %m %d %H %M %S %b（英文月缩写，Llama-3.2 模板用） */
    private static String strftime(String fmt) {
        LocalDateTime now = LocalDateTime.now();
        String out = fmt.replace("%Y", String.format("%04d", now.getYear()))
                .replace("%m", String.format("%02d", now.getMonthValue()))
                .replace("%d", String.format("%02d", now.getDayOfMonth()))
                .replace("%H", String.format("%02d", now.getHour()))
                .replace("%M", String.format("%02d", now.getMinute()))
                .replace("%S", String.format("%02d", now.getSecond()))
                .replace("%b", MONTH_ABBR[now.getMonthValue() - 1]);
        return out;
    }

    private static final String[] MONTH_ABBR = {
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
}
