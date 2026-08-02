package io.leavesfly.minivllm.tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CoderResult;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * BpeTokenizer —— Qwen2/Qwen3 词表的 byte-level BPE 分词器（零依赖手写）。
 *
 * 学习要点：
 * 1. 词表来源：HF 仓库的 vocab.json（token->id，151936 项）与 merges.txt（合并规则），
 *    或把两者合并在一起的单文件 tokenizer.json（新版 HF tokenizers 的默认导出格式）。
 *    token 字符串使用 byte-level unicode 编码（见 {@link ByteLevelBpe}）。
 * 2. encode 流程：special token 切分 -> pre-token 正则切分 -> 每段转 byte-level 字符
 *    -> BPE 合并 -> 查词表得 id。
 * 3. pre-token 正则把文本切成"单词/数字段/标点段/空白段"，BPE 合并不跨越这些段
 *    （这是与 HF 结果逐 id 一致的关键）。具体规则随模型而异，不能写死：
 *    Qwen 系在 tokenizer.json 里显式声明 Split 正则（数字按 1~3 位分组），
 *    GPT-2 系（如 MiniMind3）用 ByteLevel 预分词器的内置正则（数字不限长合并）。
 * 4. decode 流程：id -> token 字符串拼接 -> byte-level 字符还原为字节 -> UTF-8 解码。
 *    流式输出用 {@link IncrementalDecoder} 处理跨 token 的 UTF-8 截断。
 */
public final class BpeTokenizer implements SimpleTokenizer {

    /** Qwen2/GPT-4 风格 pre-tokenization 正则（未声明时的默认规则） */
    private static final Pattern PRE_TOKEN = Pattern.compile(
            "'(?i:[sdmt]|ll|ve|re)"
                    + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+");

    /**
     * GPT-2 风格 pre-tokenization 正则：HF ByteLevel 预分词器 use_regex=true 时的内置规则。
     * 与 Qwen2 版的关键区别：数字用 " ?\p{N}+" 不限长分组（而非 {1,3}），
     * 因此 "1234567890" 会作为单个 pre-token 参与 BPE 合并。
     */
    private static final Pattern PRE_TOKEN_GPT2 = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d"
                    + "| ?\\p{L}+"
                    + "| ?\\p{N}+"
                    + "| ?[^\\s\\p{L}\\p{N}]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+");

    private final Map<String, Integer> vocab;      // token 字符串 -> id
    private final String[] idToToken;              // id -> token 字符串
    private final Map<String, Integer> mergeRanks; // "A B" -> rank
    private final Map<String, Integer> special;    // special token 字面文本 -> id
    private final Pattern specialPattern;          // special token 切分正则（可能为 null）
    private final Pattern preToken;                // pre-token 切分正则
    private final int vocabSize;

    private BpeTokenizer(Map<String, Integer> vocab, Map<String, Integer> mergeRanks,
                         Map<String, Integer> special, Pattern preToken) {
        this.vocab = vocab;
        this.mergeRanks = mergeRanks;
        this.special = special;
        this.preToken = preToken;
        int maxId = 0;
        for (int id : vocab.values()) {
            maxId = Math.max(maxId, id);
        }
        this.vocabSize = maxId + 1;
        this.idToToken = new String[vocabSize];
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            idToToken[e.getValue()] = e.getKey();
        }
        this.specialPattern = buildSpecialPattern(special);
    }

    /**
     * 从 HF 模型目录加载分词器，兼容两种词表布局：
     *   vocab.json + merges.txt（Qwen3-0.6B 等老式导出）
     *   tokenizer.json（单文件导出，如 MiniMind3；无 vocab.json/merges.txt 时使用）
     * special tokens 从 tokenizer_config.json 的 added_tokens_decoder 读取
     * （Qwen 的 26 个 added token 不在 vocab.json 中）。
     */
    public static BpeTokenizer fromModelDir(Path modelDir) throws IOException {
        Map<String, Integer> special = new HashMap<>();
        Path tokCfg = modelDir.resolve("tokenizer_config.json");
        if (Files.exists(tokCfg)) {
            special.putAll(parseAddedTokens(Files.readString(tokCfg, StandardCharsets.UTF_8)));
        }
        Path vocabJson = modelDir.resolve("vocab.json");
        Path mergesTxt = modelDir.resolve("merges.txt");
        if (Files.isRegularFile(vocabJson) && Files.isRegularFile(mergesTxt)) {
            return fromFiles(vocabJson, mergesTxt, special);
        }
        Path tokenizerJson = modelDir.resolve("tokenizer.json");
        if (Files.isRegularFile(tokenizerJson)) {
            return fromTokenizerJson(tokenizerJson, special);
        }
        throw new IOException("模型目录缺少分词器文件（需 vocab.json + merges.txt 或 tokenizer.json）: "
                + modelDir);
    }

    /**
     * 从单文件 tokenizer.json 加载：词表在 model.vocab，合并规则在 model.merges。
     * merges 有两种形态，均需支持：新版为 ["A","B"] 数组对，旧版为 "A B" 字符串。
     */
    public static BpeTokenizer fromTokenizerJson(Path tokenizerJson,
                                                 Map<String, Integer> extraSpecial) throws IOException {
        Map<String, Object> root = io.leavesfly.minivllm.json.SimpleJson.parseObject(
                Files.readString(tokenizerJson, StandardCharsets.UTF_8));
        if (!(root.get("model") instanceof Map<?, ?> model)) {
            throw new IOException("tokenizer.json 缺少 model 段: " + tokenizerJson);
        }
        if (!(model.get("vocab") instanceof Map<?, ?> vocabJson)) {
            throw new IOException("tokenizer.json 缺少 model.vocab: " + tokenizerJson);
        }
        Map<String, Integer> vocab = new HashMap<>(vocabJson.size() * 2);
        for (Map.Entry<?, ?> e : vocabJson.entrySet()) {
            vocab.put(String.valueOf(e.getKey()), ((Number) e.getValue()).intValue());
        }

        Map<String, Integer> ranks = new HashMap<>(1 << 18);
        if (model.get("merges") instanceof List<?> merges) {
            int rank = 0;
            for (Object m : merges) {
                // 整行 "A B" 作为 key，与 ByteLevelBpe 查询格式一致
                String key = m instanceof List<?> pair && pair.size() == 2
                        ? pair.get(0) + " " + pair.get(1)
                        : String.valueOf(m);
                ranks.put(key, rank++);
            }
        }

        // added_tokens 是 tokenizer.json 自带的 special 列表（tokenizer_config.json 缺失时的来源）
        Map<String, Integer> special = new HashMap<>(extraSpecial);
        if (root.get("added_tokens") instanceof List<?> added) {
            for (Object a : added) {
                if (a instanceof Map<?, ?> t && t.get("content") != null && t.get("id") instanceof Number id) {
                    special.putIfAbsent(String.valueOf(t.get("content")), id.intValue());
                }
            }
        }
        return build(vocab, ranks, special, resolvePreTokenPattern(root.get("pre_tokenizer")));
    }

    /**
     * 从 tokenizer.json 的 pre_tokenizer 段得出 pre-token 切分正则。
     * 两种已知形态：
     *   Sequence[Split(Regex), ByteLevel(use_regex=false)] —— Qwen 系，直接用声明的正则
     *   ByteLevel(use_regex=true)                         —— GPT-2 系，用其内置正则
     */
    private static Pattern resolvePreTokenPattern(Object preTokenizer) {
        String declared = findSplitRegex(preTokenizer);
        if (declared != null) {
            try {
                return Pattern.compile(declared);
            } catch (PatternSyntaxException e) {
                // Rust regex 语法与 Java 不完全重叠，无法编译时退回默认规则
                System.out.println("pre_tokenizer 正则无法在 Java 下编译，回退 Qwen2 规则: "
                        + e.getDescription());
                return PRE_TOKEN;
            }
        }
        return hasRegexByteLevel(preTokenizer) ? PRE_TOKEN_GPT2 : PRE_TOKEN;
    }

    /** 递归查找 {"type":"Split","pattern":{"Regex":...}} 中声明的正则，没有则返回 null */
    private static String findSplitRegex(Object node) {
        if (node instanceof Map<?, ?> m) {
            if ("Split".equals(m.get("type")) && m.get("pattern") instanceof Map<?, ?> p
                    && p.get("Regex") instanceof String regex) {
                return regex;
            }
            for (Object v : m.values()) {
                String found = findSplitRegex(v);
                if (found != null) {
                    return found;
                }
            }
        } else if (node instanceof List<?> list) {
            for (Object v : list) {
                String found = findSplitRegex(v);
                if (found != null) {
                    return found;
                }
            }
        }
        return null;
    }

    /** 是否存在开启内置正则的 ByteLevel 预分词器（use_regex 缺省为 true） */
    private static boolean hasRegexByteLevel(Object node) {
        if (node instanceof Map<?, ?> m) {
            if ("ByteLevel".equals(m.get("type")) && !Boolean.FALSE.equals(m.get("use_regex"))) {
                return true;
            }
            for (Object v : m.values()) {
                if (hasRegexByteLevel(v)) {
                    return true;
                }
            }
        } else if (node instanceof List<?> list) {
            for (Object v : list) {
                if (hasRegexByteLevel(v)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 从 vocab.json + merges.txt 加载分词器（无 special token 表时使用）。
     */
    public static BpeTokenizer fromFiles(Path vocabJson, Path mergesTxt) throws IOException {
        return fromFiles(vocabJson, mergesTxt, new HashMap<>());
    }

    /**
     * 从 vocab.json + merges.txt + 外部 special token 表加载分词器。
     */
    public static BpeTokenizer fromFiles(Path vocabJson, Path mergesTxt,
                                         Map<String, Integer> extraSpecial) throws IOException {
        String vocabStr = Files.readString(vocabJson, StandardCharsets.UTF_8);
        Map<String, Integer> vocab = parseVocabJson(vocabStr);

        Map<String, Integer> ranks = new HashMap<>(1 << 18);
        List<String> lines = Files.readAllLines(mergesTxt, StandardCharsets.UTF_8);
        int rank = 0;
        for (String line : lines) {
            if (line.isEmpty() || line.startsWith("#")) {
                continue; // 跳过 "#version: 0.2" 头
            }
            int sp = line.indexOf(' ');
            if (sp <= 0 || sp == line.length() - 1) {
                continue;
            }
            ranks.put(line, rank++); // 整行 "A B" 作为 key，与 ByteLevelBpe 查询格式一致
        }
        return build(vocab, ranks, extraSpecial, PRE_TOKEN);
    }

    /** 汇总 special token 表并构造实例（vocab.json 与 tokenizer.json 两条路径共用） */
    private static BpeTokenizer build(Map<String, Integer> vocab, Map<String, Integer> ranks,
                                      Map<String, Integer> extraSpecial, Pattern preToken) {
        // special token：词表中所有 <|...|> 形式的 token + 外部 added_tokens
        Map<String, Integer> special = new HashMap<>(extraSpecial);
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            String t = e.getKey();
            if (t.startsWith("<|") && t.endsWith("|>")) {
                special.putIfAbsent(t, e.getValue());
            }
        }
        // added token 也要能从 id 反查（decode 用）
        for (Map.Entry<String, Integer> e : special.entrySet()) {
            vocab.putIfAbsent(e.getKey(), e.getValue());
        }
        return new BpeTokenizer(vocab, ranks, special, preToken);
    }

    /** 解析 tokenizer_config.json 的 added_tokens_decoder：{"151643": {"content": "<|endoftext|>"}, ...} */
    @SuppressWarnings("unchecked")
    private static Map<String, Integer> parseAddedTokens(String json) {
        Map<String, Integer> out = new HashMap<>();
        Map<String, Object> cfg = io.leavesfly.minivllm.json.SimpleJson.parseObject(json);
        Object atd = cfg.get("added_tokens_decoder");
        if (!(atd instanceof Map)) {
            return out;
        }
        for (Map.Entry<String, Object> e : ((Map<String, Object>) atd).entrySet()) {
            Object content = ((Map<String, Object>) e.getValue()).get("content");
            if (content != null) {
                out.put(content.toString(), Integer.parseInt(e.getKey()));
            }
        }
        return out;
    }

    // ─── SimpleTokenizer 接口 ───

    @Override
    public int[] encode(String text) {
        List<Integer> out = new ArrayList<>();
        encodeInto(text, out);
        int[] ids = new int[out.size()];
        for (int i = 0; i < out.size(); i++) {
            ids[i] = out.get(i);
        }
        return ids;
    }

    @Override
    public String decode(int[] ids) {
        StringBuilder chars = new StringBuilder();
        for (int id : ids) {
            String t = id >= 0 && id < idToToken.length ? idToToken[id] : null;
            if (t != null) {
                chars.append(t);
            }
        }
        byte[] bytes = new byte[chars.length()];
        int n = 0;
        for (int i = 0; i < chars.length(); i++) {
            int b = ByteLevelBpe.charToByte(chars.charAt(i));
            if (b >= 0) {
                bytes[n++] = (byte) b;
            }
        }
        return new String(bytes, 0, n, StandardCharsets.UTF_8);
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }

    /** special token id 查询（如 "<|im_end|>" -> 151645），不存在返回 -1 */
    public int specialId(String token) {
        return special.getOrDefault(token, -1);
    }

    /** 创建流式增量解码器：逐 token 喂入，只返回可完整解码的文本片段 */
    @Override
    public IncrementalDecoder incrementalDecoder() {
        return new BpeIncrementalDecoder(this);
    }

    // ─── 内部实现 ───

    private void encodeInto(String text, List<Integer> out) {
        if (text == null || text.isEmpty()) {
            return;
        }
        if (specialPattern == null) {
            encodeOrdinary(text, out);
            return;
        }
        // 先按 special token 字面文本切分，special 直接映射 id，其余走 BPE
        Matcher m = specialPattern.matcher(text);
        int last = 0;
        while (m.find()) {
            if (m.start() > last) {
                encodeOrdinary(text.substring(last, m.start()), out);
            }
            out.add(special.get(m.group()));
            last = m.end();
        }
        if (last < text.length()) {
            encodeOrdinary(text.substring(last), out);
        }
    }

    private void encodeOrdinary(String text, List<Integer> out) {
        Matcher m = preToken.matcher(text);
        while (m.find()) {
            String piece = m.group();
            String chars = ByteLevelBpe.bytesToChars(piece.getBytes(StandardCharsets.UTF_8));
            for (String token : ByteLevelBpe.bpe(chars, mergeRanks)) {
                Integer id = vocab.get(token);
                if (id == null) {
                    throw new IllegalStateException("BPE 产物不在词表中: " + escape(token));
                }
                out.add(id);
            }
        }
    }

    private static Pattern buildSpecialPattern(Map<String, Integer> special) {
        if (special.isEmpty()) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (String t : special.keySet()) {
            if (sb.length() > 0) {
                sb.append('|');
            }
            sb.append(Pattern.quote(t));
        }
        return Pattern.compile(sb.toString());
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r");
    }

    /**
     * vocab.json 专用解析器：格式为扁平的 {"token":id,"token":id,...}。
     * 相比通用 JSON 解析，避免 15 万个 Double 装箱，速度快一个数量级。
     */
    private static Map<String, Integer> parseVocabJson(String s) throws IOException {
        Map<String, Integer> map = new HashMap<>(1 << 18);
        int i = 0;
        int len = s.length();
        while (i < len && s.charAt(i) != '{') {
            i++;
        }
        i++; // 跳过 '{'
        while (i < len) {
            // 跳过空白与逗号
            while (i < len && (s.charAt(i) == ',' || Character.isWhitespace(s.charAt(i)))) {
                i++;
            }
            if (i >= len || s.charAt(i) == '}') {
                break;
            }
            // 解析 key 字符串
            if (s.charAt(i) != '"') {
                throw new IOException("vocab.json 格式错误 @" + i);
            }
            i++; // 跳过开头引号
            StringBuilder tok = new StringBuilder();
            i = parseJsonString(s, i, tok);
            // 跳过冒号与空白
            while (i < len && (s.charAt(i) == ':' || Character.isWhitespace(s.charAt(i)))) {
                i++;
            }
            // 解析数值 id
            int id = 0;
            while (i < len && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                id = id * 10 + (s.charAt(i++) - '0');
            }
            map.put(tok.toString(), id);
        }
        return map;
    }

    /** 从位置 i 开始解析 JSON 字符串内容（处理转义），结果写入 out，返回结束引号后的位置 */
    private static int parseJsonString(String s, int i, StringBuilder out) {
        while (true) {
            char c = s.charAt(i++);
            if (c == '"') {
                return i;
            }
            if (c == '\\') {
                char esc = s.charAt(i++);
                switch (esc) {
                    case '"': out.append('"'); break;
                    case '\\': out.append('\\'); break;
                    case '/': out.append('/'); break;
                    case 'n': out.append('\n'); break;
                    case 'r': out.append('\r'); break;
                    case 't': out.append('\t'); break;
                    case 'b': out.append('\b'); break;
                    case 'f': out.append('\f'); break;
                    case 'u':
                        out.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default: out.append(esc);
                }
            } else {
                out.append(c);
            }
        }
    }

    /**
     * 流式增量解码器的 BPE 实现：把逐 token 的字节先缓冲，只输出可完整解码的 UTF-8 文本。
     * 解决多字节字符（中文/emoji）被 token 边界切开时输出乱码（'�'）的问题。
     */
    private static final class BpeIncrementalDecoder implements IncrementalDecoder {
        private final BpeTokenizer tokenizer;
        private final ByteArrayOutputStream pending = new ByteArrayOutputStream(64);
        private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPLACE)
                .onUnmappableCharacter(CodingErrorAction.REPLACE);

        private BpeIncrementalDecoder(BpeTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /** 喂入一个 token，返回本次可输出的文本（可能为空串） */
        @Override
        public synchronized String accept(int tokenId) {
            String t = tokenId >= 0 && tokenId < tokenizer.idToToken.length
                    ? tokenizer.idToToken[tokenId] : null;
            if (t == null) {
                return "";
            }
            for (int i = 0; i < t.length(); i++) {
                int b = ByteLevelBpe.charToByte(t.charAt(i));
                if (b >= 0) {
                    pending.write(b);
                }
            }
            byte[] all = pending.toByteArray();
            ByteBuffer in = ByteBuffer.wrap(all);
            CharBuffer out = CharBuffer.allocate(all.length + 1);
            String text;
            int consumed;
            try {
                CoderResult r = decoder.decode(in, out, false);
                consumed = in.position();
                out.flip();
                text = out.toString();
                if (r.isError()) {
                    r.throwException();
                }
            } catch (CharacterCodingException e) {
                // 理论上 REPLACE 模式不会到这里；兜底全部输出
                consumed = all.length;
                text = new String(all, StandardCharsets.UTF_8);
            }
            pending.reset();
            pending.write(all, consumed, all.length - consumed);
            return text;
        }

        /** 冲刷剩余字节（生成结束时调用） */
        @Override
        public synchronized String flush() {
            byte[] all = pending.toByteArray();
            pending.reset();
            return all.length == 0 ? "" : new String(all, StandardCharsets.UTF_8);
        }
    }
}
