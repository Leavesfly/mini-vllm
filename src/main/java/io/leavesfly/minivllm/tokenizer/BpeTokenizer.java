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

/**
 * BpeTokenizer —— Qwen2/Qwen3 词表的 byte-level BPE 分词器（零依赖手写）。
 *
 * 学习要点：
 * 1. 词表来源：HF 仓库的 vocab.json（token->id，151936 项）与 merges.txt（合并规则）。
 *    token 字符串使用 byte-level unicode 编码（见 {@link ByteLevelBpe}）。
 * 2. encode 流程：special token 切分 -> pre-token 正则切分 -> 每段转 byte-level 字符
 *    -> BPE 合并 -> 查词表得 id。
 * 3. pre-token 正则与 Qwen2/GPT-4 一致：把文本切成"单词/数字段/标点段/空白段"，
 *    BPE 合并不跨越这些段（这是与 HF 结果逐 id 一致的关键）。
 * 4. decode 流程：id -> token 字符串拼接 -> byte-level 字符还原为字节 -> UTF-8 解码。
 *    流式输出用 {@link IncrementalDecoder} 处理跨 token 的 UTF-8 截断。
 */
public final class BpeTokenizer implements SimpleTokenizer {

    /** Qwen2/GPT-4 风格 pre-tokenization 正则 */
    private static final Pattern PRE_TOKEN = Pattern.compile(
            "'(?i:[sdmt]|ll|ve|re)"
                    + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+");

    private final Map<String, Integer> vocab;      // token 字符串 -> id
    private final String[] idToToken;              // id -> token 字符串
    private final Map<String, Integer> mergeRanks; // "A B" -> rank
    private final Map<String, Integer> special;    // special token 字面文本 -> id
    private final Pattern specialPattern;          // special token 切分正则（可能为 null）
    private final int vocabSize;

    private BpeTokenizer(Map<String, Integer> vocab, Map<String, Integer> mergeRanks,
                         Map<String, Integer> special) {
        this.vocab = vocab;
        this.mergeRanks = mergeRanks;
        this.special = special;
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
     * 从 HF 模型目录加载分词器（vocab.json + merges.txt + tokenizer_config.json）。
     * special tokens 从 tokenizer_config.json 的 added_tokens_decoder 读取
     * （Qwen 的 26 个 added token 不在 vocab.json 中）。
     */
    public static BpeTokenizer fromModelDir(Path modelDir) throws IOException {
        Map<String, Integer> special = new HashMap<>();
        Path tokCfg = modelDir.resolve("tokenizer_config.json");
        if (Files.exists(tokCfg)) {
            special.putAll(parseAddedTokens(Files.readString(tokCfg, StandardCharsets.UTF_8)));
        }
        return fromFiles(modelDir.resolve("vocab.json"), modelDir.resolve("merges.txt"), special);
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
        return new BpeTokenizer(vocab, ranks, special);
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

    // ===================== SimpleTokenizer 接口 =====================

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
    public IncrementalDecoder incrementalDecoder() {
        return new IncrementalDecoder(this);
    }

    // ===================== 内部实现 =====================

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
        Matcher m = PRE_TOKEN.matcher(text);
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
        StringBuilder tok = new StringBuilder();
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
            i++;
            tok.setLength(0);
            while (true) {
                char c = s.charAt(i++);
                if (c == '"') {
                    break;
                }
                if (c == '\\') {
                    char esc = s.charAt(i++);
                    switch (esc) {
                        case '"': tok.append('"'); break;
                        case '\\': tok.append('\\'); break;
                        case '/': tok.append('/'); break;
                        case 'n': tok.append('\n'); break;
                        case 'r': tok.append('\r'); break;
                        case 't': tok.append('\t'); break;
                        case 'b': tok.append('\b'); break;
                        case 'f': tok.append('\f'); break;
                        case 'u':
                            tok.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                            i += 4;
                            break;
                        default: tok.append(esc);
                    }
                } else {
                    tok.append(c);
                }
            }
            // 冒号
            while (i < len && (s.charAt(i) == ':' || Character.isWhitespace(s.charAt(i)))) {
                i++;
            }
            // 数值 id
            int id = 0;
            while (i < len && s.charAt(i) >= '0' && s.charAt(i) <= '9') {
                id = id * 10 + (s.charAt(i++) - '0');
            }
            map.put(tok.toString(), id);
        }
        return map;
    }

    /**
     * 流式增量解码器：把逐 token 的字节先缓冲，只输出可完整解码的 UTF-8 文本。
     * 解决多字节字符（中文/emoji）被 token 边界切开时输出乱码（'�'）的问题。
     */
    public static final class IncrementalDecoder {
        private final BpeTokenizer tokenizer;
        private final ByteArrayOutputStream pending = new ByteArrayOutputStream(64);
        private final CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPLACE)
                .onUnmappableCharacter(CodingErrorAction.REPLACE);

        private IncrementalDecoder(BpeTokenizer tokenizer) {
            this.tokenizer = tokenizer;
        }

        /** 喂入一个 token，返回本次可输出的文本（可能为空串） */
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
        public synchronized String flush() {
            byte[] all = pending.toByteArray();
            pending.reset();
            return all.length == 0 ? "" : new String(all, StandardCharsets.UTF_8);
        }
    }
}
