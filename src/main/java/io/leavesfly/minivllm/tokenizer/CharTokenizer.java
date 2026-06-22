package io.leavesfly.minivllm.tokenizer;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * 字符级分词器 —— 基于给定字符表，每个字符一个 token。
 *
 * 学习要点：
 * 1. 适合 char-level 微模型（如 nanoGPT 训练的 Shakespeare 小模型），词表=字符表大小。
 * 2. 构造时传入训练语料的全部唯一字符，建立 char→id 与 id→char 映射。
 * 3. 遇到字符表外的字符会报错——真实场景需做 fallback（如映射到 <unk>）。
 */
public final class CharTokenizer implements SimpleTokenizer {

    private final String chars;
    private final Map<Character, Integer> charToId;
    private final int vocabSize;

    public CharTokenizer(String chars) {
        this.chars = chars;
        this.vocabSize = chars.length();
        this.charToId = new HashMap<>();
        for (int i = 0; i < chars.length(); i++) {
            charToId.putIfAbsent(chars.charAt(i), i);
        }
    }

    @Override
    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            Integer id = charToId.get(text.charAt(i));
            if (id == null) {
                throw new IllegalArgumentException("未知字符: '" + text.charAt(i) + "'");
            }
            ids[i] = id;
        }
        return ids;
    }

    @Override
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder(ids.length);
        for (int id : ids) {
            if (id < 0 || id >= vocabSize) {
                throw new IllegalArgumentException("非法 token id: " + id);
            }
            sb.append(chars.charAt(id));
        }
        return sb.toString();
    }

    @Override
    public int vocabSize() {
        return vocabSize;
    }

    /** 从语料构建字符表（保留首次出现顺序） */
    public static CharTokenizer fromCorpus(String corpus) {
        Set<Character> seen = new LinkedHashSet<>();
        for (int i = 0; i < corpus.length(); i++) {
            seen.add(corpus.charAt(i));
        }
        StringBuilder sb = new StringBuilder(seen.size());
        for (char c : seen) sb.append(c);
        return new CharTokenizer(sb.toString());
    }
}
