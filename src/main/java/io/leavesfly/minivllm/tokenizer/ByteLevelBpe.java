package io.leavesfly.minivllm.tokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * ByteLevelBpe —— GPT-2 / Qwen2 风格 byte-level BPE 的核心算法。
 *
 * 学习要点：
 * 1. byte-level 编码：把每个 UTF-8 字节映射到一个"可打印 unicode 字符"，
 *    使 BPE 词表只需 256 个基础 token 就能表示任意字节序列（不会出现 UNK）。
 *    映射规则（GPT-2 公开算法）：可打印字节（'!'..'~'、0xA1..0xAC、0xAE..0xFF）
 *    映射为自身字符，其余 68 个字节依次映射到 256+n。
 *    例如空格 0x20 -> 'Ġ'(0x120)，换行 0x0A -> 'Ċ'(0x10A)。
 * 2. BPE 合并：把一段字符序列按 merges 规则的优先级（rank）反复合并，
 *    每轮找出 rank 最低的相邻对并合并其所有出现，直到没有可合并的对。
 * 3. 本类只处理"已转为 byte-level 字符"的字符串，分词入口见 {@link BpeTokenizer}。
 */
final class ByteLevelBpe {

    /** byte(0..255) -> byte-level unicode 字符 */
    static final char[] BYTE_TO_CHAR = new char[256];
    /** byte-level unicode 字符 -> byte（无映射时为 -1）。最大字符值为 256+67=323 */
    private static final int[] CHAR_TO_BYTE = new int[324];

    static {
        boolean[] printable = new boolean[256];
        for (int b = '!'; b <= '~'; b++) printable[b] = true;       // 0x21..0x7E
        for (int b = 0xA1; b <= 0xAC; b++) printable[b] = true;      // ¡..¬
        for (int b = 0xAE; b <= 0xFF; b++) printable[b] = true;      // ®..ÿ
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (printable[b]) {
                BYTE_TO_CHAR[b] = (char) b;
            } else {
                BYTE_TO_CHAR[b] = (char) (256 + n++);
            }
        }
        Arrays.fill(CHAR_TO_BYTE, -1);
        for (int b = 0; b < 256; b++) {
            CHAR_TO_BYTE[BYTE_TO_CHAR[b]] = b;
        }
    }

    private ByteLevelBpe() {
    }

    /** UTF-8 字节数组 -> byte-level 字符序列（每个字节一个字符） */
    static String bytesToChars(byte[] bytes) {
        char[] chars = new char[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            chars[i] = BYTE_TO_CHAR[bytes[i] & 0xFF];
        }
        return new String(chars);
    }

    /** byte-level 字符 -> 原始字节（非法字符返回 -1） */
    static int charToByte(char c) {
        return c < CHAR_TO_BYTE.length ? CHAR_TO_BYTE[c] : -1;
    }

    /**
     * 对一段 byte-level 字符序列执行 BPE 合并。
     *
     * @param chars 已转为 byte-level 字符的字符串（每个字符对应一个原始字节）
     * @param ranks 合并规则表："A B" -> rank（rank 越小优先级越高）
     * @return 合并后的 token 列表（每个元素是一个词表 token 的字符串形式）
     */
    static List<String> bpe(String chars, Map<String, Integer> ranks) {
        if (chars.length() <= 1) {
            List<String> single = new ArrayList<>(1);
            single.add(chars);
            return single;
        }
        // 初始：每个字符一个 token
        List<String> word = new ArrayList<>(chars.length());
        for (int i = 0; i < chars.length(); i++) {
            word.add(chars.substring(i, i + 1));
        }
        while (word.size() > 1) {
            // 找 rank 最低的相邻对
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i < word.size() - 1; i++) {
                Integer r = ranks.get(word.get(i) + " " + word.get(i + 1));
                if (r != null && r < bestRank) {
                    bestRank = r;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) {
                break; // 无可合并对
            }
            // 合并该对的所有出现（标准 BPE：一轮合并同一对的所有位置）
            String a = word.get(bestIdx);
            String b = word.get(bestIdx + 1);
            List<String> merged = new ArrayList<>(word.size());
            for (int i = 0; i < word.size(); i++) {
                if (i < word.size() - 1 && word.get(i).equals(a) && word.get(i + 1).equals(b)) {
                    merged.add(a + b);
                    i++; // 跳过 b
                } else {
                    merged.add(word.get(i));
                }
            }
            word = merged;
        }
        return word;
    }
}
