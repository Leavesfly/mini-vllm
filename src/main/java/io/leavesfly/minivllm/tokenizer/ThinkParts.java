package io.leavesfly.minivllm.tokenizer;

/**
 * ThinkParts —— 把 assistant 的原始输出切成「思考」与「答案」两段。
 *
 * 学习要点：
 * 1. 思考型模型一轮输出的形态是「思考内容 + 闭合标记 + 正式答案」，开启标记可能来自
 *    模型自己（Qwen3 官方），也可能是模板预填的、并不出现在输出里（MiniMind3）。
 *    所以判据只能取闭合标记 </think>，不能依赖开启标记存在。
 * 2. 切分规则与 HF chat_template.jinja 的 split 语义一致：思考取首个闭合标记之前，
 *    答案取末个闭合标记之后；不含闭合标记时视为纯答案。
 * 3. 各模型模板对历史消息的处理不同（Qwen3 官方丢弃历史思考、MiniMind3 重建 think 块），
 *    但切分这一步是共用的，因此收拢在这里。
 *
 * @param reasoning 思考内容（已剥离首尾换行）
 * @param answer    正式答案（已剥离开头换行）
 */
record ThinkParts(String reasoning, String answer) {

    private static final String THINK_OPEN = Qwen3ChatMLTemplate.THINK_OPEN;
    private static final String THINK_CLOSE = Qwen3ChatMLTemplate.THINK_CLOSE;

    static ThinkParts split(String content) {
        int firstClose = content.indexOf(THINK_CLOSE);
        if (firstClose < 0) {
            return new ThinkParts("", stripLeadingNewlines(content));
        }
        String head = content.substring(0, firstClose);
        int lastOpen = head.lastIndexOf(THINK_OPEN);
        if (lastOpen >= 0) {
            head = head.substring(lastOpen + THINK_OPEN.length());
        }
        String tail = content.substring(content.lastIndexOf(THINK_CLOSE) + THINK_CLOSE.length());
        return new ThinkParts(stripNewlines(head), stripLeadingNewlines(tail));
    }

    /** 只剥离首尾换行（对齐 jinja 的 strip 换行语义，不动空格等其他空白） */
    private static String stripNewlines(String s) {
        int begin = 0;
        int end = s.length();
        while (begin < end && s.charAt(begin) == '\n') {
            begin++;
        }
        while (end > begin && s.charAt(end - 1) == '\n') {
            end--;
        }
        return s.substring(begin, end);
    }

    /** 只剥离开头换行（对齐 jinja 的 lstrip 换行语义） */
    private static String stripLeadingNewlines(String s) {
        int begin = 0;
        while (begin < s.length() && s.charAt(begin) == '\n') {
            begin++;
        }
        return s.substring(begin);
    }
}
