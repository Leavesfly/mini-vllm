package io.leavesfly.minivllm.tokenizer;

import java.util.List;

/**
 * MiniMind3ChatMLTemplate —— MiniMind3 系列的 ChatML 对话模板实现。
 *
 * 学习要点：
 * 1. MiniMind3 与 Qwen3 同为 qwen3 架构、同用 ChatML 骨架，但**思考模式的约定相反**：
 *      Qwen3 官方：思考模式下模板不预填，由模型自己生成 <think> ... </think>
 *      MiniMind3：思考模式下模板预填 <think>\n，模型直接从思考内容写起，写完自己吐 </think>
 *    因此不能共用 {@link Qwen3ChatMLTemplate}：少了 <think>\n 前缀，模型仍会按训练习惯
 *    直接输出思考内容并补一个 </think>，产生"有闭合无开启"的畸形输出。
 * 2. 历史 assistant 消息必须按训练格式重建 think 块（开启标记 + 思考 + 闭合标记 + 答案）。
 *    上一轮的原始输出里 </think> 之前是思考、之后是答案，原样拼回去会让第二轮起的
 *    提示词偏离训练分布，放大重复与错乱。
 * 3. 关闭思考模式时预填一个空 think 块，与 Qwen3 官方写法一致。
 *
 * 以上格式与模型目录下 chat_template.jinja 严格对齐（add_generation_prompt=true）。
 */
public final class MiniMind3ChatMLTemplate implements ChatTemplate {

    /** ChatML 骨架标记与思考标记复用 Qwen3 的定义（两者完全相同） */
    private static final String IM_START = Qwen3ChatMLTemplate.IM_START;
    private static final String IM_END = Qwen3ChatMLTemplate.IM_END;
    private static final String THINK_OPEN = Qwen3ChatMLTemplate.THINK_OPEN;
    private static final String THINK_CLOSE = Qwen3ChatMLTemplate.THINK_CLOSE;
    private static final String ASSISTANT = "assistant";

    /**
     * 渲染 messages 为 ChatML 提示词（以 assistant 生成起点结尾）。
     *
     * @param enableThinking true 时末尾预填 <think>\n，模型接着写思考内容；
     *                       false 时预填空 think 块，直接作答。
     */
    @Override
    public String render(List<Message> messages, boolean enableThinking) {
        var sb = new StringBuilder();
        for (var m : messages) {
            sb.append(IM_START).append(m.role()).append('\n');
            if (ASSISTANT.equals(m.role())) {
                appendThinkBlock(sb, m.content());
            } else {
                sb.append(m.content());
            }
            sb.append(IM_END).append('\n');
        }
        // 生成起点：思考模式仅预填 <think>\n，非思考模式补齐为空 think 块
        sb.append(IM_START).append(ASSISTANT).append('\n')
                .append(THINK_OPEN).append('\n');
        if (!enableThinking) {
            sb.append('\n').append(THINK_CLOSE).append("\n\n");
        }
        return sb.toString();
    }

    /**
     * 把历史 assistant 消息重建为「<think> 换行 思考 换行 </think> 空行 答案」。
     *
     * 思考为空时仍保留空 think 块（与 jinja 模板一致）。
     */
    private static void appendThinkBlock(StringBuilder sb, String content) {
        ThinkParts parts = ThinkParts.split(content);
        sb.append(THINK_OPEN).append('\n').append(parts.reasoning()).append('\n')
                .append(THINK_CLOSE).append("\n\n").append(parts.answer());
    }
}
