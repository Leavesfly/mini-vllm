package io.leavesfly.minivllm.tokenizer;

import java.util.List;

/**
 * ChatTemplate —— 把 OpenAI messages 数组渲染为 Qwen3 的 ChatML 提示词。
 *
 * 学习要点：
 * 1. Qwen3 使用 ChatML 格式组织多轮对话：
 *      <|im_start|>system\n{system}<|im_end|>\n
 *      <|im_start|>user\n{user}<|im_end|>\n
 *      <|im_start|>assistant\n{assistant}<|im_end|>\n
 *      ...
 *      <|im_start|>assistant\n        <- 生成起点
 * 2. Qwen3 官方模板只在首条消息为 system 时才输出 system 块，
 *    不自动补默认 system（与 HF apply_chat_template 严格对齐）。
 * 3. <|im_start|> / <|im_end|> 是 special token，由 BpeTokenizer 直接映射 id，
 *    不参与 BPE 合并。
 */
public final class ChatTemplate {

    public static final String IM_START = "<|im_start|>";
    public static final String IM_END = "<|im_end|>";
    /** Qwen3 思考模式标记 */
    public static final String THINK_OPEN = "<think>";
    public static final String THINK_CLOSE = "</think>";

    private ChatTemplate() {
    }

    /** 一条对话消息（不可变值对象） */
    public record Message(String role, String content) {
    }

    /**
     * 渲染 messages 为 ChatML 提示词（以 assistant 生成起点结尾）。
     * 默认关闭思考模式：与 Qwen3 官方 enable_thinking=false 一致，
     * 预填空的 <think></think> 块，模型跳过内部推理直接作答。
     */
    public static String applyChatML(List<Message> messages) {
        return applyChatML(messages, false);
    }

    /**
     * @param enableThinking true 时模板末尾仅 <assistant>\n，模型自行生成 <think> 推理；
     *                       false 时预填空 think 块，直接作答（响应更快、更简短）。
     */
    public static String applyChatML(List<Message> messages, boolean enableThinking) {
        var sb = new StringBuilder();
        for (var m : messages) {
            sb.append(IM_START).append(m.role()).append('\n')
                    .append(m.content()).append(IM_END).append('\n');
        }
        sb.append(IM_START).append("assistant\n");
        if (!enableThinking) {
            sb.append(THINK_OPEN).append("\n\n").append(THINK_CLOSE).append("\n\n");
        }
        return sb.toString();
    }
}
