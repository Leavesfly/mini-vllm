package io.leavesfly.minivllm.tokenizer;

import java.util.List;

/**
 * ChatTemplate —— 把 OpenAI messages 数组渲染为模型可理解的提示词。
 *
 * 学习要点：
 * 1. 每个模型家族有自己的对话格式：Qwen3 用 ChatML（{@link Qwen3ChatMLTemplate}），
 *    Llama3 用 header 标记，学习用微模型则直接拼接纯文本（{@link PlainTextTemplate}）。
 * 2. 接口化后 API 层（OpenAiHandler）只依赖本接口，接入新模型时
 *    只需随 ModelFamily 提供对应实现，无需改动 HTTP 处理逻辑。
 */
public interface ChatTemplate {

    /** 一条对话消息（不可变值对象） */
    record Message(String role, String content) {
    }

    /**
     * 渲染 messages 为模型提示词（以 assistant 生成起点结尾）。
     *
     * @param enableThinking 思考模式开关；不支持思考模式的模型可忽略该参数
     */
    String render(List<Message> messages, boolean enableThinking);

    /** 渲染 messages（默认关闭思考模式） */
    default String render(List<Message> messages) {
        return render(messages, false);
    }
}
