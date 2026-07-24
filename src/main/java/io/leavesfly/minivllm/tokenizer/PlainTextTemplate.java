package io.leavesfly.minivllm.tokenizer;

import java.util.List;

/**
 * PlainTextTemplate —— 学习用微模型的纯文本对话模板。
 *
 * 学习要点：
 * 随机初始化的 GPT-2/GPT-3 微模型没有经过对话微调，不认识任何特殊标记，
 * 因此只做最朴素的「角色: 内容」逐行拼接，末尾以 "assistant:" 引导生成。
 * 这也演示了 ChatTemplate 接口的最小实现形态。
 */
public final class PlainTextTemplate implements ChatTemplate {

    @Override
    public String render(List<Message> messages, boolean enableThinking) {
        var sb = new StringBuilder();
        for (var m : messages) {
            sb.append(m.role()).append(": ").append(m.content()).append('\n');
        }
        sb.append("assistant: ");
        return sb.toString();
    }
}
