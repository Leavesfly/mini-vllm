package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import io.leavesfly.minivllm.tokenizer.MiniMind3ChatMLTemplate;
import io.leavesfly.minivllm.tokenizer.Qwen3ChatMLTemplate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

/**
 * Qwen3FamilyTest —— 对话模板的探测选择逻辑。
 *
 * 同为 qwen3 架构（model_type=qwen3）的模型，思考模式的模板约定可能相反，
 * 因此模板不能写死，要按模型目录 chat_template.jinja 中是否声明 open_thinking 变量来选。
 */
class Qwen3FamilyTest {

    @Test
    void picksMiniMindTemplateWhenJinjaDeclaresOpenThinking(@TempDir Path modelDir) throws IOException {
        Files.writeString(modelDir.resolve("chat_template.jinja"),
                "{%- if open_thinking is defined and open_thinking is true %}"
                        + "{{- '<think>' }}{%- endif %}");
        ChatTemplate template = Qwen3Family.resolveChatTemplate(modelDir);
        assertInstanceOf(MiniMind3ChatMLTemplate.class, template);
    }

    @Test
    void picksQwen3TemplateWhenJinjaHasNoOpenThinking(@TempDir Path modelDir) throws IOException {
        Files.writeString(modelDir.resolve("chat_template.jinja"),
                "{%- if enable_thinking is defined and enable_thinking is false %}"
                        + "{{- '<think>' }}{%- endif %}");
        ChatTemplate template = Qwen3Family.resolveChatTemplate(modelDir);
        assertInstanceOf(Qwen3ChatMLTemplate.class, template);
    }

    /** Qwen3-0.6B 把模板写在 tokenizer_config.json 里，目录下没有 chat_template.jinja */
    @Test
    void fallsBackToQwen3TemplateWhenJinjaMissing(@TempDir Path modelDir) {
        ChatTemplate template = Qwen3Family.resolveChatTemplate(modelDir);
        assertInstanceOf(Qwen3ChatMLTemplate.class, template);
    }
}
