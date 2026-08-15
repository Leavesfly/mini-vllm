package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import io.leavesfly.minivllm.tokenizer.JinjaChatTemplate;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * LlamaFamily —— Llama-3 系列模型的家族实现（model_type = "llama"）。
 *
 * 学习要点：
 * 1. Llama-3 与 Qwen3 架构同构（RMSNorm + RoPE half-split + GQA + SwiGLU，
 *    权重张量名也一致），全部差异收敛在三处配置而非代码：
 *      - 无 QK-Norm：由 {@link ModelConfig#qkNorm()} 承载（fromConfigJson 自动判定）
 *      - RoPE 频率缩放（llama3 风格）：由 {@link ModelConfig#ropeScaling()} 承载
 *      - 对话模板/EOS 不同：模板走通用 {@link JinjaChatTemplate}（渲染模型自带
 *        chat_template），EOS 按 <|eot_id|> / <|end_of_text|> 反查
 *    模型骨架直接复用 Qwen3Model/Qwen3Loader——这正是「家族差异配置化」的价值。
 * 2. 分词器是 byte-level BPE（tokenizer.json 单文件导出，无 vocab.json/merges.txt），
 *    BpeTokenizer 已通用支持；chat 模板自带 bos_token，encode 无需额外加 BOS。
 * 3. 1B/3B 小尺寸为 tied embeddings（lm_head 复用 embed_tokens），与 Qwen3-0.6B 相同。
 */
public final class LlamaFamily implements ModelFamily {

    /** <|eot_id|>：Llama-3 对话回合结束标记 */
    private static final String EOT_ID = "<|eot_id|>";
    /** <|end_of_text|>：文档结束标记 */
    private static final String END_OF_TEXT = "<|end_of_text|>";
    private static final int DEFAULT_MAX_SEQS = 2;

    @Override
    public boolean supports(String modelType) {
        return "llama".equals(modelType);
    }

    @Override
    public LoadedModel load(Path modelDir, Precision precision, boolean random, int maxSeqLenCap)
            throws IOException {
        ModelConfig cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                Files.readString(modelDir.resolve("config.json"))));
        cfg.maxSeqLen(Math.min(cfg.maxSeqLen(), maxSeqLenCap));

        LlmModel model = ModelFamily.loadModelWeights(cfg, modelDir, precision, random);
        SimpleTokenizer tokenizer = BpeTokenizer.fromModelDir(modelDir);
        int[] eosTokens = ModelFamily.resolveEosTokens(modelDir, cfg, tokenizer, EOT_ID, END_OF_TEXT);
        ChatTemplate chatTemplate = JinjaChatTemplate.fromModelDir(modelDir);

        System.out.println("Llama 模型就绪: " + cfg.nLayer() + " 层, 参数量 "
                + model.numParameters() + ", vocab=" + tokenizer.vocabSize());
        return new LoadedModel(cfg, model, tokenizer, chatTemplate,
                eosTokens, cfg.kvDim(), DEFAULT_MAX_SEQS);
    }
}
