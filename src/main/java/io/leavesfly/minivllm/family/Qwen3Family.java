package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.Qwen3ChatMLTemplate;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.weights.Qwen3Loader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Qwen3Family —— Qwen3 系列模型的家族实现。
 *
 * 学习要点：
 * 1. Qwen3 的全部架构差异都收拢在这里：config.json 解析（GQA / RoPE theta /
 *    RMSNorm eps）、safetensors 权重加载（f32 / bf16 常驻）、BPE 分词器、
 *    ChatML 对话模板、EOS 集合。
 * 2. EOS 读取顺序与 HF 生态一致：generation_config.json 优先（这是生成期配置的
 *    权威来源），回退 config.json 的 eos_token_id，最后兜底 Qwen3 已知常量。
 * 3. CPU 推理 0.6B 模型算力有限，建议并发数默认 2（decode 批大小），
 *    可用 --max-seqs 覆盖。
 */
public final class Qwen3Family implements ModelFamily {

    /** <|im_end|>：ChatML 回合结束标记 */
    private static final int EOS_IM_END = 151645;
    /** <|endoftext|>：文档结束标记 */
    private static final int EOS_ENDOFTEXT = 151643;
    private static final int DEFAULT_MAX_SEQS = 2;

    @Override
    public boolean supports(String modelType) {
        return "qwen3".equals(modelType);
    }

    @Override
    public LoadedModel load(Path modelDir, Precision precision, boolean random, int maxSeqLenCap)
            throws IOException {
        ModelConfig cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                Files.readString(modelDir.resolve("config.json"))));
        cfg.maxSeqLen(Math.min(cfg.maxSeqLen(), maxSeqLenCap));

        LlmModel model = random ? randomModel(cfg) : loadWeights(cfg, modelDir, precision);
        SimpleTokenizer tokenizer = BpeTokenizer.fromModelDir(modelDir);
        int[] eosTokens = resolveEosTokens(modelDir, cfg);

        System.out.println("Qwen3 模型就绪: " + cfg.nLayer() + " 层, 参数量 "
                + model.numParameters() + ", vocab=" + tokenizer.vocabSize());
        return new LoadedModel(cfg, model, tokenizer, new Qwen3ChatMLTemplate(),
                eosTokens, cfg.kvDim(), DEFAULT_MAX_SEQS);
    }

    private static LlmModel randomModel(ModelConfig cfg) {
        System.out.println("使用随机初始化 Qwen3 模型（输出无意义，仅验证流程）");
        return Qwen3Loader.randomInit(cfg);
    }

    private static LlmModel loadWeights(ModelConfig cfg, Path modelDir, Precision precision)
            throws IOException {
        Path file = modelDir.resolve("model.safetensors");
        System.out.println("加载权重: " + file
                + (precision == Precision.BF16 ? " (bf16 常驻)" : " (f32 常驻)"));
        long t0 = System.currentTimeMillis();
        LlmModel model;
        if (precision == Precision.BF16) {
            Map<String, short[]> weights = SafetensorsLoader.loadBf16Bits(file);
            System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                    weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
            model = Qwen3Loader.loadBf16(cfg, weights);
        } else {
            Map<String, float[]> weights = SafetensorsLoader.load(file);
            System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                    weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
            model = Qwen3Loader.load(cfg, weights);
        }
        return model;
    }

    /** EOS 解析：generation_config.json > config.json > Qwen3 已知常量兜底 */
    private static int[] resolveEosTokens(Path modelDir, ModelConfig cfg) {
        Path genCfg = modelDir.resolve("generation_config.json");
        if (Files.isRegularFile(genCfg)) {
            try {
                Object eos = SimpleJson.parseObject(Files.readString(genCfg)).get("eos_token_id");
                int[] ids = eosIdsOf(eos);
                if (ids.length > 0) {
                    return ids;
                }
            } catch (IOException | RuntimeException e) {
                System.out.println("generation_config.json 解析失败，回退 config.json: " + e.getMessage());
            }
        }
        return cfg.eosTokenIds().length > 0
                ? cfg.eosTokenIds()
                : new int[]{EOS_IM_END, EOS_ENDOFTEXT};
    }

    /** eos_token_id 字段兼容两种形态：单个数字 或 数字数组 */
    private static int[] eosIdsOf(Object eos) {
        if (eos instanceof List<?> list) {
            int[] ids = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                ids[i] = ((Number) list.get(i)).intValue();
            }
            return ids;
        }
        if (eos instanceof Number n) {
            return new int[]{n.intValue()};
        }
        return new int[0];
    }
}
