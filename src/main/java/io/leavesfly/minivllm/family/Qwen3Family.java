package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import io.leavesfly.minivllm.tokenizer.MiniMind3ChatMLTemplate;
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
 *    权威来源），回退 config.json 的 eos_token_id，最后按 token 名从分词器反查。
 *    不能把 id 写成常量：同为 qwen3 架构的模型词表规模差异很大
 *    （Qwen3-0.6B 为 151936，MiniMind3 仅 6400），写死的 id 会越界导致永不停止。
 * 3. CPU 推理 0.6B 模型算力有限，建议并发数默认 2（decode 批大小），
 *    可用 --max-seqs 覆盖。
 * 4. 同为 qwen3 架构，对话模板的思考模式约定却可能不同，因此模板也要按模型目录
 *    探测后选择，不能写死（见 {@link #resolveChatTemplate}）。
 */
public final class Qwen3Family implements ModelFamily {

    /** <|im_end|>：ChatML 回合结束标记 */
    private static final String IM_END = "<|im_end|>";
    /** <|endoftext|>：文档结束标记 */
    private static final String ENDOFTEXT = "<|endoftext|>";
    /** chat_template.jinja 中声明该变量的模型，思考模式由模板预填 <think> 开启标记 */
    private static final String OPEN_THINKING = "open_thinking";
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
        int[] eosTokens = resolveEosTokens(modelDir, cfg, tokenizer);
        ChatTemplate chatTemplate = resolveChatTemplate(modelDir);

        System.out.println("Qwen3 模型就绪: " + cfg.nLayer() + " 层, 参数量 "
                + model.numParameters() + ", vocab=" + tokenizer.vocabSize());
        return new LoadedModel(cfg, model, tokenizer, chatTemplate,
                eosTokens, cfg.kvDim(), DEFAULT_MAX_SEQS);
    }

    /**
     * 对话模板选择：读模型目录的 chat_template.jinja，声明了 open_thinking 变量的走
     * MiniMind3 变体，否则用 Qwen3 官方模板。
     *
     * 两者的差异在思考模式的生成起点：Qwen3 官方不预填、由模型自己生成 &lt;think&gt;；
     * MiniMind3 由模板预填 &lt;think&gt; 开启标记。用错模板会让模型输出
     * 只有闭合标记、没有开启标记的畸形 think 块。
     */
    static ChatTemplate resolveChatTemplate(Path modelDir) {
        Path jinja = modelDir.resolve("chat_template.jinja");
        if (Files.isRegularFile(jinja)) {
            try {
                if (Files.readString(jinja).contains(OPEN_THINKING)) {
                    System.out.println("检测到 open_thinking 模板约定，使用 MiniMind3 ChatML 模板");
                    return new MiniMind3ChatMLTemplate();
                }
            } catch (IOException | RuntimeException e) {
                System.out.println("chat_template.jinja 读取失败，回退 Qwen3 官方模板: " + e.getMessage());
            }
        }
        return new Qwen3ChatMLTemplate();
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

    /** EOS 解析：generation_config.json > config.json > 按 token 名从分词器反查 */
    private static int[] resolveEosTokens(Path modelDir, ModelConfig cfg, SimpleTokenizer tokenizer) {
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
                : eosFromTokenizer(tokenizer);
    }

    /**
     * 从分词器的 special token 表按名字反查 EOS id。
     * 适用于 config 与 generation_config 都未声明 eos_token_id 的模型（如 MiniMind3）：
     * 名字在 Qwen3 生态内稳定，而 id 随词表规模变化。
     */
    private static int[] eosFromTokenizer(SimpleTokenizer tokenizer) {
        if (!(tokenizer instanceof BpeTokenizer bpe)) {
            return new int[0];
        }
        int[] ids = new int[2];
        int n = 0;
        for (String name : new String[]{IM_END, ENDOFTEXT}) {
            int id = bpe.specialId(name);
            if (id >= 0) {
                ids[n++] = id;
            }
        }
        if (n == 0) {
            System.out.println("警告: 未能解析出 EOS token，生成将仅受 max_tokens 限制");
        }
        return java.util.Arrays.copyOf(ids, n);
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
