package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.weights.MmapWeights;
import io.leavesfly.minivllm.weights.Qwen3Loader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * ModelFamily —— 模型家族的接入点（SPI）。
 *
 * 学习要点：
 * 1. 一个「家族」收拢一种架构的全部差异：config.json 解析、权重加载、
 *    分词器、对话模板、EOS 集合、建议并发数。
 * 2. 引擎核心（PagedAttention / Continuous Batching / OpenAI API）只依赖
 *    LlmModel 与 ChatTemplate 等接口，与家族无关——接入新模型时
 *    只需新增一个 ModelFamily 实现并在
 *    META-INF/services/io.leavesfly.minivllm.family.ModelFamily 中注册，
 *    主干代码零改动（对照 vLLM 的 ModelRegistry 机制）。
 * 3. 通过 JDK 标准库 {@link java.util.ServiceLoader} 发现实现，保持零外部依赖。
 */
public interface ModelFamily {

    /** 是否支持该架构（匹配 config.json 的 model_type，如 "qwen3"） */
    boolean supports(String modelType);

    /**
     * 从 HuggingFace 风格的模型目录加载完整模型。
     *
     * @param modelDir     模型目录（含 config.json / model.safetensors / tokenizer 文件）
     * @param precision    权重常驻精度
     * @param random       true 时跳过权重文件，随机初始化（仅验证流程）
     * @param maxSeqLenCap 上下文窗口上限（取 min(config, cap)，控制 KV 池内存）
     * @return 装配引擎所需的完整加载产物
     */
    LoadedModel load(Path modelDir, Precision precision, boolean random, int maxSeqLenCap)
            throws IOException;

    // ─── 家族共用装配助手 ───

    /**
     * 按精度从权重文件装配模型（Qwen3/Llama 等 RMSNorm+RoPE+GQA+SwiGLU 同构骨架共用）：
     * 权重名一致，qk_norm 有无与 RoPE 缩放由 {@link ModelConfig} 承载。
     * random=true 时跳过权重文件随机初始化（仅验证流程）。
     */
    static LlmModel loadModelWeights(ModelConfig cfg, Path modelDir, Precision precision,
                                     boolean random) throws IOException {
        if (random) {
            System.out.println("使用随机初始化模型（输出无意义，仅验证流程）");
            return Qwen3Loader.randomInit(cfg);
        }
        // 预检堆容量：f32 常驻 0.6B 级权重就要 ~2.5GB，堆不够时与其加载到一半 OOM
        //（浪费已读的 IO，还会把 OutOfMemoryError 抛到 HTTP 线程），不如提前报可操作的错。
        // mmap 不落堆（权重由 OS 页缓存驻留），无需预检。
        if (precision != Precision.MMAP) {
            ensureHeapFor(modelDir, precision);
        }
        System.out.println("加载权重: " + modelDir + " (" + precision.name().toLowerCase() + " 常驻)");
        long t0 = System.currentTimeMillis();
        try {
            if (precision == Precision.MMAP) {
                // mmap：只建页表不读数据，加载近乎瞬时；真正的 IO 推迟到首次推理缺页时
                MmapWeights weights = MmapWeights.open(modelDir);
                System.out.printf("权重映射完成: %d 个张量, %.1fGB, %.1f s（按需调页，不占堆）%n",
                        weights.keys().size(), weights.mappedBytes() / 1e9,
                        (System.currentTimeMillis() - t0) / 1000.0);
                return Qwen3Loader.loadMmap(cfg, weights);
            }
            if (precision == Precision.F32) {
                Map<String, float[]> weights = SafetensorsLoader.loadDir(modelDir);
                System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                        weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
                return Qwen3Loader.load(cfg, weights);
            }
            Map<String, short[]> weights = SafetensorsLoader.loadDirBf16Bits(modelDir);
            System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                    weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
            return switch (precision) {
                case BF16 -> Qwen3Loader.loadBf16(cfg, weights);
                case INT8 -> Qwen3Loader.loadInt8(cfg, weights);
                case INT4 -> Qwen3Loader.loadInt4(cfg, weights);
                default -> throw new IllegalStateException("不可达: " + precision);
            };
        } catch (OutOfMemoryError oom) {
            // 预检是启发式的（并发请求/其它模型也在用堆），真碰了 OOM 也要兜住：
            // 转成 IOException 让调用方（API 层 500 / 启动失败）拿到可操作的指引，
            // 而不是 Error 穿透直接杀死线程
            throw new IOException(heapHint("权重加载中途堆内存耗尽", precision), oom);
        }
    }

    /**
     * 加载前堆容量预检：按精度估算内存峰值，不够时直接报带指引的错。
     *
     * 峰值系数（以磁盘权重文件字节数为基准）：
     *   F32  ≈ 2.2×：HF 权重主流是 bf16，常驻要加宽成 f32（2×）再加余量
     *   BF16 ≈ 1.3×：short[] 原宽常驻
     *   INT8 ≈ 1.8×：先全量读 bf16 位（1×）再逐张量量化，峰值 ≈ bf16 字典 + int8 常驻
     *   INT4 ≈ 1.5×：同上，int4 常驻更省
     * 另加 256MB 固定余量容纳运行期缓冲（prefill 激活、采样 logits 等）。
     */
    private static void ensureHeapFor(Path modelDir, Precision precision) throws IOException {
        long fileBytes = SafetensorsLoader.weightFileBytes(modelDir);
        double factor = switch (precision) {
            case F32 -> 2.2;
            case BF16 -> 1.3;
            case INT8 -> 1.8;
            case INT4 -> 1.5;
            case MMAP -> throw new AssertionError("mmap 不落堆，调用方已跳过预检");
        };
        long need = (long) (fileBytes * factor) + (256L << 20);
        Runtime rt = Runtime.getRuntime();
        long avail = rt.maxMemory() - rt.totalMemory() + rt.freeMemory();
        if (avail >= need) {
            return;
        }
        long gb = 1024L * 1024 * 1024;
        long used = rt.totalMemory() - rt.freeMemory();
        long suggestGb = (used + need + gb - 1) / gb; // 向上取整到 GB
        throw new IOException(String.format(
                "堆内存不足以加载权重：%s 常驻约需 %.1fGB，当前可用 %.1fGB（maxHeap %.1fGB）。"
                        + "请用 -Xmx%dg 增大堆%s",
                precision.name().toLowerCase(), need / 1e9, avail / 1e9,
                rt.maxMemory() / 1e9, suggestGb, lowerPrecisionHint(precision)));
    }

    /**
     * 降内存的备选建议：只列比当前更省的（枚举顺序 F32→BF16→INT8→INT4→MMAP 递减，
     * MMAP 排末尾故对任何堆内精度都会被自然列出）；已是 MMAP 则无可再降。
     */
    private static String lowerPrecisionHint(Precision current) {
        if (current == Precision.MMAP) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Precision p : Precision.values()) {
            if (p.ordinal() > current.ordinal()) {
                sb.append(sb.length() == 0 ? "，或用 --precision " : "/").append(p.name().toLowerCase());
            }
        }
        return sb.length() == 0 ? "" : sb + " 降低常驻精度";
    }

    /** OOM 兜底的统一报错文案（带堆现状与两条出路） */
    private static String heapHint(String what, Precision precision) {
        Runtime rt = Runtime.getRuntime();
        return String.format("%s（maxHeap %.1fGB）。请用 -Xmx 增大堆%s",
                what, rt.maxMemory() / 1e9, lowerPrecisionHint(precision));
    }

    /**
     * EOS 解析（家族共用）：generation_config.json 优先（生成期配置的权威来源），
     * 回退 config.json 的 eos_token_id，最后按 token 名从分词器反查（名字在生态内
     * 稳定，id 随词表规模变化）。
     *
     * @param fallbackNames 按名反查的候选 special token（如 "<|im_end|>" / "<|eot_id|>"）
     */
    static int[] resolveEosTokens(Path modelDir, ModelConfig cfg, SimpleTokenizer tokenizer,
                                  String... fallbackNames) throws IOException {
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
        if (cfg.eosTokenIds().length > 0) {
            return cfg.eosTokenIds();
        }
        return eosFromTokenizer(tokenizer, fallbackNames);
    }

    /** 从分词器的 special token 表按名字反查 EOS id（目前仅 BpeTokenizer 支持反查） */
    private static int[] eosFromTokenizer(SimpleTokenizer tokenizer, String... names) {
        if (!(tokenizer instanceof BpeTokenizer bpe)) {
            return new int[0];
        }
        int[] ids = new int[names.length];
        int n = 0;
        for (String name : names) {
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
