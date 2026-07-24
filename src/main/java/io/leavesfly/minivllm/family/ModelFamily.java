package io.leavesfly.minivllm.family;

import java.io.IOException;
import java.nio.file.Path;

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
}
