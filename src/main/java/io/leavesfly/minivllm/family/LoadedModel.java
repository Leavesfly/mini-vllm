package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;

/**
 * LoadedModel —— 一次模型加载的完整产物（不可变值对象）。
 *
 * 学习要点：
 * 引擎装配需要的所有模型侧信息都收拢在这里：权重（model）、分词器、
 * 对话模板、EOS 集合、KV cache 维度与默认并发数。
 * ModelFamily 负责生产它，MiniVllmServer 只做纯装配，
 * 两者之间不再泄露任何架构特定的常量或判断。
 *
 * @param config         模型超参数配置
 * @param model          可推理的模型实例
 * @param tokenizer      分词器
 * @param chatTemplate   对话模板（messages → prompt）
 * @param eosTokens      停止 token 集合（空数组表示无 EOS）
 * @param kvDim          KV cache 每 token 的向量长度（决定 KV 池布局）
 * @param defaultMaxSeqs 该模型在 CPU 上的建议并发序列数
 */
public record LoadedModel(ModelConfig config, LlmModel model, SimpleTokenizer tokenizer,
                          ChatTemplate chatTemplate, int[] eosTokens, int kvDim,
                          int defaultMaxSeqs) {
}
