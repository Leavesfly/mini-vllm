package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.family.LoadedModel;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;

/**
 * ModelRuntime —— 一个模型的完整可服务运行时（不可变值对象）。
 *
 * 学习要点：
 * 1. 一次「服务一个模型」需要三样东西：模型权重（LoadedModel）、独占的 KV 池、
 *    以及驱动 continuous batching 的引擎线程。这三者绑定成一个运行时，
 *    多模型服务就是多个互不干扰的运行时（各自的 KV 池与调度队列）。
 * 2. KV 池不能跨模型共享：block 的 kvDim 由模型的 GQA 头数与 headDim 决定，
 *    不同模型的 block 布局不兼容。
 *
 * @param id          对外暴露的模型 id（OpenAI 请求体 model 字段的取值，通常为模型目录名）
 * @param loaded      模型加载产物（权重 / 分词器 / 对话模板 / EOS）
 * @param engine      已启动的推理引擎
 * @param maxNumSeqs  该运行时的并发序列上限
 * @param numBlocks   该运行时 KV 池的 block 数
 */
public record ModelRuntime(String id, LoadedModel loaded, LLMEngine engine,
                           int maxNumSeqs, int numBlocks) {

    /** 该模型的对话模板（messages → prompt），API 层按请求选中的模型渲染 */
    public ChatTemplate chatTemplate() {
        return loaded.chatTemplate();
    }
}
