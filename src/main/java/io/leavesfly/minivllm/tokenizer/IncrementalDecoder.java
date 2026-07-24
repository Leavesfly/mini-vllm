package io.leavesfly.minivllm.tokenizer;

/**
 * IncrementalDecoder —— 流式增量解码器接口：逐 token 喂入，返回可安全输出的文本片段。
 *
 * 学习要点：
 * 1. 流式输出时一个多字节字符（中文/emoji）可能被 token 边界切开，
 *    直接逐 token decode 会输出乱码（'�'）。增量解码器缓冲未完成的字节，
 *    只在凑齐完整字符后输出。
 * 2. 引擎（core 层）只依赖本接口，不关心具体分词器如何缓冲——
 *    BPE 实现按 byte-level 映射缓冲 UTF-8 字节，简单分词器逐 token 直接解码即可。
 */
public interface IncrementalDecoder {

    /** 喂入一个 token，返回本次可输出的文本（可能为空串） */
    String accept(int tokenId);

    /** 冲刷剩余缓冲字节（生成结束时调用），无剩余时返回空串 */
    String flush();
}
