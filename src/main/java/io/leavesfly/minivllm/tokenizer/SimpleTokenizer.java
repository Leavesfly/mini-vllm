package io.leavesfly.minivllm.tokenizer;

/**
 * 分词器接口 —— 文本与 token id 序列之间的双向转换。
 *
 * 学习要点：
 * 1. 真实 vLLM 用 BPE/SentencePiece 等复杂分词器（如 GPT-2 的 byte-level BPE，词表 50257）。
 * 2. 零依赖下手写完整 BPE 不现实，本项目提供两种极简实现：
 *      - ByteTokenizer：UTF-8 字节级，每个字节一个 token，词表 256，能编码任意文本。
 *      - CharTokenizer：字符级，基于给定字符表，适合 char-level 微模型（如 nanoGPT Shakespeare）。
 * 3. 接口抽象后，未来可替换为调用外部 tokenizer 服务的实现。
 */
public interface SimpleTokenizer {

    /** 文本 -> token id 序列 */
    int[] encode(String text);

    /** token id 序列 -> 文本 */
    String decode(int[] ids);

    /** 词表大小 */
    int vocabSize();
}
