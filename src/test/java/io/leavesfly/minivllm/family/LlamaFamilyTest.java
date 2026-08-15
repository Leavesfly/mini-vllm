package io.leavesfly.minivllm.family;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.RotaryEmbedding;
import io.leavesfly.minivllm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * LlamaFamilyTest —— Llama-3 家族接入的单元测试。
 *
 * 验证「家族差异配置化」的三处差异承载点：
 * 1. config.json 解析：qkNorm=false、rope_scaling(llama3) 参数提取
 * 2. RoPE llama3 频率缩放：与手算参考值逐点比对（短波长不变 / 长波长除 factor / 中间平滑）
 * 3. 家族装配：合成最小模型目录走通 load（随机权重），jinja 模板渲染、EOS 反查、前向跑通
 */
class LlamaFamilyTest {

    private static final String LLAMA_CONFIG = """
            {"model_type": "llama", "hidden_size": 32, "num_attention_heads": 2,
             "num_key_value_heads": 1, "num_hidden_layers": 2, "intermediate_size": 64,
             "vocab_size": 16, "max_position_embeddings": 131072, "rms_norm_eps": 1e-5,
             "rope_theta": 500000.0, "tie_word_embeddings": true,
             "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                              "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}}
            """;

    @Test
    void llamaConfigJsonParsed() {
        ModelConfig cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(LLAMA_CONFIG));
        assertEquals("llama", cfg.arch());
        assertFalse(cfg.qkNorm(), "Llama 无 QK-Norm");
        assertEquals(500000f, cfg.ropeTheta(), 1e-3);
        assertArrayEquals(new float[]{32f, 1f, 4f, 8192f}, cfg.ropeScaling());
        assertEquals(32, cfg.dModel());
        assertEquals(1, cfg.kvHeads());
        assertEquals(131072, cfg.maxSeqLen());
    }

    @Test
    void qwen3ConfigKeepsQkNormAndNoScaling() {
        // 回归：Qwen3 的 config（无 rope_scaling）保持 qkNorm=true、不缩放
        ModelConfig cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                "{\"model_type\": \"qwen3\", \"hidden_size\": 32, \"num_attention_heads\": 2}"));
        assertTrue(cfg.qkNorm());
        assertNull(cfg.ropeScaling());
    }

    @Test
    void unsupportedRopeScalingTypeIsRejected() {
        IllegalArgumentException e = org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class,
                () -> ModelConfig.fromConfigJson(SimpleJson.parseObject(
                        "{\"model_type\": \"llama\", \"rope_scaling\": {\"rope_type\": \"yarn\"}}")));
        assertTrue(e.getMessage().contains("yarn"));
    }

    @Test
    void llama3RopeScalingMatchesHandComputed() {
        // headDim=8, theta=500000, llama3 scaling [32, 1, 4, 8192]
        RotaryEmbedding rope = new RotaryEmbedding(8, 8, 500000f, new float[]{32f, 1f, 4f, 8192f});
        // i=0：wavelen 2π < 2048 → 不缩放，cos(pos=2) = cos(2·1.0)
        assertEquals(-0.4161468365f, rope.cosAt(2, 0), 1e-5f);
        // i=2：wavelen 4442.88 ∈ (2048, 8192) → 平滑插值后 invFreq=4.2955679656e-4
        assertEquals(0.9999996310f, rope.cosAt(2, 2), 1e-6f);
        // i=3：wavelen 118142 > 8192 → invFreq / 32，cos(pos=2) ≈ 1
        assertEquals(1.0f, rope.cosAt(2, 3), 1e-5f);
        // 对照：无缩放时 i=3 的 cos(pos=2) = cos(2·5.3182959e-5)
        RotaryEmbedding plain = new RotaryEmbedding(8, 8, 500000f);
        assertEquals((float) Math.cos(2 * 5.31829590e-05), plain.cosAt(2, 3), 1e-7f);
    }

    @Test
    void llamaFamilyAssemblesRandomModel(@TempDir Path dir) throws IOException {
        // 合成最小模型目录：llama config + tokenizer.json（含 special）+ tokenizer_config（含模板）
        Files.writeString(dir.resolve("config.json"), LLAMA_CONFIG);
        Files.writeString(dir.resolve("tokenizer.json"), """
                {"model": {"type": "BPE", "vocab": {"<|begin_of_text|>": 0, "<|eot_id|>": 1,
                 "<|end_of_text|>": 2, "a": 3, "b": 4}, "merges": []},
                 "added_tokens": [{"id": 0, "content": "<|begin_of_text|>", "special": true},
                                  {"id": 1, "content": "<|eot_id|>", "special": true},
                                  {"id": 2, "content": "<|end_of_text|>", "special": true}]}
                """);
        Files.writeString(dir.resolve("tokenizer_config.json"), """
                {"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>",
                 "chat_template": "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\\n\\n{{ m.content }}<|eot_id|>{% endfor %}<|start_header_id|>assistant<|end_header_id|>\\n\\n"}
                """);

        LoadedModel loaded = new LlamaFamily().load(dir, Precision.BF16, true, 512);

        // 配置：qkNorm=false、scaling 提取、上下文被 cap 裁剪
        assertFalse(loaded.config().qkNorm());
        assertEquals(512, loaded.config().maxSeqLen());
        // EOS：generation_config 缺失 → config 缺失 → 按名反查到 <|eot_id|>=1 / <|end_of_text|>=2
        assertArrayEquals(new int[]{1, 2}, loaded.eosTokens());
        // jinja 模板渲染（含 bos_token 上下文）；模板尾换行按 jinja2 keep_trailing_newline=false 剥离一个
        String prompt = loaded.chatTemplate().render(List.of(new ChatTemplate.Message("user", "Hi")), false);
        assertEquals("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                + "<|start_header_id|>assistant<|end_header_id|>\n", prompt);
        // 随机模型前向跑通（KV cache 装配维度 = kvDim = 1 头 × 16；每层一张 BlockTable）
        KVCacheManager kvMgr = new KVCacheManager(8, 16, loaded.config().kvDim());
        BlockTable[] bts = new BlockTable[loaded.config().nLayer()];
        for (int i = 0; i < bts.length; i++) {
            bts[i] = new BlockTable();
            kvMgr.ensureCapacity(bts[i], 4);
        }
        float[] logits = loaded.model().decodeLogits(3, 0, kvMgr, bts);
        assertEquals(loaded.config().vocabSize(), logits.length);
    }

    @Test
    void registryDispatchesLlamaModelType(@TempDir Path dir) throws IOException {
        // ModelRegistry 按 model_type 路由：llama 应命中 LlamaFamily（random 模式跳过权重文件）
        Files.writeString(dir.resolve("config.json"), LLAMA_CONFIG);
        Files.writeString(dir.resolve("tokenizer.json"), """
                {"model": {"type": "BPE", "vocab": {"<|eot_id|>": 1, "a": 3}, "merges": []}}
                """);
        Files.writeString(dir.resolve("tokenizer_config.json"),
                "{\"chat_template\": \"{{ messages[0].content }}\"}");
        ModelRegistry registry = new ModelRegistry(List.of(new Qwen3Family(), new LlamaFamily()));
        LoadedModel loaded = registry.load(dir, Precision.BF16, true, 512);
        assertEquals("llama", loaded.config().arch());
        assertFalse(loaded.config().qkNorm());
    }
}
