package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.math.Tensor;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.Embedding;
import io.leavesfly.minivllm.model.Linear;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.Qwen3Model;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MmapWeightsTest —— mmap 权重加载的单元测试。
 *
 * 覆盖：
 * 1. 单文件 / 多分片映射：视图数值与磁盘 bf16 位一致
 * 2. 磁盘 dtype=F32 的张量被拒绝（mmap 仅支持 bf16 零转换直读）
 * 3. 分片内容与 weight_map 不一致时报错
 * 4. Linear/Embedding mmap 模式与堆内 bf16 模式全路径 bitwise 一致
 *    （forward/forwardBatch/dotRow/lookup/projectToVocab/projectToVocabFused）
 * 5. Qwen3 骨架级对齐：mmap 加载与 bf16 堆内加载的 prefill/decode logits 完全一致
 */
class MmapWeightsTest {

    @TempDir
    Path dir;

    @Test
    void singleFileBf16MappedValuesCorrect() throws IOException {
        writeBf16Shard("model.safetensors", Map.of(
                "a.weight", bits(1.5f, -2.25f, 0.125f),
                "b.weight", bits(3f)));

        try (MmapWeightsHolder w = MmapWeightsHolder.open(MmapWeights.open(dir))) {
            assertEquals(2, w.m.keys().size());
            assertEquals(3, w.m.length("a.weight"));
            assertEquals((short) (Float.floatToIntBits(1.5f) >>> 16), w.m.view("a.weight").get(0));
            assertEquals((short) (Float.floatToIntBits(-2.25f) >>> 16), w.m.view("a.weight").get(1));
            assertEquals((short) (Float.floatToIntBits(3f) >>> 16), w.m.view("b.weight").get(0));
            assertEquals(-1, w.m.length("missing.weight"));
            assertTrue(w.m.mappedBytes() > 0);
            // readShorts 小张量辅助路径
            assertArrayEquals(bits(3f), w.m.readShorts("b.weight"));
        }
    }

    @Test
    void shardedMmapMergesAllTensors() throws IOException {
        writeBf16Shard("model-00001-of-00002.safetensors", Map.of("a.weight", bits(1f, 2f)));
        writeBf16Shard("model-00002-of-00002.safetensors", Map.of("b.weight", bits(-3f)));
        writeIndex(Map.of(
                "a.weight", "model-00001-of-00002.safetensors",
                "b.weight", "model-00002-of-00002.safetensors"));

        try (MmapWeightsHolder w = MmapWeightsHolder.open(MmapWeights.open(dir))) {
            assertEquals(2, w.m.keys().size());
            assertEquals(bits(1f, 2f)[0], w.m.view("a.weight").get(0));
            assertEquals(bits(-3f)[0], w.m.view("b.weight").get(0));
        }
    }

    @Test
    void f32TensorOnDiskIsRejected() throws IOException {
        writeF32Shard("model.safetensors", Map.of("a.weight", new float[]{1f, 2f}));

        IOException e = assertThrows(IOException.class, () -> MmapWeights.open(dir));
        assertTrue(e.getMessage().contains("BF16"), e.getMessage());
    }

    @Test
    void missingTensorInShardIsDetected() throws IOException {
        writeBf16Shard("model-00001-of-00001.safetensors", Map.of("a.weight", bits(1f)));
        writeIndex(Map.of(
                "a.weight", "model-00001-of-00001.safetensors",
                "c.weight", "model-00001-of-00001.safetensors"));

        assertThrows(IOException.class, () -> MmapWeights.open(dir));
    }

    @Test
    void linearMmapMatchesHeapBf16() throws IOException {
        int in = 32, out = 20;
        short[] w = randomBits(in * out, 42L);
        writeBf16Shard("model.safetensors", Map.of("w", w));
        float[] x = randomFloats(in, 7L);
        float[] xm = randomFloats(5 * in, 8L); // forwardBatch 输入

        Linear heap = Linear.ofBf16(w, in, out);
        try (MmapWeightsHolder h = MmapWeightsHolder.open(MmapWeights.open(dir))) {
            Linear mmap = Linear.ofMmapBf16(h.m.view("w"), in, out);
            assertTrue(mmap.isMmapBf16());
            assertEquals(out * in, mmap.weightMmapBf16().capacity());
            assertArrayEquals(heap.forward(x), mmap.forward(x), 0f, "forward 不一致");
            assertArrayEquals(heap.forwardBatch(xm, 5), mmap.forwardBatch(xm, 5), 0f, "forwardBatch 不一致");
            for (int r = 0; r < out; r++) {
                assertEquals(heap.dotRow(x, 0, r), mmap.dotRow(x, 0, r), 0f, "dotRow 行 " + r);
            }
            assertEquals(heap.numParameters(), mmap.numParameters());
        }
    }

    @Test
    void embeddingMmapMatchesHeapBf16() throws IOException {
        int vocab = 40, d = 16;
        short[] w = randomBits(vocab * d, 3L);
        writeBf16Shard("model.safetensors", Map.of("emb", w));
        float[] hidden = randomFloats(d, 11L);
        float[] hiddenBatch = randomFloats(3 * d, 12L);

        Embedding heap = Embedding.ofBf16(w, vocab, d);
        try (MmapWeightsHolder h = MmapWeightsHolder.open(MmapWeights.open(dir))) {
            Embedding mmap = Embedding.ofMmapBf16(h.m.view("emb"), vocab, d);
            assertTrue(mmap.isMmapBf16());
            assertArrayEquals(heap.lookup(7), mmap.lookup(7), 0f, "lookup 不一致");
            assertArrayEquals(heap.lookupBatch(new int[]{0, 5, 39}),
                    mmap.lookupBatch(new int[]{0, 5, 39}), 0f, "lookupBatch 不一致");
            assertArrayEquals(heap.projectToVocab(hidden), mmap.projectToVocab(hidden), 0f,
                    "projectToVocab 不一致");
            float[][] fusedH = heap.projectToVocabFused(hiddenBatch, 3);
            float[][] fusedM = mmap.projectToVocabFused(hiddenBatch, 3);
            for (int b = 0; b < 3; b++) {
                assertArrayEquals(fusedH[b], fusedM[b], 0f, "projectToVocabFused 序列 " + b);
            }
            assertEquals(heap.numParameters(), mmap.numParameters());
        }
    }

    @Test
    void qwen3MmapMatchesHeapBf16Logits() throws IOException {
        // 骨架级对齐：同一份 bf16 权重，堆内加载 vs mmap 加载，
        // prefill / decode 的 logits 必须 bitwise 一致（同行数据、同一点积内核）
        ModelConfig cfg = new ModelConfig()
                .arch("qwen3")
                .name("qwen3-mmap-test")
                .vocabSize(50)
                .dModel(32)
                .nHead(4)
                .nKVHead(2)
                .headDimExplicit(16)
                .nLayer(2)
                .dFfn(64)
                .blockSize(4)
                .maxSeqLen(64)
                .ropeTheta(10000f)
                .rmsNormEps(1e-6f);

        Map<String, short[]> tensors = tinyQwen3Tensors(cfg, 99L);
        writeBf16Shard("model.safetensors", tensors);

        Qwen3Model heap = Qwen3Loader.loadBf16(cfg, new LinkedHashMap<>(tensors));
        try (MmapWeightsHolder h = MmapWeightsHolder.open(MmapWeights.open(dir))) {
            Qwen3Model mmap = Qwen3Loader.loadMmap(cfg, h.m);

            int[] ids = {3, 1, 4, 1, 5, 9, 2, 6};
            Tensor fwdH = heap.forward(ids);
            Tensor fwdM = mmap.forward(ids);
            assertArrayEquals(fwdH.data(), fwdM.data(), 0f, "PyTorch 风格 forward 不一致");

            // prefill（forwardBatch 路径）
            KVCacheManager kvH = newKvMgr(cfg);
            KVCacheManager kvM = newKvMgr(cfg);
            BlockTable[] btsH = newBlockTables(cfg);
            BlockTable[] btsM = newBlockTables(cfg);
            ensureCapacity(kvH, btsH, ids.length + 1);
            ensureCapacity(kvM, btsM, ids.length + 1);
            float[] preH = heap.prefillLogits(ids, kvH, btsH, 0);
            float[] preM = mmap.prefillLogits(ids, kvM, btsM, 0);
            assertArrayEquals(preH, preM, 0f, "prefill logits 不一致");

            // decode（堆内走融合 QKV、mmap 走三路独立投影，点积逐行一致）
            float[] decH = heap.decodeLogits(8, ids.length, kvH, btsH);
            float[] decM = mmap.decodeLogits(8, ids.length, kvM, btsM);
            assertArrayEquals(decH, decM, 0f, "decode logits 不一致");

            assertEquals(heap.numParameters(), mmap.numParameters());
        }
    }

    // ─── 测试数据构造 ───

    /** 小尺寸 Qwen3 全套权重（无 qk_norm：qkNorm 默认 false，走 Llama 同构骨架） */
    private static Map<String, short[]> tinyQwen3Tensors(ModelConfig cfg, long seed) {
        Random rnd = new Random(seed);
        Map<String, short[]> t = new LinkedHashMap<>();
        t.put("model.embed_tokens.weight", randomBits(cfg.vocabSize() * cfg.dModel(), rnd));
        for (int i = 0; i < cfg.nLayer(); i++) {
            String p = "model.layers." + i + ".";
            t.put(p + "input_layernorm.weight", onesBits(cfg.dModel()));
            t.put(p + "post_attention_layernorm.weight", onesBits(cfg.dModel()));
            t.put(p + "self_attn.q_proj.weight", randomBits(cfg.qDim() * cfg.dModel(), rnd));
            t.put(p + "self_attn.k_proj.weight", randomBits(cfg.kvDim() * cfg.dModel(), rnd));
            t.put(p + "self_attn.v_proj.weight", randomBits(cfg.kvDim() * cfg.dModel(), rnd));
            t.put(p + "self_attn.o_proj.weight", randomBits(cfg.dModel() * cfg.qDim(), rnd));
            t.put(p + "mlp.gate_proj.weight", randomBits(cfg.dFfn() * cfg.dModel(), rnd));
            t.put(p + "mlp.up_proj.weight", randomBits(cfg.dFfn() * cfg.dModel(), rnd));
            t.put(p + "mlp.down_proj.weight", randomBits(cfg.dModel() * cfg.dFfn(), rnd));
        }
        t.put("model.norm.weight", onesBits(cfg.dModel()));
        return t;
    }

    /** 随机 bf16 位（高斯 std=0.05 截断到 bf16 可表达范围） */
    private static short[] randomBits(int n, Random rnd) {
        short[] s = new short[n];
        for (int i = 0; i < n; i++) {
            s[i] = (short) (Float.floatToIntBits((float) (rnd.nextGaussian() * 0.05)) >>> 16);
        }
        return s;
    }

    private static short[] randomBits(int n, long seed) {
        return randomBits(n, new Random(seed));
    }

    private static short[] onesBits(int n) {
        short one = (short) (Float.floatToIntBits(1f) >>> 16);
        short[] s = new short[n];
        java.util.Arrays.fill(s, one);
        return s;
    }

    private static short[] bits(float... floats) {
        short[] s = new short[floats.length];
        for (int i = 0; i < floats.length; i++) {
            s[i] = (short) (Float.floatToIntBits(floats[i]) >>> 16);
        }
        return s;
    }

    private static float[] randomFloats(int n, long seed) {
        Random rnd = new Random(seed);
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) rnd.nextGaussian();
        }
        return f;
    }

    // ─── 最小 safetensors 写入器（BF16 / F32） ───

    /** 写一个合法 safetensors 文件：8B header 长度 + header JSON + 裸 bf16 数据 */
    private void writeBf16Shard(String fileName, Map<String, short[]> tensors) throws IOException {
        writeShard(fileName, tensors, null, "BF16");
    }

    private void writeF32Shard(String fileName, Map<String, float[]> tensors) throws IOException {
        writeShard(fileName, null, tensors, "F32");
    }

    private void writeShard(String fileName, Map<String, short[]> bf16, Map<String, float[]> f32,
                            String dtype) throws IOException {
        int elemBytes = "F32".equals(dtype) ? 4 : 2;
        Map<String, Integer> lens = new LinkedHashMap<>();
        if (bf16 != null) {
            for (Map.Entry<String, short[]> e : bf16.entrySet()) {
                lens.put(e.getKey(), e.getValue().length);
            }
        } else {
            for (Map.Entry<String, float[]> e : f32.entrySet()) {
                lens.put(e.getKey(), e.getValue().length);
            }
        }
        StringBuilder json = new StringBuilder("{");
        long offset = 0;
        int i = 0;
        for (Map.Entry<String, Integer> e : lens.entrySet()) {
            if (i++ > 0) {
                json.append(',');
            }
            long bytes = (long) elemBytes * e.getValue();
            json.append('"').append(e.getKey()).append("\":{\"dtype\":\"").append(dtype)
                    .append("\",\"shape\":[").append(e.getValue()).append("],\"data_offsets\":[")
                    .append(offset).append(',').append(offset + bytes).append("]}");
            offset += bytes;
        }
        json.append('}');
        // HF 官方 safetensors writer 会用空格把 header 补齐到 8 字节对齐（保证数据区对齐），
        // 测试写入器同样补齐，mmap 的 short 视图依赖这一对齐
        while ((json.length() + 8) % 8 != 0) {
            json.append(' ');
        }
        byte[] header = json.toString().getBytes(StandardCharsets.UTF_8);

        Path file = dir.resolve(fileName);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length);
            len.flip();
            ch.write(len);
            ch.write(ByteBuffer.wrap(header));
            if (bf16 != null) {
                for (short[] data : bf16.values()) {
                    ByteBuffer buf = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);
                    buf.asShortBuffer().put(data);
                    ch.write(buf);
                }
            } else {
                for (float[] data : f32.values()) {
                    ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                    buf.asFloatBuffer().put(data);
                    ch.write(buf);
                }
            }
        }
    }

    /** 写 index.json：{"weight_map": {tensor名: 分片文件名}} */
    private void writeIndex(Map<String, String> weightMap) throws IOException {
        StringBuilder json = new StringBuilder("{\"weight_map\":{");
        int i = 0;
        for (Map.Entry<String, String> e : weightMap.entrySet()) {
            if (i++ > 0) {
                json.append(',');
            }
            json.append('"').append(e.getKey()).append("\":\"").append(e.getValue()).append('"');
        }
        json.append("}}");
        Files.writeString(dir.resolve("model.safetensors.index.json"), json.toString());
    }

    // ─── KV cache 辅助 ───

    private static KVCacheManager newKvMgr(ModelConfig cfg) {
        return new KVCacheManager(64, cfg.blockSize(), cfg.kvDim());
    }

    private static BlockTable[] newBlockTables(ModelConfig cfg) {
        BlockTable[] bts = new BlockTable[cfg.nLayer()];
        for (int i = 0; i < bts.length; i++) {
            bts[i] = new BlockTable();
        }
        return bts;
    }

    private static void ensureCapacity(KVCacheManager kvMgr, BlockTable[] bts, int tokens) {
        for (BlockTable bt : bts) {
            assertTrue(kvMgr.ensureCapacity(bt, tokens), "KV 池容量不足");
        }
    }

    /** MmapWeights 本身无需 close（映射随 GC 释放），此 holder 仅为测试内统一写法 */
    private record MmapWeightsHolder(MmapWeights m) implements AutoCloseable {
        static MmapWeightsHolder open(MmapWeights w) {
            assertNotNull(w);
            return new MmapWeightsHolder(w);
        }

        @Override
        public void close() {
            // 映射由 GC Cleaner 释放，无需显式操作
        }
    }
}
