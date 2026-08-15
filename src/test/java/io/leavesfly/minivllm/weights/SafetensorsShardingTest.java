package io.leavesfly.minivllm.weights;

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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SafetensorsShardingTest —— 多分片权重加载（loadDir / loadDirBf16Bits）的单元测试。
 *
 * 覆盖：
 * 1. 多分片合并：index.json 路由 + 并行读取后按 tensor 名合并，数值无损
 * 2. bf16 常驻路径同样支持分片
 * 3. 合并结果与 weight_map 不一致（缺张量 / 多张量）时报错——防不完整下载
 * 4. 单文件目录自动回退，行为与 load 一致
 * 5. 既无单文件也无索引时给出明确错误
 */
class SafetensorsShardingTest {

    @TempDir
    Path dir;

    @Test
    void shardedLoadMergesAllTensors() throws IOException {
        writeShard("model-00001-of-00002.safetensors", Map.of(
                "a.weight", new float[]{1f, 2f, 3f},
                "b.weight", new float[]{4f}));
        writeShard("model-00002-of-00002.safetensors", Map.of(
                "c.weight", new float[]{5f, 6f}));
        writeIndex(Map.of(
                "a.weight", "model-00001-of-00002.safetensors",
                "b.weight", "model-00001-of-00002.safetensors",
                "c.weight", "model-00002-of-00002.safetensors"));

        Map<String, float[]> merged = SafetensorsLoader.loadDir(dir);

        assertEquals(3, merged.size());
        assertArrayEquals(new float[]{1f, 2f, 3f}, merged.get("a.weight"));
        assertArrayEquals(new float[]{4f}, merged.get("b.weight"));
        assertArrayEquals(new float[]{5f, 6f}, merged.get("c.weight"));
    }

    @Test
    void shardedBf16BitsLoad() throws IOException {
        writeShard("model-00001-of-00002.safetensors", Map.of("a.weight", new float[]{1.5f}));
        writeShard("model-00002-of-00002.safetensors", Map.of("b.weight", new float[]{-2.25f}));
        writeIndex(Map.of(
                "a.weight", "model-00001-of-00002.safetensors",
                "b.weight", "model-00002-of-00002.safetensors"));

        Map<String, short[]> merged = SafetensorsLoader.loadDirBf16Bits(dir);

        assertEquals(2, merged.size());
        // f32 -> bf16 截断 = 高 16 位
        assertEquals((short) (Float.floatToIntBits(1.5f) >>> 16), merged.get("a.weight")[0]);
        assertEquals((short) (Float.floatToIntBits(-2.25f) >>> 16), merged.get("b.weight")[0]);
    }

    @Test
    void missingTensorInShardIsDetected() throws IOException {
        // 索引声明了 c.weight，但分片里没有：必须报错而非静默缺权重
        writeShard("model-00001-of-00001.safetensors", Map.of("a.weight", new float[]{1f}));
        writeIndex(Map.of(
                "a.weight", "model-00001-of-00001.safetensors",
                "c.weight", "model-00001-of-00001.safetensors"));

        IOException e = assertThrows(IOException.class, () -> SafetensorsLoader.loadDir(dir));
        assertTrue(e.getMessage().contains("weight_map"), e.getMessage());
    }

    @Test
    void unexpectedTensorInShardIsDetected() throws IOException {
        // 分片多出索引外的张量：同样视为损坏
        writeShard("model-00001-of-00001.safetensors", Map.of(
                "a.weight", new float[]{1f},
                "ghost.weight", new float[]{9f}));
        writeIndex(Map.of("a.weight", "model-00001-of-00001.safetensors"));

        assertThrows(IOException.class, () -> SafetensorsLoader.loadDir(dir));
    }

    @Test
    void singleFileFallback() throws IOException {
        writeShard("model.safetensors", Map.of("only.weight", new float[]{7f, 8f}));

        Map<String, float[]> merged = SafetensorsLoader.loadDir(dir);

        assertEquals(1, merged.size());
        assertArrayEquals(new float[]{7f, 8f}, merged.get("only.weight"));
    }

    @Test
    void emptyDirFailsWithClearMessage() {
        IOException e = assertThrows(IOException.class, () -> SafetensorsLoader.loadDir(dir));
        assertTrue(e.getMessage().contains("缺少权重文件"), e.getMessage());
    }

    // ─── 测试数据构造：最小 safetensors 写入器（F32 only） ───

    /** 写一个合法 safetensors 文件：8B header 长度 + header JSON + 裸 f32 数据 */
    private void writeShard(String fileName, Map<String, float[]> tensors) throws IOException {
        StringBuilder json = new StringBuilder("{");
        long offset = 0;
        int i = 0;
        for (Map.Entry<String, float[]> e : new LinkedHashMap<>(tensors).entrySet()) {
            if (i++ > 0) {
                json.append(',');
            }
            long bytes = 4L * e.getValue().length;
            json.append('"').append(e.getKey()).append("\":{\"dtype\":\"F32\",\"shape\":[")
                    .append(e.getValue().length).append("],\"data_offsets\":[")
                    .append(offset).append(',').append(offset + bytes).append("]}");
            offset += bytes;
        }
        json.append('}');
        byte[] header = json.toString().getBytes(StandardCharsets.UTF_8);

        Path file = dir.resolve(fileName);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE,
                StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length);
            len.flip();
            ch.write(len);
            ch.write(ByteBuffer.wrap(header));
            for (float[] data : tensors.values()) {
                ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                buf.asFloatBuffer().put(data); // 写视图缓冲，不影响 buf 的 position/limit
                ch.write(buf);
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
}
