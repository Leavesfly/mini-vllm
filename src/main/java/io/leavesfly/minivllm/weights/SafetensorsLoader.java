package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.json.SimpleJson;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * SafetensorsLoader —— 纯 Java 解析 .safetensors 权重文件。
 *
 * Safetensors 文件格式（简单且高效）：
 *   [8 字节 little-endian uint64] header JSON 字节数 N
 *   [N 字节]                    header JSON：{tensor_name: {dtype, shape, data_offsets}, ...}
 *   [剩余]                      原始张量数据，按 data_offsets 定位
 *
 * 学习要点：
 * 1. 相比 PyTorch .pt（pickle，零依赖下几乎无法解析），safetensors 用明文 JSON header + 裸数据，
 *    非常适合纯 Java 解析——这正是本项目选 safetensors 作为权重载体的原因。
 * 2. F32 张量每个元素 4 字节，little-endian；用 ByteBuffer.asFloatBuffer() 高效读取。
 * 3. 只需一个手写 JSON 解析器 + 几十行二进制处理，就完成了"零依赖加载深度学习权重"。
 */
public final class SafetensorsLoader {

    private SafetensorsLoader() {
    }

    /**
     * 加载 safetensors 文件，返回 tensor 名 -> float[] 数据。
     * 当前仅支持 F32（float32）dtype，学习用微模型足够。
     */
    public static Map<String, float[]> load(Path path) throws IOException {
        byte[] all = Files.readAllBytes(path);

        // 1. 读 8 字节 header 长度（little-endian uint64）
        long headerLen = ByteBuffer.wrap(all, 0, 8).order(ByteOrder.LITTLE_ENDIAN).getLong();
        if (headerLen <= 0 || headerLen > all.length) {
            throw new IOException("非法 safetensors header 长度: " + headerLen);
        }

        // 2. 解析 header JSON
        String headerJson = new String(all, 8, (int) headerLen, java.nio.charset.StandardCharsets.UTF_8);
        Map<String, Object> header = SimpleJson.parseObject(headerJson);

        // 3. 数据区起始偏移
        int dataBase = 8 + (int) headerLen;
        Map<String, float[]> tensors = new LinkedHashMap<>();

        for (Map.Entry<String, Object> e : header.entrySet()) {
            String name = e.getKey();
            if ("__metadata__".equals(name)) {
                continue; // 元数据跳过
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> info = (Map<String, Object>) e.getValue();
            String dtype = (String) info.get("dtype");
            if (!"F32".equals(dtype)) {
                throw new IOException("暂不支持 dtype=" + dtype + "（仅支持 F32）, tensor=" + name);
            }
            @SuppressWarnings("unchecked")
            List<Object> offsets = (List<Object>) info.get("data_offsets");
            long start = ((Number) offsets.get(0)).longValue();
            long end = ((Number) offsets.get(1)).longValue();
            int n = (int) ((end - start) / 4L); // F32 每元素 4 字节

            float[] data = new float[n];
            ByteBuffer bb = ByteBuffer.wrap(all, (int) (dataBase + start), n * 4)
                    .order(ByteOrder.LITTLE_ENDIAN);
            bb.asFloatBuffer().get(data);
            tensors.put(name, data);
        }
        return tensors;
    }
}
