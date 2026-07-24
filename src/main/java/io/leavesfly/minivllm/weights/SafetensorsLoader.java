package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.math.Bf16;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
 * 2. dtype 支持 F32 / BF16 / F16：BF16 是 HuggingFace LLM 权重的主流格式（如 Qwen3），
 *    半精度转 F32 的位运算见 {@link Bf16}。
 * 3. 大文件用 FileChannel 按 tensor 分段读取（而非 readAllBytes 一次性载入）：
 *    规避 byte[] 2GB 上限，也把内存峰值从"文件两倍"降到"单个 tensor 两倍"。
 */
public final class SafetensorsLoader {

    private SafetensorsLoader() {
    }

    /**
     * 加载 safetensors 文件，返回 tensor 名 -> float[] 数据（BF16/F16 自动转 F32）。
     */
    public static Map<String, float[]> load(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            // 1. 读 8 字节 header 长度（little-endian uint64）
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, lenBuf, 0);
            lenBuf.flip();
            long headerLen = lenBuf.getLong();
            if (headerLen <= 0 || headerLen > ch.size()) {
                throw new IOException("非法 safetensors header 长度: " + headerLen);
            }

            // 2. 读 header JSON 并解析
            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
            readFully(ch, headerBuf, 8);
            String headerJson = new String(headerBuf.array(), StandardCharsets.UTF_8);
            Map<String, Object> header = SimpleJson.parseObject(headerJson);

            // 3. 数据区起始偏移
            long dataBase = 8 + headerLen;
            Map<String, float[]> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : header.entrySet()) {
                String name = e.getKey();
                if ("__metadata__".equals(name)) {
                    continue; // 元数据跳过
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> info = (Map<String, Object>) e.getValue();
                String dtype = (String) info.get("dtype");
                @SuppressWarnings("unchecked")
                List<Object> offsets = (List<Object>) info.get("data_offsets");
                long start = ((Number) offsets.get(0)).longValue();
                long end = ((Number) offsets.get(1)).longValue();
                tensors.put(name, readTensor(ch, dataBase + start, end - start, dtype, name));
            }
            return tensors;
        }
    }

    /**
     * 加载 safetensors，返回 tensor 名 -> bf16 位（short[]）。
     * BF16 张量原样读取（零转换、更省内存/更快）；F32/F16 张量截断为 bf16 位。
     * 供 bf16 常驻推理路径使用：权重以 bf16 存储，点积时逐元素加宽回 f32，算术与 F32 加载一致。
     */
    public static Map<String, short[]> loadBf16Bits(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, lenBuf, 0);
            lenBuf.flip();
            long headerLen = lenBuf.getLong();
            if (headerLen <= 0 || headerLen > ch.size()) {
                throw new IOException("非法 safetensors header 长度: " + headerLen);
            }
            ByteBuffer headerBuf = ByteBuffer.allocate((int) headerLen);
            readFully(ch, headerBuf, 8);
            String headerJson = new String(headerBuf.array(), StandardCharsets.UTF_8);
            Map<String, Object> header = SimpleJson.parseObject(headerJson);
            long dataBase = 8 + headerLen;
            Map<String, short[]> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : header.entrySet()) {
                String name = e.getKey();
                if ("__metadata__".equals(name)) {
                    continue;
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> info = (Map<String, Object>) e.getValue();
                String dtype = (String) info.get("dtype");
                @SuppressWarnings("unchecked")
                List<Object> offsets = (List<Object>) info.get("data_offsets");
                long start = ((Number) offsets.get(0)).longValue();
                long end = ((Number) offsets.get(1)).longValue();
                tensors.put(name, readTensorBf16(ch, dataBase + start, end - start, dtype, name));
            }
            return tensors;
        }
    }

    /** 按 dtype 读取一个 tensor 并转为 bf16 位（short[]） */
    private static short[] readTensorBf16(FileChannel ch, long absOffset, long byteLen,
                                          String dtype, String name) throws IOException {
        int elemBytes;
        switch (dtype) {
            case "F32": elemBytes = 4; break;
            case "BF16": case "F16": elemBytes = 2; break;
            default:
                throw new IOException("暂不支持 dtype=" + dtype + ", tensor=" + name);
        }
        if (byteLen % elemBytes != 0) {
            throw new IOException("tensor " + name + " 字节数 " + byteLen + " 与 dtype " + dtype + " 不对齐");
        }
        int n = (int) (byteLen / elemBytes);
        short[] data = new short[n];
        ByteBuffer chunk = ByteBuffer.allocate((int) Math.min(byteLen, 1 << 22));
        int idx = 0;
        long pos = absOffset;
        long remain = byteLen;
        while (remain > 0) {
            int want = (int) Math.min(remain, chunk.capacity());
            chunk.clear();
            chunk.limit(want);
            readFully(ch, chunk, pos);
            chunk.flip();
            int elems = want / elemBytes;
            ByteBuffer le = chunk.order(ByteOrder.LITTLE_ENDIAN);
            switch (dtype) {
                case "BF16":
                    for (int i = 0; i < elems; i++) {
                        data[idx + i] = le.getShort();
                    }
                    break;
                case "F16":
                    for (int i = 0; i < elems; i++) {
                        float f = Bf16.f16ToFloat(le.getShort() & 0xFFFF);
                        data[idx + i] = (short) (Float.floatToIntBits(f) >>> 16); // f32 -> bf16 截断
                    }
                    break;
                default: // F32
                    for (int i = 0; i < elems; i++) {
                        data[idx + i] = (short) (le.getInt() >>> 16); // f32 -> bf16 截断
                    }
                    break;
            }
            idx += elems;
            pos += want;
            remain -= want;
        }
        return data;
    }

    /** 按 dtype 读取一个 tensor 并转为 F32 */
    private static float[] readTensor(FileChannel ch, long absOffset, long byteLen,
                                      String dtype, String name) throws IOException {
        int elemBytes;
        switch (dtype) {
            case "F32": elemBytes = 4; break;
            case "BF16": case "F16": elemBytes = 2; break;
            default:
                throw new IOException("暂不支持 dtype=" + dtype + ", tensor=" + name);
        }
        if (byteLen % elemBytes != 0) {
            throw new IOException("tensor " + name + " 字节数 " + byteLen + " 与 dtype " + dtype + " 不对齐");
        }
        int n = (int) (byteLen / elemBytes);
        float[] data = new float[n];

        // 分块读取，避免单个 tensor 的临时字节缓冲过大
        ByteBuffer chunk = ByteBuffer.allocate((int) Math.min(byteLen, 1 << 22)); // 4MB 块
        int idx = 0;
        long pos = absOffset;
        long remain = byteLen;
        while (remain > 0) {
            int want = (int) Math.min(remain, chunk.capacity());
            chunk.clear();
            chunk.limit(want);
            readFully(ch, chunk, pos);
            chunk.flip();
            int elems = want / elemBytes;
            switch (dtype) {
                case "F32":
                    chunk.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data, idx, elems);
                    break;
                case "BF16":
                    readBf16Into(chunk, data, idx, elems);
                    break;
                default: // "F16"
                    readF16Into(chunk, data, idx, elems);
                    break;
            }
            idx += elems;
            pos += want;
            remain -= want;
        }
        return data;
    }

    /** BF16 段转换：写入 out[outOff, outOff+n) */
    private static void readBf16Into(ByteBuffer buf, float[] out, int outOff, int n) {
        ByteBuffer le = buf.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            out[outOff + i] = Bf16.bf16ToFloat(le.getShort() & 0xFFFF);
        }
    }

    /** F16 段转换：写入 out[outOff, outOff+n) */
    private static void readF16Into(ByteBuffer buf, float[] out, int outOff, int n) {
        ByteBuffer le = buf.order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            out[outOff + i] = Bf16.f16ToFloat(le.getShort() & 0xFFFF);
        }
    }

    /** 从 channel 的指定位置读满缓冲区 */
    private static void readFully(FileChannel ch, ByteBuffer buf, long pos) throws IOException {
        long p = pos;
        while (buf.hasRemaining()) {
            int r = ch.read(buf, p);
            if (r < 0) {
                throw new IOException("文件提前结束");
            }
            p += r;
        }
    }
}
