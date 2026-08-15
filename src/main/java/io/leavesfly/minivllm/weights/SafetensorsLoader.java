package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.math.Bf16;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

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
 * 4. 大模型（如 Qwen3-4B）权重拆成多个分片，目录下以 model.safetensors.index.json
 *    的 weight_map 记录「tensor 名 -> 分片文件」路由；{@link #loadDir} 自动识别
 *    单文件/多分片，分片间并行读取，合并后与 weight_map 全量校验（防不完整下载）。
 */
public final class SafetensorsLoader {

    /** 单文件权重名 */
    public static final String SINGLE_FILE = "model.safetensors";
    /** 多分片索引文件名 */
    public static final String INDEX_JSON = "model.safetensors.index.json";

    private SafetensorsLoader() {
    }

    // ─── 目录级入口（自动识别单文件 / 多分片） ───

    /**
     * 加载模型目录的权重，返回 tensor 名 -> float[]（BF16/F16 自动转 F32）。
     * 目录含 {@link #INDEX_JSON} 时按分片路由并行加载，否则读单文件 {@link #SINGLE_FILE}。
     */
    public static Map<String, float[]> loadDir(Path modelDir) throws IOException {
        Map<String, String> weightMap = readWeightMap(modelDir);
        if (weightMap == null) {
            return load(modelDir.resolve(SINGLE_FILE));
        }
        return loadShards(modelDir, shardList(weightMap), weightMap.keySet(), false);
    }

    /**
     * {@link #loadDir} 的 bf16 常驻版：返回 tensor 名 -> bf16 位（short[]），
     * 供 bf16 常驻与 int8 量化加载路径使用。
     */
    public static Map<String, short[]> loadDirBf16Bits(Path modelDir) throws IOException {
        Map<String, String> weightMap = readWeightMap(modelDir);
        if (weightMap == null) {
            return loadBf16Bits(modelDir.resolve(SINGLE_FILE));
        }
        return loadShards(modelDir, shardList(weightMap), weightMap.keySet(), true);
    }

    /**
     * 目录全部权重文件的字节总和（单文件或多分片），供加载前的堆内存预检。
     * 注意含 8 字节长度头与 header JSON，但占比可忽略。
     */
    public static long weightFileBytes(Path modelDir) throws IOException {
        Map<String, String> weightMap = readWeightMap(modelDir);
        if (weightMap == null) {
            return Files.size(modelDir.resolve(SINGLE_FILE));
        }
        long total = 0;
        for (String shard : shardList(weightMap)) {
            total += Files.size(modelDir.resolve(shard));
        }
        return total;
    }

    /**
     * 读取分片索引的 weight_map（tensor 名 -> 分片文件名）；单文件目录返回 null。
     * 既无索引也无单文件时直接报错（比 NoSuchFileException 更易懂的提示）。
     */
    static Map<String, String> readWeightMap(Path modelDir) throws IOException {
        Path index = modelDir.resolve(INDEX_JSON);
        if (!Files.isRegularFile(index)) {
            if (!Files.isRegularFile(modelDir.resolve(SINGLE_FILE))) {
                throw new IOException("模型目录缺少权重文件（既无 " + SINGLE_FILE
                        + " 也无 " + INDEX_JSON + "）: " + modelDir);
            }
            return null;
        }
        Object wm = SimpleJson.parseObject(Files.readString(index)).get("weight_map");
        if (!(wm instanceof Map)) {
            throw new IOException(INDEX_JSON + " 缺少 weight_map: " + index);
        }
        Map<String, String> weightMap = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) wm).entrySet()) {
            weightMap.put(String.valueOf(e.getKey()), String.valueOf(e.getValue()));
        }
        return weightMap;
    }

    /** weight_map 引用的分片文件去重排序（文件名自带 00001-of-N 序号，排序即确定顺序） */
    static List<String> shardList(Map<String, String> weightMap) {
        return List.copyOf(new TreeSet<>(weightMap.values()));
    }

    /**
     * 并行加载各分片并合并。每个分片是标准 safetensors 文件（header 自描述 tensor 清单），
     * 因此读取逻辑复用单文件路径；合并后与 weight_map 全量校验——两者不一致说明
     * 下载不完整或索引损坏，必须在加载期暴露而非带病运行。
     */
    @SuppressWarnings("unchecked")
    private static <T> Map<String, T> loadShards(Path modelDir, List<String> shards,
                                                 Set<String> expectedTensors, boolean bf16)
            throws IOException {
        int threads = Math.min(shards.size(), Runtime.getRuntime().availableProcessors());
        System.out.printf("检测到 %d 个权重分片，%d 线程并行加载%n", shards.size(), threads);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Map<String, ?>>> futures = new ArrayList<>();
            for (String shard : shards) {
                Path file = modelDir.resolve(shard);
                futures.add(pool.submit(() -> bf16 ? loadBf16Bits(file) : load(file)));
            }
            Map<String, T> merged = new LinkedHashMap<>();
            for (Future<Map<String, ?>> f : futures) {
                try {
                    for (Map.Entry<String, ?> e : f.get().entrySet()) {
                        merged.put(e.getKey(), (T) e.getValue());
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IOException("分片加载被中断", e);
                } catch (ExecutionException e) {
                    Throwable cause = e.getCause();
                    throw cause instanceof IOException io ? io : new IOException("分片加载失败", cause);
                }
            }
            verifyAgainstIndex(merged.keySet(), expectedTensors);
            return merged;
        } finally {
            pool.shutdown();
        }
    }

    /** 合并结果与 weight_map 全量比对：缺失或索引外张量都视为损坏 */
    static void verifyAgainstIndex(Set<String> actual, Set<String> expected) throws IOException {
        if (actual.equals(expected)) {
            return;
        }
        Set<String> missing = new TreeSet<>(expected);
        missing.removeAll(actual);
        Set<String> extra = new TreeSet<>(actual);
        extra.removeAll(expected);
        throw new IOException("分片内容与 weight_map 不一致：缺失 " + missing.size() + " 个张量"
                + (missing.isEmpty() ? "" : "（如 " + missing.iterator().next() + "）")
                + "，索引外多出 " + extra.size() + " 个");
    }

    /** 张量元信息（dtype + 数据区偏移），header 解析的中间产物 */
    record TensorInfo(String dtype, long start, long end) {
    }

    /**
     * 解析 safetensors 文件 header，返回 tensor 名 -> 元信息与数据区起始偏移。
     * load / loadBf16Bits / {@link MmapWeights} 共用的格式解析入口。
     */
    static ParsedHeader readHeader(FileChannel ch) throws IOException {
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

        // 3. 逐 tensor 提取 dtype 与 data_offsets
        long dataBase = 8 + headerLen;
        Map<String, TensorInfo> tensors = new LinkedHashMap<>();
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
            tensors.put(name, new TensorInfo(dtype, dataBase + start, dataBase + end));
        }
        return new ParsedHeader(tensors, dataBase);
    }

    /** header 解析结果：tensor 索引 + 数据区起始偏移 */
    record ParsedHeader(Map<String, TensorInfo> tensors, long dataBase) {
    }

    /**
     * 加载 safetensors 文件，返回 tensor 名 -> float[] 数据（BF16/F16 自动转 F32）。
     */
    public static Map<String, float[]> load(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ParsedHeader header = readHeader(ch);
            Map<String, float[]> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, TensorInfo> e : header.tensors().entrySet()) {
                TensorInfo t = e.getValue();
                tensors.put(e.getKey(), readTensor(ch, t.start(), t.end() - t.start(), t.dtype(), e.getKey()));
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
            ParsedHeader header = readHeader(ch);
            Map<String, short[]> tensors = new LinkedHashMap<>();
            for (Map.Entry<String, TensorInfo> e : header.tensors().entrySet()) {
                TensorInfo t = e.getValue();
                tensors.put(e.getKey(), readTensorBf16(ch, t.start(), t.end() - t.start(), t.dtype(), e.getKey()));
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
