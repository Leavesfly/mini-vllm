package io.leavesfly.minivllm.weights;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * MmapWeights —— 内存映射（mmap）权重库：权重不落堆，由 OS 页缓存按需调页。
 *
 * 学习要点（对照 vLLM --load-format / llama.cpp 默认 mmap 加载）：
 * 1. {@link FileChannel#map} 把权重文件映射进进程虚拟地址空间，只建立页表、不读数据；
 *    首次访问某页才缺页读盘，之后热页驻留物理内存，内存不足时由 OS 自动逐出——
 *    因此「物理内存 < 模型体积」也能运行，这是堆内全量加载做不到的。
 * 2. 仅支持磁盘 dtype=BF16 的张量：bf16 可零转换地以 {@link ShortBuffer} 视图直读；
 *    F32/F16 需要在线加宽（失去零拷贝意义），拒绝并提示改用 --precision bf16 等堆内路径。
 * 3. 生命周期零成本：算子（Linear/Embedding）持有 ShortBuffer 视图即间接持有
 *    {@link MappedByteBuffer} 强引用，模型存活即映射存活；模型卸载后由 GC Cleaner
 *    释放映射（JDK 无公开 unmap API，学习场景足够）。
 * 4. 限制：单个权重文件 ≤ 2GB（JDK MappedByteBuffer 以 int 寻址）；HF 主流分片满足。
 *
 * 线程安全：所有读取均走 buffer 的绝对位置 get（不移动 position），可并发直读。
 */
public final class MmapWeights {

    /** tensor 名 -> bf16 位视图（只读、绝对位置访问） */
    private final Map<String, ShortBuffer> views;
    /** 映射强引用：防止 MappedByteBuffer 被 GC 提前回收映射 */
    private final List<MappedByteBuffer> mappings;
    /** 映射的文件总字节数（启动日志用） */
    private final long mappedBytes;

    private MmapWeights(Map<String, ShortBuffer> views, List<MappedByteBuffer> mappings,
                        long mappedBytes) {
        this.views = views;
        this.mappings = mappings;
        this.mappedBytes = mappedBytes;
    }

    /**
     * 映射模型目录的全部权重（自动识别单文件 / 多分片）。
     * 只解析 header 与建立页表，不读张量数据——加载近乎瞬时，IO 推迟到首次推理。
     */
    public static MmapWeights open(Path modelDir) throws IOException {
        Map<String, String> weightMap = SafetensorsLoader.readWeightMap(modelDir);
        Map<String, ShortBuffer> views = new LinkedHashMap<>();
        List<MappedByteBuffer> mappings = new ArrayList<>();
        long mappedBytes = 0;
        if (weightMap == null) {
            mappedBytes += mapFile(modelDir.resolve(SafetensorsLoader.SINGLE_FILE), views, mappings);
        } else {
            for (String shard : SafetensorsLoader.shardList(weightMap)) {
                mappedBytes += mapFile(modelDir.resolve(shard), views, mappings);
            }
            // 分片内容与 weight_map 全量比对：下载不完整/索引损坏必须在加载期暴露
            SafetensorsLoader.verifyAgainstIndex(views.keySet(), weightMap.keySet());
        }
        return new MmapWeights(views, mappings, mappedBytes);
    }

    /** 映射单个 safetensors 文件，把其中所有 BF16 张量登记为视图，返回文件字节数 */
    private static long mapFile(Path file, Map<String, ShortBuffer> views,
                                List<MappedByteBuffer> mappings) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = ch.size();
            if (size > Integer.MAX_VALUE) {
                throw new IOException("权重分片超过 2GB 无法 mmap（JDK MappedByteBuffer 限制）: "
                        + file + "（" + (size >> 30) + "GB），请改用 --precision bf16 堆内加载");
            }
            MappedByteBuffer mapped = ch.map(FileChannel.MapMode.READ_ONLY, 0, size);
            mappings.add(mapped);
            SafetensorsLoader.ParsedHeader header = SafetensorsLoader.readHeader(ch);
            for (Map.Entry<String, SafetensorsLoader.TensorInfo> e : header.tensors().entrySet()) {
                String name = e.getKey();
                SafetensorsLoader.TensorInfo t = e.getValue();
                long byteLen = t.end() - t.start();
                if (!"BF16".equals(t.dtype())) {
                    throw new IOException("mmap 仅支持磁盘 dtype=BF16 的权重，张量 " + name
                            + " dtype=" + t.dtype() + "（文件 " + file.getFileName()
                            + "）。请改用 --precision bf16/int8/int4 堆内加载");
                }
                if (byteLen % 2 != 0 || (t.start() & 1) != 0) {
                    throw new IOException("张量 " + name + " 偏移/长度未按 2 字节对齐，无法建立 short 视图"
                            + "（正规导出会用空格把 header 补齐到 8 字节对齐，此文件可能已损坏）");
                }
                // 视图 buffer 持有 mapped 的引用链：模型可达即映射存活
                ByteBuffer region = mapped.duplicate();
                region.order(ByteOrder.LITTLE_ENDIAN);
                region.position((int) t.start());
                region.limit((int) t.end());
                views.put(name, region.asShortBuffer().asReadOnlyBuffer());
            }
            return size;
        }
    }

    /** 张量 bf16 位视图（绝对位置访问，线程安全）；不存在返回 null */
    public ShortBuffer view(String name) {
        return views.get(name);
    }

    /** 张量元素个数；不存在返回 -1 */
    public int length(String name) {
        ShortBuffer v = views.get(name);
        return v == null ? -1 : v.capacity();
    }

    /** 全部张量名 */
    public Set<String> keys() {
        return views.keySet();
    }

    /** 映射的文件总字节数 */
    public long mappedBytes() {
        return mappedBytes;
    }

    /**
     * 小张量（如 RmsNorm gamma）一次性读进堆内 short[]：
     * 这类张量每次前向都全量访问，常驻堆避免反复缺页，占比也可忽略。
     */
    public short[] readShorts(String name) throws IOException {
        ShortBuffer v = views.get(name);
        if (v == null) {
            throw new IOException("缺少权重: " + name);
        }
        short[] out = new short[v.capacity()];
        v.get(0, out, 0, out.length);
        return out;
    }
}
