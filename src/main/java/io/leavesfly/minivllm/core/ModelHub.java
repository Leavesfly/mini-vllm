package io.leavesfly.minivllm.core;

import io.leavesfly.minivllm.family.LoadedModel;
import io.leavesfly.minivllm.family.ModelRegistry;
import io.leavesfly.minivllm.family.Precision;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ModelHub —— 多模型服务注册表（对照 vLLM 的多模型部署：一个进程按 model 字段路由）。
 *
 * 学习要点：
 * 1. 注册与加载分离：register() 只记下「模型 id → 模型目录」，真正的权重加载
 *    发生在第一次请求该模型时（get()）。这样启动只需付默认模型的加载代价，
 *    Qwen3-0.6B 这类 2.4GB 权重不会拖慢没用到它的场景。
 * 2. 每个模型一套独立的 KV 池 + 引擎线程（见 {@link ModelRuntime}），
 *    因此不同模型的请求互不排队、互不污染 KV cache。
 * 3. 懒加载在整个 hub 上串行（synchronized）：并行加载多个数十亿字节的权重
 *    只会加剧内存峰值与磁盘争抢，串行反而更稳。
 * 4. 代价是常驻内存随「用过的模型数」累加（本类不做换出），
 *    同时服务 MiniMind3 与 Qwen3-0.6B 建议 -Xmx6g。
 */
public final class ModelHub {

    /** KV 池最小 block 数：过小会让长 prompt 直接分配失败 */
    private static final int MIN_NUM_BLOCKS = 1024;
    /** 兼容旧客户端与 curl 示例的别名：一律路由到默认模型 */
    private static final List<String> DEFAULT_ALIASES = List.of("mini-vllm", "default", "auto");

    /**
     * 全局装配参数（命令行给出，对所有模型生效）。
     *
     * @param precision  权重常驻精度
     * @param random     true 时随机初始化权重（仅验证流程，不读权重文件）
     * @param maxSeqLen  上下文窗口上限
     * @param maxNumSeqs 并发序列数；≤0 表示用模型家族的建议值
     * @param numBlocks  KV 池 block 数；≤0 表示按模型规模自动推算
     * @param verbose    引擎是否打印每步调度信息
     * @param seed       采样随机种子
     */
    public record Options(Precision precision, boolean random, int maxSeqLen,
                          int maxNumSeqs, int numBlocks, boolean verbose, long seed) {
    }

    private final Options options;
    /** id → 模型目录（懒加载来源）；仅在启动装配阶段写入 */
    private final Map<String, Path> sources = new LinkedHashMap<>();
    /** id → 已就绪运行时；HTTP 线程并发读，加载时串行写 */
    private final Map<String, ModelRuntime> runtimes = new ConcurrentHashMap<>();
    private final List<String> ids = new ArrayList<>();
    private String defaultId;

    public ModelHub(Options options) {
        this.options = options;
    }

    /** 全局装配参数（遗留路径自行装配运行时时复用同一份配置） */
    public Options options() {
        return options;
    }

    /** 注册一个懒加载模型：id 通常取模型目录名，首个注册的模型即默认模型 */
    public void register(String id, Path modelDir) {
        sources.put(id, modelDir);
        remember(id);
    }

    /** 登记一个已构造好的运行时（遗留学习路径：随机微模型 / GPT-3 预设） */
    public void adopt(ModelRuntime runtime) {
        runtimes.put(runtime.id(), runtime);
        remember(runtime.id());
    }

    /** 按注册顺序返回全部模型 id（首项为默认模型） */
    public List<String> ids() {
        return List.copyOf(ids);
    }

    public String defaultId() {
        return defaultId;
    }

    public boolean isLoaded(String id) {
        return runtimes.containsKey(id);
    }

    /** 已就绪的运行时，尚未加载时返回 null（供启动信息打印，不触发加载） */
    public ModelRuntime runtimeIfLoaded(String id) {
        return runtimes.get(id);
    }

    /** 启动预热：把默认模型提前加载好，避免首个请求等在权重 IO 上 */
    public ModelRuntime preload(String id) throws IOException {
        return get(id);
    }

    /**
     * 取一个模型的运行时：命中缓存直接返回，否则加载后缓存。
     *
     * @param requestedId 请求体的 model 字段；空值或别名（mini-vllm / default / auto）走默认模型
     * @throws IllegalArgumentException 该 id 未注册（API 层据此返回 400）
     */
    public ModelRuntime get(String requestedId) throws IOException {
        String id = resolveId(requestedId);
        ModelRuntime cached = runtimes.get(id);
        if (cached != null) {
            return cached;
        }
        synchronized (this) {
            ModelRuntime again = runtimes.get(id); // 双重检查：并发请求同一模型只加载一次
            if (again != null) {
                return again;
            }
            System.out.println("加载模型 [" + id + "]: " + sources.get(id));
            LoadedModel loaded = new ModelRegistry().load(sources.get(id), options.precision(),
                    options.random(), options.maxSeqLen());
            ModelRuntime runtime = assemble(id, loaded, options);
            runtimes.put(id, runtime);
            return runtime;
        }
    }

    /**
     * 把加载产物装配成可服务的运行时：KV 池 → 引擎 → 启动引擎线程。
     * 遗留学习路径（非 HF 模型目录）也复用本方法，保证两条路径装配一致。
     */
    public static ModelRuntime assemble(String id, LoadedModel loaded, Options options) {
        ModelConfig cfg = loaded.config();
        int maxNumSeqs = options.maxNumSeqs() > 0 ? options.maxNumSeqs() : loaded.defaultMaxSeqs();
        int numBlocks = options.numBlocks() > 0 ? options.numBlocks() : autoNumBlocks(cfg, maxNumSeqs);

        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize(), loaded.kvDim());
        long kvBytes = (long) numBlocks * cfg.blockSize() * loaded.kvDim() * 2 * 4;
        System.out.printf("[%s] KV 池: %d blocks × blockSize=%d × kvDim=%d（满载约 %.1f GB）%n",
                id, numBlocks, cfg.blockSize(), loaded.kvDim(), kvBytes / 1e9);

        LLMEngine engine = new LLMEngine(loaded.model(), kvMgr, loaded.tokenizer(),
                maxNumSeqs, loaded.eosTokens(), options.seed());
        engine.setVerbose(options.verbose());
        engine.start();
        return new ModelRuntime(id, loaded, engine, maxNumSeqs, numBlocks);
    }

    /** 按「并发数 × 层数 × 每序列块数」估算 KV 池规模，保证满载并发不会中途 OOM */
    private static int autoNumBlocks(ModelConfig cfg, int maxNumSeqs) {
        int blocksPerSeq = (cfg.maxSeqLen() + cfg.blockSize() - 1) / cfg.blockSize();
        return Math.max(MIN_NUM_BLOCKS, maxNumSeqs * cfg.nLayer() * blocksPerSeq);
    }

    private void remember(String id) {
        if (!ids.contains(id)) {
            ids.add(id);
        }
        if (defaultId == null) {
            defaultId = id;
        }
    }

    /** id 解析：精确匹配 → 忽略大小写 → 别名/空值回退默认模型 */
    private String resolveId(String requestedId) {
        if (requestedId == null || requestedId.isBlank()) {
            return defaultId;
        }
        if (ids.contains(requestedId)) {
            return requestedId;
        }
        for (String id : ids) {
            if (id.equalsIgnoreCase(requestedId)) {
                return id;
            }
        }
        if (DEFAULT_ALIASES.contains(requestedId.toLowerCase())) {
            return defaultId;
        }
        throw new IllegalArgumentException("未知模型 '" + requestedId + "'，可用模型: " + ids);
    }
}
