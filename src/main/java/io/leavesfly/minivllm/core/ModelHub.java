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
     * @param kvInt8     KV cache 是否 INT8 量化存储（对照 vLLM --kv-cache-dtype int8）
     * @param verbose    引擎是否打印每步调度信息
     * @param seed       采样随机种子
     * @param speculativeK 投机采样草稿长度（0 关闭；仅 greedy 单序列生效）
     * @param draftModel 草稿模型目录或已注册 id（null 时回退 prompt-lookup 自投机；
     *                   需与目标模型同词表，且 speculativeK > 0 才生效）
     */
    public record Options(Precision precision, boolean random, int maxSeqLen,
                          int maxNumSeqs, int numBlocks, boolean kvInt8, boolean verbose, long seed,
                          int speculativeK, String draftModel) {
    }

    private final Options options;
    /** id → 模型目录（懒加载来源）；仅在启动装配阶段写入 */
    private final Map<String, Path> sources = new LinkedHashMap<>();
    /** id → 已就绪运行时；HTTP 线程并发读，加载时串行写 */
    private final Map<String, ModelRuntime> runtimes = new ConcurrentHashMap<>();
    private final List<String> ids = new ArrayList<>();
    private String defaultId;
    /** 草稿模型加载缓存（多个目标运行时共享同一份草稿权重，各持独立 KV 池） */
    private LoadedModel draftLoaded;

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

    /** 按注册顺序返回全部已加载的运行时（/metrics 端点用，不触发加载） */
    public List<ModelRuntime> loadedRuntimes() {
        List<ModelRuntime> out = new ArrayList<>();
        for (String id : ids) {
            ModelRuntime r = runtimes.get(id);
            if (r != null) {
                out.add(r);
            }
        }
        return out;
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
            maybeAttachDraft(runtime);
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

        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize(), loaded.kvDim(),
                options.kvInt8());
        int bytesPerElem = options.kvInt8() ? 1 : 4; // int8 每元素 1 字节（另每 token 2 个 scale，可忽略）
        long kvBytes = (long) numBlocks * cfg.blockSize() * loaded.kvDim() * 2 * bytesPerElem;
        System.out.printf("[%s] KV 池: %d blocks × blockSize=%d × kvDim=%d × %s（满载约 %.1f GB）%n",
                id, numBlocks, cfg.blockSize(), loaded.kvDim(),
                options.kvInt8() ? "int8" : "f32", kvBytes / 1e9);

        LLMEngine engine = new LLMEngine(loaded.model(), kvMgr, loaded.tokenizer(),
                maxNumSeqs, loaded.eosTokens(), options.seed());
        engine.setVerbose(options.verbose());
        engine.setSpeculativeK(options.speculativeK());
        engine.start();
        return new ModelRuntime(id, loaded, engine, maxNumSeqs, numBlocks);
    }

    /**
     * 为运行时装配草稿模型（draft-model 投机，对照 vLLM 的 --speculative-model）。
     * 草稿权重全局只加载一份（缓存复用）；每个目标运行时持有独立的草稿 KV 池——
     * 投机仅单序列生效，池按 1 并发估算即可。词表不一致时拒绝接入并告警。
     */
    private void maybeAttachDraft(ModelRuntime runtime) throws IOException {
        String draftRef = options.draftModel();
        if (draftRef == null || options.speculativeK() <= 0) {
            return;
        }
        if (draftLoaded == null) {
            Path dir = sources.containsKey(draftRef) ? sources.get(draftRef) : Path.of(draftRef);
            System.out.println("加载草稿模型: " + dir);
            draftLoaded = new ModelRegistry().load(dir, options.precision(),
                    options.random(), options.maxSeqLen());
        }
        if (draftLoaded.config().vocabSize() != runtime.loaded().config().vocabSize()) {
            System.out.println("[spec] 草稿模型词表（" + draftLoaded.config().vocabSize() + "）与 ["
                    + runtime.id() + "]（" + runtime.loaded().config().vocabSize() + "）不一致，跳过投机配置");
            return;
        }
        ModelConfig dcfg = draftLoaded.config();
        int draftBlocks = autoNumBlocks(dcfg, 1); // 单序列投机：按 1 并发估算
        KVCacheManager draftKv = new KVCacheManager(draftBlocks, dcfg.blockSize(),
                draftLoaded.kvDim(), options.kvInt8());
        runtime.engine().setDraftModel(draftLoaded.model(), draftKv);
        System.out.printf("[spec] [%s] 接入草稿模型（k=%d，草稿池 %d blocks × blockSize=%d × kvDim=%d）%n",
                runtime.id(), options.speculativeK(), draftBlocks, dcfg.blockSize(), draftLoaded.kvDim());
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
