package io.leavesfly.minivllm;

import com.sun.net.httpserver.HttpServer;
import io.leavesfly.minivllm.core.ModelHub;
import io.leavesfly.minivllm.core.ModelRuntime;
import io.leavesfly.minivllm.family.LoadedModel;
import io.leavesfly.minivllm.family.Precision;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.TransformerModel;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.ByteTokenizer;
import io.leavesfly.minivllm.tokenizer.PlainTextTemplate;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.weights.ModelDownloader;
import io.leavesfly.minivllm.weights.ModelLoader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;
import io.leavesfly.minivllm.api.OpenAiHandler;
import io.leavesfly.minivllm.api.WebUiHandler;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.stream.Stream;

/**
 * MiniVllmServer —— 学习型 vLLM 引擎入口。
 *
 * 启动流程：
 *   1. 解析参数（端口 / 权重路径 / 随机初始化 / 并发数 / block 数）
 *   2. 注册可选模型到 {@link ModelHub}（默认扫描 models/ 下的每个 HF 模型目录）
 *   3. 预热默认模型（其余模型首次请求时才加载，含 KV 池 + 引擎线程）
 *   4. 启动 JDK HttpServer，注册 OpenAI 兼容 API 与 Web 对话页面
 *
 * 多模型：每个模型一套独立的 KV 池与引擎，请求体的 model 字段决定路由到哪一套，
 * Web 页面右上角的下拉框就是在切换这个字段。同时服务 MiniMind3 与 Qwen3-0.6B 建议 -Xmx6g。
 *
 * 用法示例：
 *   默认（注册 models/ 下全部模型，预热 minimind-3-agent-512）：
 *     java -jar mini-vllm.jar --port 8080
 *   随机初始化微模型（无需权重，验证流程）：
 *     java -jar mini-vllm.jar --random --port 8080
 *   加载真实权重：
 *     java -jar mini-vllm.jar --weights ./model.safetensors --port 8080
 *   只服务指定模型（可重复传入或逗号分隔多个，首个为默认）：
 *     java -jar mini-vllm.jar --model-dir ./models/Qwen3-0.6B --port 8080
 *     java -jar mini-vllm.jar --model-dir ./models/minimind-3-agent-512,./models/Qwen3-0.6B
 *   KV cache INT8 量化（容量与带宽降为 f32 的约 1/4，默认 f32）：
 *     java -jar mini-vllm.jar --kv-cache-dtype int8 --port 8080
 *   权重量化常驻（int4 带宽最省、int8 次之，默认 f32；--bf16 等价 --precision bf16）：
 *     java -jar mini-vllm.jar --precision int4 --port 8080
 *   权重 mmap 按需调页（不占堆，物理内存小于模型体积也能跑；要求磁盘权重为 bf16）：
 *     java -jar mini-vllm.jar --precision mmap --port 8080
 *   prompt-lookup 投机采样（greedy 单请求无损加速，默认关闭）：
 *     java -jar mini-vllm.jar --speculative-k 4 --port 8080
 *   草稿模型投机采样（小模型起草、大模型验证，需同词表；对照 vLLM --speculative-model）：
 *     java -jar mini-vllm.jar --speculative-k 4 --draft-model ./models/Qwen3-0.6B --port 8080
 *
 * 测试：
 *   curl -X POST http://localhost:8080/v1/chat/completions \
 *        -H "Content-Type: application/json" \
 *        -d '{"model":"mini-vllm","messages":[{"role":"user","content":"Hello"}],"stream":true}'
 */
public final class MiniVllmServer {

    // ─── 默认配置常量 ───

    private static final int DEFAULT_PORT = 8080;
    private static final long DEFAULT_SEED = 12345L;
    private static final int DEFAULT_MAX_SEQ_LEN = 2048;
    /** 默认模型扫描根目录：其下每个含 config.json 的子目录都注册为一个可选模型 */
    private static final String MODELS_ROOT = "models";
    /** 首选默认模型：项目自带的 MiniMind3（~29M 参数，CPU 上秒级加载） */
    private static final String PREFERRED_MODEL = "minimind-3-agent-512";
    /** models/ 下无任何模型时的下载回退目标 */
    private static final String DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B";
    /** 遗留学习路径（随机微模型 / GPT-3 预设）对外暴露的模型 id */
    private static final String LEGACY_MODEL_ID = "mini-vllm";
    private static final int DEFAULT_LEARNING_MAX_SEQS = 8;


    // ─── 服务器配置（命令行参数解析结果） ───

    private static final class ServerConfig {
        int port = DEFAULT_PORT;
        String weightsPath = null;
        String tokenizerDir = null;
        final List<String> modelDirs = new ArrayList<>();
        boolean random = false;
        int maxNumSeqs = -1;
        int numBlocks = -1;
        int maxSeqLen = DEFAULT_MAX_SEQ_LEN;
        boolean verbose = true;
        boolean gpt3 = false;
        boolean bf16 = false;
        /** 权重常驻精度：f32（默认）/ bf16 / int8 / int4 / mmap */
        String precision = "f32";
        /** KV cache dtype："auto"/"f32" 保持 f32（默认），"int8" 启用量化存储 */
        String kvCacheDtype = "auto";
        /** 投机采样草稿长度（0 关闭；仅 greedy 单序列 decode 生效） */
        int speculativeK = 0;
        /** 草稿模型目录或已注册 id（null 时回退 prompt-lookup 自投机） */
        String draftModel = null;
        String modelRepo = DEFAULT_MODEL_REPO;
        String mirror = System.getenv("MINIVLLM_MIRROR");

        /** 遗留学习模式：没指定模型目录，而是直接构造微模型 / 读单个权重文件 */
        boolean isLegacyMode() {
            return modelDirs.isEmpty() && (random || gpt3 || weightsPath != null);
        }
    }

    public static void main(String[] args) throws Exception {
        ServerConfig config = parseArgs(args);
        printRuntimeDiagnostics();

        ModelHub hub = new ModelHub(new ModelHub.Options(
                parsePrecision(config), config.random,
                config.maxSeqLen, config.maxNumSeqs, config.numBlocks,
                "int8".equalsIgnoreCase(config.kvCacheDtype), config.verbose, DEFAULT_SEED,
                config.speculativeK, config.draftModel));

        if (config.isLegacyMode()) {
            hub.adopt(ModelHub.assemble(LEGACY_MODEL_ID, loadLegacyModel(config), hub.options()));
        } else {
            for (Path dir : resolveModelDirs(config)) {
                hub.register(dir.getFileName().toString(), dir);
            }
            // 默认模型启动即就绪，其余模型等 Web 页面 / API 首次选中时再加载
            hub.preload(hub.defaultId());
        }

        startHttpServer(hub, config.port);
        printBanner(config.port, hub);
    }


    // ─── 参数解析 ───

    private static ServerConfig parseArgs(String[] args) {
        ServerConfig config = new ServerConfig();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port" -> config.port = Integer.parseInt(args[++i]);
                case "--weights" -> config.weightsPath = args[++i];
                case "--tokenizer-dir" -> config.tokenizerDir = args[++i];
                // 可重复传入，也可一次传逗号分隔的多个目录；首个为默认模型
                case "--model-dir", "--model-dirs" -> config.modelDirs.addAll(splitDirs(args[++i]));
                case "--max-seq-len" -> config.maxSeqLen = Integer.parseInt(args[++i]);
                case "--random" -> config.random = true;
                case "--gpt3" -> config.gpt3 = true;
                case "--bf16" -> config.bf16 = true;
                // 权重常驻精度：f32 / bf16 / int8 / int4 / mmap（int4 带宽最省，mmap 不占堆）
                case "--precision" -> config.precision = args[++i];
                case "--model-repo" -> config.modelRepo = args[++i];
                case "--mirror" -> config.mirror = args[++i];
                case "--max-seqs" -> config.maxNumSeqs = Integer.parseInt(args[++i]);
                case "--num-blocks" -> config.numBlocks = Integer.parseInt(args[++i]);
                // KV cache 存储精度：auto/f32（默认）或 int8（对照 vLLM --kv-cache-dtype）
                case "--kv-cache-dtype" -> config.kvCacheDtype = args[++i];
                // 投机采样草稿长度：0 关闭（默认），建议 3~5（对照 vLLM num_speculative_tokens）
                case "--speculative-k" -> config.speculativeK = Integer.parseInt(args[++i]);
                // 草稿模型：目录路径或已注册的模型 id（需与目标模型同词表；对照 vLLM --speculative-model）
                case "--draft-model" -> config.draftModel = args[++i];
                case "--quiet" -> config.verbose = false;
                default -> { }
            }
        }
        return config;
    }

    /** 权重精度解析：--precision 优先，--bf16 为旧版兼容开关 */
    private static Precision parsePrecision(ServerConfig config) {
        if (config.bf16) {
            return Precision.BF16;
        }
        return switch (config.precision.toLowerCase()) {
            case "f32" -> Precision.F32;
            case "bf16" -> Precision.BF16;
            case "int8" -> Precision.INT8;
            case "int4" -> Precision.INT4;
            case "mmap" -> Precision.MMAP;
            default -> throw new IllegalArgumentException(
                    "未知 --precision: " + config.precision + "（可选 f32/bf16/int8/int4/mmap）");
        };
    }

    private static List<String> splitDirs(String value) {
        List<String> dirs = new ArrayList<>();
        for (String part : value.split(",")) {
            if (!part.isBlank()) {
                dirs.add(part.trim());
            }
        }
        return dirs;
    }


    // ─── 模型目录解析 ───

    /**
     * 确定要注册的模型目录列表（首项为默认模型）：
     *   --model-dir 显式指定 → 扫描项目内 models/ → 自动下载 DEFAULT_MODEL_REPO。
     */
    private static List<Path> resolveModelDirs(ServerConfig config) throws Exception {
        if (!config.modelDirs.isEmpty()) {
            return config.modelDirs.stream().map(Path::of).toList();
        }
        List<Path> scanned = scanModelsRoot(Path.of(MODELS_ROOT));
        if (!scanned.isEmpty()) {
            System.out.println("扫描到可用模型目录: " + scanned);
            return scanned;
        }
        return List.of(new ModelDownloader(config.modelRepo, config.mirror).resolve());
    }

    /**
     * 扫描 models/ 下的 HF 模型目录（以 config.json 存在为凭）。
     * PREFERRED_MODEL 排在最前做默认模型：它最小、加载最快，适合启动预热。
     */
    private static List<Path> scanModelsRoot(Path root) throws IOException {
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        try (Stream<Path> entries = Files.list(root)) {
            return entries
                    .filter(dir -> Files.isRegularFile(dir.resolve("config.json")))
                    .sorted(Comparator
                            .comparing((Path dir) -> PREFERRED_MODEL.equals(dir.getFileName().toString()) ? 0 : 1)
                            .thenComparing(dir -> dir.getFileName().toString()))
                    .toList();
        }
    }


    // ─── 遗留学习路径（随机微模型 / GPT-3 预设，非 HF 模型目录） ───

    private static LoadedModel loadLegacyModel(ServerConfig config) throws Exception {
        ModelConfig cfg = config.gpt3 ? ModelConfig.gpt3Nano() : ModelConfig.small();

        TransformerModel gptModel;
        if (config.weightsPath != null && !config.random) {
            System.out.println("加载权重: " + config.weightsPath);
            Map<String, float[]> weights = SafetensorsLoader.load(Path.of(config.weightsPath));
            gptModel = ModelLoader.load(cfg, weights);
            System.out.println("模型加载完成: " + weights.size() + " 个张量");
        } else {
            gptModel = ModelLoader.randomInit(cfg);
            System.out.println("使用随机初始化模型（输出无意义，仅用于验证 PagedAttention + batching 流程）");
        }

        SimpleTokenizer tokenizer = config.tokenizerDir != null
                ? BpeTokenizer.fromModelDir(Path.of(config.tokenizerDir))
                : new ByteTokenizer();
        return new LoadedModel(cfg, gptModel, tokenizer, new PlainTextTemplate(),
                new int[0], cfg.dModel(), DEFAULT_LEARNING_MAX_SEQS);
    }


    // ─── HTTP 服务 ───

    private static void startHttpServer(ModelHub hub, int port) throws Exception {
        // 流式请求会阻塞 handler 线程，线程池按默认模型的并发上限放大
        ModelRuntime primary = hub.runtimeIfLoaded(hub.defaultId());
        int maxNumSeqs = primary != null ? primary.maxNumSeqs() : DEFAULT_LEARNING_MAX_SEQS;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/v1", new OpenAiHandler(hub));
        // /metrics 不在 /v1 前缀下，需单独注册（同一 handler 按 path 分发）
        server.createContext("/metrics", new OpenAiHandler(hub));
        server.createContext("/", new WebUiHandler());
        server.setExecutor(Executors.newFixedThreadPool(Math.max(16, maxNumSeqs * 4)));
        server.start();
    }


    // ─── 启动信息 ───

    private static void printRuntimeDiagnostics() {
        long maxHeapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        System.out.println("[runtime] " + io.leavesfly.minivllm.math.Matmul.diagnostics()
                + ", maxHeap=" + maxHeapMb + "MB");
        if ("scalar".equals(io.leavesfly.minivllm.math.Matmul.kernelName())) {
            System.out.println("[runtime] 警告: 使用标量内核，decode 会慢数倍。"
                    + "请用 java --add-modules jdk.incubator.vector 运行以启用 SIMD。");
        }
    }

    private static void printBanner(int port, ModelHub hub) {
        System.out.println("==================================================");
        System.out.println("  mini-vllm 学习型引擎已启动");
        System.out.println("  地址: http://localhost:" + port);
        System.out.println("  页面: http://localhost:" + port + "/  (对话演示，可在右上角切换模型)");
        System.out.println("  端点: POST /v1/chat/completions  (model 字段选模型)");
        System.out.println("        GET  /v1/models");
        System.out.println("        GET  /metrics  (引擎指标：TTFT/ITL/吞吐/KV 利用率)");
        System.out.println("  模型: (* 为默认)");
        for (String id : hub.ids()) {
            System.out.println("    " + (id.equals(hub.defaultId()) ? "*" : " ") + " "
                    + id + " — " + describe(hub.runtimeIfLoaded(id)));
        }
        System.out.println("==================================================");
        System.out.println("测试命令:");
        System.out.println("  curl -X POST http://localhost:" + port + "/v1/chat/completions \\");
        System.out.println("       -H 'Content-Type: application/json' \\");
        System.out.println("       -d '{\"model\":\"" + hub.defaultId()
                + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}'");
    }

    /** 已加载模型报告规模与并发，未加载模型则说明懒加载时机 */
    private static String describe(ModelRuntime runtime) {
        if (runtime == null) {
            return "待加载（首次选中时载入）";
        }
        return "已就绪, 参数量 " + runtime.loaded().model().numParameters()
                + ", maxSeqs=" + runtime.maxNumSeqs() + ", blocks=" + runtime.numBlocks()
                + ", " + runtime.loaded().config();
    }
}
