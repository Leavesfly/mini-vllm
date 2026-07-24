package io.leavesfly.minivllm;

import com.sun.net.httpserver.HttpServer;
import io.leavesfly.minivllm.core.LLMEngine;
import io.leavesfly.minivllm.family.LoadedModel;
import io.leavesfly.minivllm.family.ModelRegistry;
import io.leavesfly.minivllm.family.Precision;
import io.leavesfly.minivllm.memory.KVCacheManager;
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

import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * MiniVllmServer —— 学习型 vLLM 引擎入口。
 *
 * 启动流程：
 *   1. 解析参数（端口 / 权重路径 / 随机初始化 / 并发数 / block 数）
 *   2. 加载模型（--model-dir 经 ModelRegistry 按 model_type 分发到 ModelFamily；
 *      遗留学习路径 --random/--gpt3/--weights 直接构造微模型）
 *   3. 构建 KVCacheManager（PagedAttention 内存池）
 *   4. 构建 LLMEngine 并 start()（独立线程跑 continuous batching）
 *   5. 启动 JDK HttpServer，注册 OpenAI 兼容 API
 *
 * 用法示例：
 *   默认（加载 Qwen3-0.6B，本地无缓存时自动下载）：
 *     java -jar mini-vllm.jar --port 8080
 *   随机初始化微模型（无需权重，验证流程）：
 *     java -jar mini-vllm.jar --random --port 8080
 *   加载真实权重：
 *     java -jar mini-vllm.jar --weights ./model.safetensors --port 8080
 *   指定本地 Qwen3 模型目录：
 *     java -jar mini-vllm.jar --model-dir ./Qwen3-0.6B --port 8080
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
    private static final String DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B";
    private static final int DEFAULT_LEARNING_MAX_SEQS = 8;
    private static final int MIN_NUM_BLOCKS = 1024;


    // ─── 服务器配置（命令行参数解析结果） ───

    private static final class ServerConfig {
        int port = DEFAULT_PORT;
        String weightsPath = null;
        String tokenizerDir = null;
        String modelDir = null;
        boolean random = false;
        int maxNumSeqs = -1;
        int numBlocks = -1;
        int maxSeqLen = DEFAULT_MAX_SEQ_LEN;
        boolean verbose = true;
        boolean gpt3 = false;
        boolean bf16 = false;
        String modelRepo = DEFAULT_MODEL_REPO;
        String mirror = System.getenv("MINIVLLM_MIRROR");
    }

    public static void main(String[] args) throws Exception {
        ServerConfig config = parseArgs(args);
        printRuntimeDiagnostics();

        // 未显式指定任何模式时，默认加载 Qwen3-0.6B
        if (config.modelDir == null && !config.random && !config.gpt3 && config.weightsPath == null) {
            config.modelDir = new ModelDownloader(config.modelRepo, config.mirror).resolve().toString();
        }

        // HF 风格模型目录统一走 ModelRegistry（ServiceLoader 插件分发），
        // 遗留学习路径（随机微模型 / GPT-3 预设）保留直接构造
        LoadedModel loaded = config.modelDir != null
                ? new ModelRegistry().load(Path.of(config.modelDir),
                        config.bf16 ? Precision.BF16 : Precision.F32,
                        config.random, config.maxSeqLen)
                : loadLegacyModel(config);

        int maxNumSeqs = resolveMaxNumSeqs(config, loaded);
        KVCacheManager kvMgr = buildKvPool(loaded.config(), loaded.kvDim(),
                maxNumSeqs, config.numBlocks);
        LLMEngine engine = new LLMEngine(loaded.model(), kvMgr, loaded.tokenizer(),
                maxNumSeqs, loaded.eosTokens(), DEFAULT_SEED);
        engine.setVerbose(config.verbose);
        engine.start();

        startHttpServer(engine, loaded, config.port, maxNumSeqs);
        printBanner(config.port, loaded, maxNumSeqs, config.numBlocks);
    }


    // ─── 参数解析 ───

    private static ServerConfig parseArgs(String[] args) {
        ServerConfig config = new ServerConfig();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port" -> config.port = Integer.parseInt(args[++i]);
                case "--weights" -> config.weightsPath = args[++i];
                case "--tokenizer-dir" -> config.tokenizerDir = args[++i];
                case "--model-dir" -> config.modelDir = args[++i];
                case "--max-seq-len" -> config.maxSeqLen = Integer.parseInt(args[++i]);
                case "--random" -> config.random = true;
                case "--gpt3" -> config.gpt3 = true;
                case "--bf16" -> config.bf16 = true;
                case "--model-repo" -> config.modelRepo = args[++i];
                case "--mirror" -> config.mirror = args[++i];
                case "--max-seqs" -> config.maxNumSeqs = Integer.parseInt(args[++i]);
                case "--num-blocks" -> config.numBlocks = Integer.parseInt(args[++i]);
                case "--quiet" -> config.verbose = false;
                default -> { }
            }
        }
        return config;
    }

    /** 优先命令行 --max-seqs，否则用模型家族给出的建议并发数 */
    private static int resolveMaxNumSeqs(ServerConfig config, LoadedModel loaded) {
        return config.maxNumSeqs > 0 ? config.maxNumSeqs : loaded.defaultMaxSeqs();
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


    // ─── KV 池构建 ───

    private static KVCacheManager buildKvPool(ModelConfig cfg, int kvDim, int maxNumSeqs, int numBlocks) {
        if (numBlocks < 0) {
            int blocksPerSeq = (cfg.maxSeqLen() + cfg.blockSize() - 1) / cfg.blockSize();
            numBlocks = Math.max(MIN_NUM_BLOCKS, maxNumSeqs * cfg.nLayer() * blocksPerSeq);
        }
        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize(), kvDim);
        long kvBytes = (long) numBlocks * cfg.blockSize() * kvDim * 2 * 4;
        System.out.printf("KV 池: %d blocks × blockSize=%d × kvDim=%d（满载约 %.1f GB）%n",
                numBlocks, cfg.blockSize(), kvDim, kvBytes / 1e9);
        return kvMgr;
    }


    // ─── HTTP 服务 ───

    private static void startHttpServer(LLMEngine engine, LoadedModel loaded, int port,
                                        int maxNumSeqs) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/v1", new OpenAiHandler(engine, "mini-vllm", loaded.chatTemplate()));
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

    private static void printBanner(int port, LoadedModel loaded, int maxNumSeqs, int numBlocks) {
        System.out.println("==================================================");
        System.out.println("  mini-vllm 学习型引擎已启动");
        System.out.println("  地址: http://localhost:" + port);
        System.out.println("  页面: http://localhost:" + port + "/  (对话演示)");
        System.out.println("  端点: POST /v1/chat/completions");
        System.out.println("        GET  /v1/models");
        System.out.println("  配置: " + loaded.config());
        System.out.println("  参数量: " + loaded.model().numParameters());
        System.out.println("  并发: maxSeqs=" + maxNumSeqs + ", blocks=" + numBlocks);
        System.out.println("==================================================");
        System.out.println("测试命令:");
        System.out.println("  curl -X POST http://localhost:" + port + "/v1/chat/completions \\");
        System.out.println("       -H 'Content-Type: application/json' \\");
        System.out.println("       -d '{\"model\":\"mini-vllm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}'");
    }
}
