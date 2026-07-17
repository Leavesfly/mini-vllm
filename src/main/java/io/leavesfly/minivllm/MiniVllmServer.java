package io.leavesfly.minivllm;

import com.sun.net.httpserver.HttpServer;
import io.leavesfly.minivllm.core.LLMEngine;
import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.LlmModel;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.TransformerModel;
import io.leavesfly.minivllm.tokenizer.BpeTokenizer;
import io.leavesfly.minivllm.tokenizer.ByteTokenizer;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.weights.ModelDownloader;
import io.leavesfly.minivllm.weights.ModelLoader;
import io.leavesfly.minivllm.weights.Qwen3Loader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;
import io.leavesfly.minivllm.api.OpenAiHandler;
import io.leavesfly.minivllm.api.WebUiHandler;

import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.Executors;

/**
 * MiniVllmServer —— 学习型 vLLM 引擎入口。
 *
 * 启动流程：
 *   1. 解析参数（端口 / 权重路径 / 随机初始化 / 并发数 / block 数）
 *   2. 加载模型（safetensors 权重 或 随机初始化兜底）
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

    public static void main(String[] args) throws Exception {
        // ---------- 1. 解析参数 ----------
        int port = 8080;
        String weightsPath = null;
        String tokenizerDir = null;
        String modelDir = null;
        boolean random = false;
        int maxNumSeqs = -1;   // -1 = 按模式取默认值
        int numBlocks = -1;    // -1 = 自动估算
        int maxSeqLen = 2048;  // Qwen3 模式的上下文上限（RoPE 表与 KV 池按此分配）
        boolean verbose = true;
        boolean gpt3 = false;
        boolean bf16 = false;   // Qwen3 权重是否以 bf16 常驻（省一半内存/带宽）
        String modelRepo = "Qwen/Qwen3-0.6B";
        String mirror = System.getenv("MINIVLLM_MIRROR"); // auto | hf | modelscope

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port": port = Integer.parseInt(args[++i]); break;
                case "--weights": weightsPath = args[++i]; break;
                case "--tokenizer-dir": tokenizerDir = args[++i]; break;
                case "--model-dir": modelDir = args[++i]; break;
                case "--max-seq-len": maxSeqLen = Integer.parseInt(args[++i]); break;
                case "--random": random = true; break;
                case "--gpt3": gpt3 = true; break;
                case "--bf16": bf16 = true; break;
                case "--model-repo": modelRepo = args[++i]; break;
                case "--mirror": mirror = args[++i]; break;
                case "--max-seqs": maxNumSeqs = Integer.parseInt(args[++i]); break;
                case "--num-blocks": numBlocks = Integer.parseInt(args[++i]); break;
                case "--quiet": verbose = false; break;
                default: break;
            }
        }

        // ---------- 2. 加载模型与分词器 ----------
        // 运行时体检：确认 SIMD 内核是否生效、线程数、可用堆——decode 慢时先看这行
        long maxHeapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024);
        System.out.println("[runtime] " + io.leavesfly.minivllm.math.Matmul.diagnostics()
                + ", maxHeap=" + maxHeapMb + "MB");
        if ("scalar".equals(io.leavesfly.minivllm.math.Matmul.kernelName())) {
            System.out.println("[runtime] 警告: 使用标量内核，decode 会慢数倍。"
                    + "请用 java --add-modules jdk.incubator.vector 运行以启用 SIMD。");
        }
        // 未显式指定任何模式时，默认加载 Qwen3-0.6B：优先复用本地缓存，缺失则自动下载
        if (modelDir == null && !random && !gpt3 && weightsPath == null) {
            modelDir = new ModelDownloader(modelRepo, mirror).resolve().toString();
        }
        ModelConfig cfg;
        LlmModel model;
        SimpleTokenizer tokenizer;
        int[] eosTokens;
        int kvDim;

        if (modelDir != null) {
            // ===== Qwen3 模式：加载 HuggingFace 模型目录 =====
            Path dir = Path.of(modelDir);
            cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                    Files.readString(dir.resolve("config.json"))));
            cfg.maxSeqLen = Math.min(cfg.maxSeqLen, maxSeqLen); // 上限封顶，控制 RoPE 表与 KV 池
            if (maxNumSeqs < 0) {
                maxNumSeqs = 2; // Qwen3 默认并发（KV 池内存可控）
            }
            if (random) {
                model = Qwen3Loader.randomInit(cfg);
                System.out.println("使用随机初始化 Qwen3 模型（输出无意义，仅验证流程）");
            } else {
                System.out.println("加载权重: " + dir.resolve("model.safetensors")
                        + (bf16 ? " (bf16 常驻)" : " (f32 常驻)"));
                long t0 = System.currentTimeMillis();
                if (bf16) {
                    Map<String, short[]> weights = SafetensorsLoader.loadBf16Bits(dir.resolve("model.safetensors"));
                    System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                            weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
                    model = Qwen3Loader.loadBf16(cfg, weights);
                } else {
                    Map<String, float[]> weights = SafetensorsLoader.load(dir.resolve("model.safetensors"));
                    System.out.printf("权重读取完成: %d 个张量, %.1f s%n",
                            weights.size(), (System.currentTimeMillis() - t0) / 1000.0);
                    model = Qwen3Loader.load(cfg, weights);
                }
            }
            tokenizer = BpeTokenizer.fromModelDir(dir);
            eosTokens = cfg.eosTokenIds.length > 0 ? cfg.eosTokenIds : new int[]{151645, 151643};
            kvDim = cfg.kvDim();
            System.out.println("Qwen3 模型就绪: " + cfg.nLayer + " 层, 参数量 "
                    + model.numParameters() + ", vocab=" + tokenizer.vocabSize());
        } else {
            // ===== 学习模式：GPT-2/GPT-3 微模型 =====
            cfg = gpt3 ? ModelConfig.gpt3Nano() : ModelConfig.small();
            if (maxNumSeqs < 0) {
                maxNumSeqs = 8;
            }
            TransformerModel gptModel;
            if (weightsPath != null && !random) {
                System.out.println("加载权重: " + weightsPath);
                Map<String, float[]> weights = SafetensorsLoader.load(Path.of(weightsPath));
                gptModel = ModelLoader.load(cfg, weights);
                System.out.println("模型加载完成: " + weights.size() + " 个张量");
            } else {
                gptModel = ModelLoader.randomInit(cfg);
                System.out.println("使用随机初始化模型（输出无意义，仅用于验证 PagedAttention + batching 流程）");
            }
            model = gptModel;
            tokenizer = tokenizerDir != null
                    ? BpeTokenizer.fromModelDir(Path.of(tokenizerDir))
                    : new ByteTokenizer();
            eosTokens = new int[0];
            kvDim = cfg.dModel;
        }

        // ---------- 3. 构建 KV cache 内存池 ----------
        // numBlocks 需 >= maxNumSeqs * nLayer * (maxSeqLen/blockSize) 才能满载并发
        if (numBlocks < 0) {
            int blocksPerSeq = (cfg.maxSeqLen + cfg.blockSize - 1) / cfg.blockSize;
            numBlocks = Math.max(1024, maxNumSeqs * cfg.nLayer * blocksPerSeq);
        }
        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize, kvDim);
        long kvBytes = (long) numBlocks * cfg.blockSize * kvDim * 2 * 4;
        System.out.printf("KV 池: %d blocks × blockSize=%d × kvDim=%d（满载约 %.1f GB）%n",
                numBlocks, cfg.blockSize, kvDim, kvBytes / 1e9);

        // ---------- 4. 构建引擎并启动 continuous batching 循环 ----------
        LLMEngine engine = new LLMEngine(model, kvMgr, tokenizer, maxNumSeqs, eosTokens, 12345L);
        engine.setVerbose(verbose);
        engine.start();

        // ---------- 5. 启动 HTTP 服务 ----------
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/v1", new OpenAiHandler(engine, "mini-vllm")); // OpenAI 兼容 API
        server.createContext("/", new WebUiHandler());                       // 对话演示页面
        // 流式请求会阻塞 handler 线程，线程池需足够大
        server.setExecutor(Executors.newFixedThreadPool(Math.max(16, maxNumSeqs * 4)));
        server.start();

        System.out.println("==================================================");
        System.out.println("  mini-vllm 学习型引擎已启动");
        System.out.println("  地址: http://localhost:" + port);
        System.out.println("  页面: http://localhost:" + port + "/  (对话演示)");
        System.out.println("  端点: POST /v1/chat/completions");
        System.out.println("        GET  /v1/models");
        System.out.println("  配置: " + cfg);
        System.out.println("  参数量: " + model.numParameters());
        System.out.println("  并发: maxSeqs=" + maxNumSeqs + ", blocks=" + numBlocks);
        System.out.println("==================================================");
        System.out.println("测试命令:");
        System.out.println("  curl -X POST http://localhost:" + port + "/v1/chat/completions \\");
        System.out.println("       -H 'Content-Type: application/json' \\");
        System.out.println("       -d '{\"model\":\"mini-vllm\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}'");
    }
}
