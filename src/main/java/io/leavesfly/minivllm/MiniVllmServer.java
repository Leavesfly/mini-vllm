package io.leavesfly.minivllm;

import com.sun.net.httpserver.HttpServer;
import io.leavesfly.minivllm.core.LLMEngine;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.TransformerModel;
import io.leavesfly.minivllm.tokenizer.ByteTokenizer;
import io.leavesfly.minivllm.tokenizer.SimpleTokenizer;
import io.leavesfly.minivllm.weights.ModelLoader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;
import io.leavesfly.minivllm.api.OpenAiHandler;

import java.net.InetSocketAddress;
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
 *   随机初始化（无需权重，验证流程）：
 *     java -jar mini-vllm.jar --random --port 8080
 *   加载真实权重：
 *     java -jar mini-vllm.jar --weights ./model.safetensors --port 8080
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
        boolean random = false;
        int maxNumSeqs = 8;
        int numBlocks = 1024;
        boolean verbose = true;
        boolean gpt3 = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port": port = Integer.parseInt(args[++i]); break;
                case "--weights": weightsPath = args[++i]; break;
                case "--random": random = true; break;
                case "--gpt3": gpt3 = true; break;
                case "--max-seqs": maxNumSeqs = Integer.parseInt(args[++i]); break;
                case "--num-blocks": numBlocks = Integer.parseInt(args[++i]); break;
                case "--quiet": verbose = false; break;
                default: break;
            }
        }

        // ---------- 2. 加载模型 ----------
        // --gpt3：使用 GPT-3 风格小模型（开启交替 dense/稀疏注意力）；否则用 GPT-2 风格 small()
        ModelConfig cfg = gpt3 ? ModelConfig.gpt3Nano() : ModelConfig.small();
        TransformerModel model;
        if (weightsPath != null && !random) {
            System.out.println("加载权重: " + weightsPath);
            Map<String, float[]> weights = SafetensorsLoader.load(Path.of(weightsPath));
            model = ModelLoader.load(cfg, weights);
            System.out.println("模型加载完成: " + weights.size() + " 个张量");
        } else {
            model = ModelLoader.randomInit(cfg);
            System.out.println("使用随机初始化模型（输出无意义，仅用于验证 PagedAttention + batching 流程）");
        }

        // ---------- 3. 构建 KV cache 内存池 ----------
        // numBlocks 需 >= maxNumSeqs * nLayer * (maxSeqLen/blockSize) 才能满载并发
        SimpleTokenizer tokenizer = new ByteTokenizer();
        KVCacheManager kvMgr = new KVCacheManager(numBlocks, cfg.blockSize, cfg.dModel);

        // ---------- 4. 构建引擎并启动 continuous batching 循环 ----------
        LLMEngine engine = new LLMEngine(model, kvMgr, tokenizer, maxNumSeqs, -1, 12345L);
        engine.setVerbose(verbose);
        engine.start();

        // ---------- 5. 启动 HTTP 服务 ----------
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/", new OpenAiHandler(engine, "mini-vllm"));
        // 流式请求会阻塞 handler 线程，线程池需足够大
        server.setExecutor(Executors.newFixedThreadPool(Math.max(16, maxNumSeqs * 4)));
        server.start();

        System.out.println("==================================================");
        System.out.println("  mini-vllm 学习型引擎已启动");
        System.out.println("  地址: http://localhost:" + port);
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
