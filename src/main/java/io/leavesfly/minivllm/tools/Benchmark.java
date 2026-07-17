package io.leavesfly.minivllm.tools;

import io.leavesfly.minivllm.json.SimpleJson;
import io.leavesfly.minivllm.math.Matmul;
import io.leavesfly.minivllm.memory.BlockTable;
import io.leavesfly.minivllm.memory.KVCacheManager;
import io.leavesfly.minivllm.model.ModelConfig;
import io.leavesfly.minivllm.model.Qwen3Model;
import io.leavesfly.minivllm.weights.Qwen3Loader;
import io.leavesfly.minivllm.weights.SafetensorsLoader;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * Benchmark —— Qwen3-0.6B 推理性能基准。
 *
 * 测量 prefill（一次性处理整段 prompt）与 decode（逐 token 生成）的吞吐，
 * 并打印当前点积内核与线程数，便于对比优化效果。
 *
 * 用法（需 Vector API 模块与大堆）：
 *   java --add-modules jdk.incubator.vector -Xmx6g -cp target/classes \
 *        -Dmatmul.kernel=vector -Dmatmul.threads=8 \
 *        io.leavesfly.minivllm.tools.Benchmark [模型目录]
 *
 * 对比示例：
 *   - 单线程标量：-Dmatmul.kernel=scalar -Dmatmul.threads=1
 *   - 多线程向量：-Dmatmul.kernel=vector  -Dmatmul.threads=8
 */
public final class Benchmark {

    private static final int PREFILL_TOKENS = 64;
    private static final int DECODE_TOKENS = 16;
    private static final int WARMUP = 2;

    public static void main(String[] args) throws Exception {
        Path modelDir = resolveModelDir(args);
        if (modelDir == null) {
            System.err.println("未找到模型目录，用法: Benchmark [模型目录]");
            return;
        }
        System.out.println("内核: " + Matmul.kernelName() + ", 线程: " + Matmul.cores());

        ModelConfig cfg = ModelConfig.fromConfigJson(SimpleJson.parseObject(
                Files.readString(modelDir.resolve("config.json"))));
        cfg.maxSeqLen = 2048;
        System.out.println("加载权重...");
        long t0 = System.currentTimeMillis();
        Map<String, float[]> weights = SafetensorsLoader.load(modelDir.resolve("model.safetensors"));
        Qwen3Model model = Qwen3Loader.load(cfg, weights);
        System.out.printf("模型加载完成: %d 参数, %.1f s%n",
                model.numParameters(), (System.currentTimeMillis() - t0) / 1000.0);

        // 构造一段固定 prompt（用 token id 直接构造，不依赖 tokenizer）
        int[] prompt = new int[PREFILL_TOKENS];
        for (int i = 0; i < prompt.length; i++) {
            prompt[i] = 100 + i; // 任意合法 id
        }

        // warmup
        for (int i = 0; i < WARMUP; i++) {
            runDecode(model, cfg, prompt, DECODE_TOKENS / 2);
        }

        // prefill 吞吐
        long start = System.nanoTime();
        int iters = 5;
        for (int i = 0; i < iters; i++) {
            KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize, cfg.kvDim());
            BlockTable[] bts = newTables(cfg);
            ensure(kvMgr, bts, prompt.length + DECODE_TOKENS);
            model.prefillLogits(prompt, kvMgr, bts, 0);
        }
        double prefillSec = (System.nanoTime() - start) / 1e9 / iters;
        System.out.printf("prefill: %.1f tokens in %.3f s -> %.1f tok/s%n",
                (double) PREFILL_TOKENS, prefillSec, PREFILL_TOKENS / prefillSec);

        // decode 吞吐
        start = System.nanoTime();
        int totalDecode = 0;
        int decIters = 3;
        for (int i = 0; i < decIters; i++) {
            totalDecode += runDecode(model, cfg, prompt, DECODE_TOKENS);
        }
        double decodeSec = (System.nanoTime() - start) / 1e9;
        System.out.printf("decode:  %d tokens in %.3f s -> %.1f tok/s%n",
                totalDecode, decodeSec, totalDecode / decodeSec);
    }

    /** 返回实际生成的 token 数 */
    private static int runDecode(Qwen3Model model, ModelConfig cfg, int[] prompt, int n) {
        KVCacheManager kvMgr = new KVCacheManager(512, cfg.blockSize, cfg.kvDim());
        BlockTable[] bts = newTables(cfg);
        ensure(kvMgr, bts, prompt.length + n);
        float[] logits = model.prefillLogits(prompt, kvMgr, bts, 0);
        int count = 0;
        for (int t = 0; t < n; t++) {
            int next = argmax(logits);
            count++;
            if (t < n - 1) {
                logits = model.decodeLogits(next, prompt.length + t, kvMgr, bts);
            }
        }
        return count;
    }

    private static BlockTable[] newTables(ModelConfig cfg) {
        BlockTable[] bts = new BlockTable[cfg.nLayer];
        for (int i = 0; i < bts.length; i++) {
            bts[i] = new BlockTable();
        }
        return bts;
    }

    private static void ensure(KVCacheManager kvMgr, BlockTable[] bts, int total) {
        for (BlockTable bt : bts) {
            kvMgr.ensureCapacity(bt, total);
        }
    }

    private static int argmax(float[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }

    private static Path resolveModelDir(String[] args) {
        if (args.length > 0) {
            return Path.of(args[0]);
        }
        String prop = System.getProperty("qwen3.dir");
        if (prop != null) {
            return Path.of(prop);
        }
        Path snapshots = Path.of(System.getProperty("user.home"),
                ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
        if (Files.isDirectory(snapshots)) {
            try (var stream = Files.list(snapshots)) {
                return stream.filter(Files::isDirectory).findFirst().orElse(null);
            } catch (Exception e) {
                return null;
            }
        }
        return null;
    }
}
