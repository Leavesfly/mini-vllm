package io.leavesfly.minivllm.weights;

import io.leavesfly.minivllm.model.*;
import io.leavesfly.minivllm.math.LayerNorm;
import io.leavesfly.minivllm.model.TransformerModel;

import java.util.Map;
import java.util.Random;

/**
 * ModelLoader —— 把权重字典组装成 TransformerModel，或随机初始化一个可跑通的模型。
 *
 * 权重命名约定（nanoGPT 风格，分离的 q/k/v/o 投影）：
 *   wte.weight                 [vocab, d]
 *   wpe.weight                 [maxSeqLen, d]
 *   h.{i}.ln_1.weight/bias     [d]
 *   h.{i}.attn.q_proj.weight/bias  [d, d]   (行优先 [out, in]，与 PyTorch nn.Linear 一致)
 *   h.{i}.attn.k_proj.weight/bias  [d, d]
 *   h.{i}.attn.v_proj.weight/bias  [d, d]
 *   h.{i}.attn.o_proj.weight/bias  [d, d]
 *   h.{i}.mlp.fc1.weight/bias      [dFfn, d]
 *   h.{i}.mlp.fc2.weight/bias      [d, dFfn]
 *   ln_f.weight/bias            [d]
 *
 * 导出约定：若源模型用合并的 c_attn，导出 safetensors 前需拆分为 q/k/v/o 四个投影。
 *
 * 学习要点：
 * 1. 权重即一维 float[]，按行优先 reshape 成矩阵即可，无需复杂张量抽象。
 * 2. LayerNorm 的 gamma 初始化为 1、beta 为 0；Linear 的 bias 初始化为 0——这是标准初始化。
 * 3. randomInit 用 GPT-2/GPT-3 风格正态(std=0.02)初始化，残差投影(o_proj/fc2)按 modified init
 *    缩放到 std=0.02/sqrt(2*nLayer)；即便没有权重文件也能跑通整个推理流程（输出虽无意义）。
 * 4. GPT-3 开启 useSparseAttention 时，按 cfg.isSparseLayer(i) 逐层装配 dense/sparse 注意力。
 */
public final class ModelLoader {

    private ModelLoader() {
    }

    /** 从 safetensors 加载的权重字典构造 TransformerModel */
    public static TransformerModel load(ModelConfig cfg, Map<String, float[]> weights) {
        Embedding wteE = new Embedding(get(weights, "wte.weight", cfg.vocabSize * cfg.dModel),
                cfg.vocabSize, cfg.dModel);
        Embedding wpeE = new Embedding(get(weights, "wpe.weight", cfg.maxSeqLen * cfg.dModel),
                cfg.maxSeqLen, cfg.dModel);

        TransformerBlock[] blocks = new TransformerBlock[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            String p = "h." + i + ".";
            LayerNorm ln1 = layerNorm(weights, p + "ln_1", cfg.dModel, cfg.layerNormEps);
            LayerNorm ln2 = layerNorm(weights, p + "ln_2", cfg.dModel, cfg.layerNormEps);
            Linear q = linear(weights, p + "attn.q_proj", cfg.dModel, cfg.dModel);
            Linear k = linear(weights, p + "attn.k_proj", cfg.dModel, cfg.dModel);
            Linear v = linear(weights, p + "attn.v_proj", cfg.dModel, cfg.dModel);
            Linear o = linear(weights, p + "attn.o_proj", cfg.dModel, cfg.dModel);
            // GPT-3 交替：奇数层用局部带状稀疏注意力，偶数层 dense
            Attention attn = new Attention(cfg, q, k, v, o, cfg.isSparseLayer(i), cfg.sparseWindow);
            Linear fc1 = linear(weights, p + "mlp.fc1", cfg.dModel, cfg.dFfn);
            Linear fc2 = linear(weights, p + "mlp.fc2", cfg.dFfn, cfg.dModel);
            Ffn ffn = new Ffn(fc1, fc2);
            blocks[i] = new TransformerBlock(ln1, ln2, attn, ffn, cfg.dModel);
        }
        LayerNorm lnF = layerNorm(weights, "ln_f", cfg.dModel, cfg.layerNormEps);
        return new TransformerModel(cfg, wteE, wpeE, blocks, lnF);
    }

    /** 随机初始化（GPT-2 风格 std=0.02），无权重文件时用于跑通流程 */
    public static TransformerModel randomInit(ModelConfig cfg) {
        Random rnd = new Random(42L);
        float baseStd = 0.02f;                       // GPT-2/GPT-3 标准初始化标准差
        float resStd = cfg.residualInitStd(baseStd); // 残差投影 modified init：0.02/sqrt(2*nLayer)
        Embedding wteE = new Embedding(randN(rnd, cfg.vocabSize * cfg.dModel, 0.02f),
                cfg.vocabSize, cfg.dModel);
        Embedding wpeE = new Embedding(randN(rnd, cfg.maxSeqLen * cfg.dModel, 0.02f),
                cfg.maxSeqLen, cfg.dModel);

        TransformerBlock[] blocks = new TransformerBlock[cfg.nLayer];
        for (int i = 0; i < cfg.nLayer; i++) {
            LayerNorm ln1 = layerNormRand(rnd, cfg.dModel, cfg.layerNormEps);
            LayerNorm ln2 = layerNormRand(rnd, cfg.dModel, cfg.layerNormEps);
            Linear q = linearRand(rnd, cfg.dModel, cfg.dModel, baseStd);
            Linear k = linearRand(rnd, cfg.dModel, cfg.dModel, baseStd);
            Linear v = linearRand(rnd, cfg.dModel, cfg.dModel, baseStd);
            Linear o = linearRand(rnd, cfg.dModel, cfg.dModel, resStd); // 残差路径投影（modified init）
            // GPT-3 交替：奇数层用局部带状稀疏注意力，偶数层 dense
            Attention attn = new Attention(cfg, q, k, v, o, cfg.isSparseLayer(i), cfg.sparseWindow);
            Linear fc1 = linearRand(rnd, cfg.dModel, cfg.dFfn, baseStd);
            Linear fc2 = linearRand(rnd, cfg.dFfn, cfg.dModel, resStd);  // 残差路径投影（modified init）
            Ffn ffn = new Ffn(fc1, fc2);
            blocks[i] = new TransformerBlock(ln1, ln2, attn, ffn, cfg.dModel);
        }
        LayerNorm lnF = layerNormRand(rnd, cfg.dModel, cfg.layerNormEps);
        return new TransformerModel(cfg, wteE, wpeE, blocks, lnF);
    }

    // ---------- 辅助构造 ----------

    private static float[] get(Map<String, float[]> w, String name, int expect) {
        float[] d = w.get(name);
        if (d == null) {
            throw new IllegalArgumentException("缺少权重: " + name);
        }
        if (d.length != expect) {
            throw new IllegalArgumentException("权重 " + name + " 长度 " + d.length + " 期望 " + expect);
        }
        return d;
    }

    private static LayerNorm layerNorm(Map<String, float[]> w, String prefix, int dim, float eps) {
        float[] gamma = get(w, prefix + ".weight", dim);
        float[] beta = get(w, prefix + ".bias", dim);
        return new LayerNorm(gamma, beta, eps);
    }

    private static Linear linear(Map<String, float[]> w, String prefix, int in, int out) {
        float[] weight = get(w, prefix + ".weight", out * in); // [out, in]
        float[] bias = null;
        float[] b = w.get(prefix + ".bias");
        if (b != null && b.length == out) {
            bias = b;
        }
        return new Linear(weight, bias, in, out);
    }

    private static LayerNorm layerNormRand(Random rnd, int dim, float eps) {
        float[] gamma = new float[dim];
        float[] beta = new float[dim];
        for (int i = 0; i < dim; i++) gamma[i] = 1f; // gamma 初始化为 1
        return new LayerNorm(gamma, beta, eps);
    }

    private static Linear linearRand(Random rnd, int in, int out, float std) {
        float[] weight = randN(rnd, out * in, std);
        return new Linear(weight, new float[out], in, out); // bias 初始化为 0
    }

    /** Box-Muller 生成正态分布随机数 */
    private static float[] randN(Random rnd, int n, float std) {
        float[] r = new float[n];
        for (int i = 0; i < n; i += 2) {
            double u1 = Math.max(rnd.nextDouble(), 1e-12);
            double u2 = rnd.nextDouble();
            double radius = Math.sqrt(-2.0 * Math.log(u1));
            r[i] = (float) (radius * Math.cos(2 * Math.PI * u2) * std);
            if (i + 1 < n) {
                r[i + 1] = (float) (radius * Math.sin(2 * Math.PI * u2) * std);
            }
        }
        return r;
    }
}
